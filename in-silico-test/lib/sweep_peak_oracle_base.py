#!/usr/bin/env python3
"""Architectural sweep for ex/em peak regression — surrogate AND oracle, one sweep each.

The surrogate (guides design) and the oracle (independent evaluator) are selected under two
*different* protocols, so the sweep is run once per role:

  role        embedding          input dim   split column (dual_splits.csv)   output dir
  ---------   ----------------   ---------   ------------------------------   -----------------------
  surrogate   ESM-2 (esm2_650M)     1280      surrogate_role                  trained_models/surrogate_sweep
  oracle      ProstT5 (encoder)     1024      oracle_role                     trained_models/oracle_sweep

The surrogate stays on ESM-2 because generation embeds novel sequences with ESM-2; the oracle moves
to ProstT5 so it is independent of the surrogate in *both* the pLM and (via the sweep) architecture.

Each role sweeps the same three axes:

  backbone (how per-residue info is read):  mlp (pool-only) | cnn (Conv1d) | transformer (self-attn)
  pooling readout:                          mean | max | concat | concatstd | attn
  depth:                                    cnn n_conv in {1..4}; transformer nlayers in {1..4}

= 5 pools x (1 mlp + 4 cnn + 4 transformer) = 45 configs per role, plus a **covariance-probe (cov)
  pool on CNN depth 1-3** (z_i = Wᵀ x_i, C_z = (1/L) Σ z_i z_iᵀ) = 3 more -> 48 configs per role,
  **96 across both roles**, x 3 seeds. The cov configs are appended after the originals, so re-running
  a completed sweep only trains the 3 new CNN-cov fits per role (resume skips existing checkpoints).

Protocol (identical per role): targets standardized on that role's train split; standardized-peak MSE;
Adam(1e-3, wd 1e-4), batch 32; early stopping on val peak-MAE (nm, patience 20, max 200 epochs, best
weights restored). Selection is by mean val MAE across seeds; test is reported for reference only.

ALL fits are saved:  trained_models/<role>_sweep/<arch>-<pool>-d<depth>_s<seed>.pt
Per-role results:    trained_models/<role>_sweep/{results_per_fit.csv, results.csv, results.json}
Combined:            trained_models/peak_sweep_results_all.{csv,json}  (both roles, tagged role+emb)

The per-residue embedding caches are built elsewhere and must exist first:
    ESM-2   : python ../dataset_pipeline/embed.py --trait peak
    ProstT5 : python embed_prostt5.py

Usage:
    python sweep_peak_oracle.py --role both        # 90 configs (surrogate ESM + oracle ProstT5)
    python sweep_peak_oracle.py --role surrogate   # 45 configs, ESM-2 on the surrogate split
    python sweep_peak_oracle.py --role oracle       # 45 configs, ProstT5 on the oracle split
    python sweep_peak_oracle.py --role both --dry-run     # list configs, no training
    python sweep_peak_oracle.py --role oracle --limit 2   # first 2 configs (smoke)
    python sweep_peak_oracle.py --role both --max-epochs 50 --seeds 0
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch

import peak_models as pm

HERE = os.path.dirname(os.path.abspath(__file__))
CUR = os.path.join(HERE, "..", "data")        # experiment root: ./data (dataset + dual_splits.csv)
OUT_BASE = os.path.join(HERE, "..", "trained_models")
POOLS = ["mean", "max", "concat", "concatstd", "attn"]
BS, LR, WD = 32, 1e-3, 1e-4

# per-role embedding backend + split column + output subdir
ROLE_CFG = {
    "surrogate": {"emb": "esm",     "cache": "esm_residue_fp16.npy",     "len": "esm_residue_len.npy",
                  "d_in": 1280, "split_col": "surrogate_role", "out": "surrogate_sweep"},
    "oracle":    {"emb": "prostt5", "cache": "prostt5_residue_fp16.npy", "len": "prostt5_residue_len.npy",
                  "d_in": 1024, "split_col": "oracle_role",    "out": "oracle_sweep"},
}
ROLES = list(ROLE_CFG)

_BUILD_HINT = ("Build the per-residue caches first:\n"
               "  ESM-2   : python ../dataset_pipeline/embed.py --trait peak\n"
               "  ProstT5 : python embed_prostt5.py")


def device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


COV_PROBE_DIM = 32          # probe width p for the covariance-probe pool (out_dim = p(p+1)/2)


def make_configs():
    cfgs = []
    for pool in POOLS:
        cfgs.append({"arch": "mlp", "pool": pool, "depth": 0})
        for d in (1, 2, 3, 4):
            cfgs.append({"arch": "cnn", "pool": pool, "n_conv": d, "depth": d})
        for d in (1, 2, 3, 4):
            cfgs.append({"arch": "transformer", "pool": pool, "nlayers": d, "depth": d})
    # Covariance-probe pool (z_i = Wᵀ x_i, C_z = (1/L) Σ z_i z_iᵀ) added for CNN depth 1-3 only.
    # Appended separately so existing configs/checkpoints are untouched (resume skips them).
    for d in (1, 2, 3):
        cfgs.append({"arch": "cnn", "pool": "cov", "n_conv": d, "depth": d, "probe_dim": COV_PROBE_DIM})
    return cfgs


def label(spec):
    return f"{spec['arch']}-{spec['pool']}-d{spec['depth']}"


def out_dir(role):
    return os.path.join(OUT_BASE, ROLE_CFG[role]["out"])


def _load_npy(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} cache not found: {path}\n{_BUILD_HINT}")
    return np.load(path)


def load_data(role, dev, cur=CUR, to_gpu=False):
    """Load peaks + the role's embedding cache + the role-specific train/val/test split.

    to_gpu=True preloads the (N,Lmax,d_in) fp16 embedding tensor onto `dev` once, so `batches`
    slices/upcasts on-device instead of copying every batch from host each epoch (big speedup).
    Left False by default so CPU-loop consumers (e.g. sweep_results.ipynb) keep working.
    """
    cfg = ROLE_CFG[role]
    rows = list(csv.DictReader(open(os.path.join(cur, "peaks_assignments.csv"))))
    rows.sort(key=lambda r: int(r["index"]))
    names = [r["name"] for r in rows]
    peaks = np.load(os.path.join(cur, "peaks.npy")).astype(np.float32)              # (N,2) nm
    H = _load_npy(os.path.join(cur, cfg["cache"]), f"{cfg['emb']} embedding")        # (N,Lmax,d_in) fp16
    Ls = _load_npy(os.path.join(cur, cfg["len"]), f"{cfg['emb']} length").astype(np.int64)
    role_map = {int(r["index"]): r[cfg["split_col"]]
                for r in csv.DictReader(open(os.path.join(cur, "dual_splits.csv")))}
    N = len(rows)
    assert H.shape[0] == N == len(peaks) == len(Ls), "cache/target/split row mismatch"
    assert H.shape[2] == cfg["d_in"], f"{cfg['emb']} cache dim {H.shape[2]} != expected {cfg['d_in']}"
    idx = {k: np.array([i for i in range(N) if role_map[i] == k]) for k in ("train", "val", "test")}
    mean = peaks[idx["train"]].mean(0); std = peaks[idx["train"]].std(0) + 1e-6
    Lmax = int(H.shape[1])
    Ht = torch.tensor(H)                                 # (N,Lmax,d_in) fp16
    ar = torch.arange(Lmax)
    if to_gpu:                                           # preload once -> no per-batch host->device copy
        Ht = Ht.to(dev); ar = ar.to(dev)
    return {
        "role": role, "emb": cfg["emb"], "d_in": cfg["d_in"],
        "names": names, "N": N, "Lmax": Lmax,
        "Ht": Ht, "Ls": Ls, "ar": ar,
        "Pk": torch.tensor(peaks, device=dev),
        "mean": mean.astype(np.float32), "std": std.astype(np.float32),
        "sd": torch.tensor(std, device=dev),
        "tr": idx["train"], "va": idx["val"], "te": idx["test"],
    }


def batches(D, idx, dev, shuffle=False, rng=None):
    idx = np.array(idx)
    if shuffle:
        rng.shuffle(idx)
    hdev = D["Ht"].device                                # embeddings may be preloaded on the GPU
    for i in range(0, len(idx), BS):
        b = idx[i:i + BS]
        bt = torch.as_tensor(b, device=hdev)
        mk = D["ar"].unsqueeze(0) < torch.as_tensor(D["Ls"][b], device=hdev).unsqueeze(1)
        Hb = D["Ht"][bt].float()
        if hdev != dev:                                  # CPU cache -> copy this batch to the compute device
            Hb = Hb.to(dev); mk = mk.to(dev)
        yield Hb, mk, b


@torch.no_grad()
def eval_mae(net, D, idx, dev):
    net.eval(); ps = []
    for Hb, mk, b in batches(D, idx, dev):
        ps.append(net(Hb, mk).cpu().numpy())
    P = np.concatenate(ps); T = D["Pk"].cpu().numpy()[idx]
    ad = np.abs(P - T)
    return float(ad.mean()), float(ad[:, 0].mean()), float(ad[:, 1].mean())


def train_eval(spec, seed, D, dev, max_epochs=200, patience=20):
    torch.manual_seed(seed); rng = np.random.default_rng(100 + seed)
    # d_in comes from the role's embedding backend (ESM 1280 / ProstT5 1024)
    base = pm.build_base({**spec, "d_in": D["d_in"]}, dev, out=2, drop=spec.get("drop", 0.2))
    net = pm.wrap(base, D["mean"], D["std"], dev)
    opt = torch.optim.Adam(net.parameters(), LR, weight_decay=WD)
    best = {"val": float("inf"), "state": None, "epoch": -1}
    bad = 0
    for ep in range(max_epochs):
        net.train()
        for Hb, mk, b in batches(D, D["tr"], dev, shuffle=True, rng=rng):
            opt.zero_grad()
            (((net(Hb, mk) - D["Pk"][torch.as_tensor(b, device=dev)]) / D["sd"]) ** 2).mean().backward()
            opt.step()
        v = eval_mae(net, D, D["va"], dev)[0]
        if v < best["val"] - 1e-4:
            best = {"val": v, "state": copy.deepcopy({k: x.cpu() for k, x in net.base.state_dict().items()}),
                    "epoch": ep}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.base.load_state_dict(best["state"])
    va = eval_mae(net, D, D["va"], dev)
    te = eval_mae(net, D, D["te"], dev)
    m = {"val_mae": va[0], "val_ex": va[1], "val_em": va[2],
         "test_mae": te[0], "test_ex": te[1], "test_em": te[2],
         "epoch_best": best["epoch"], "n_train": int(len(D["tr"]))}
    return m, net.base


MK = ("val_mae", "val_ex", "val_em", "test_mae", "test_ex", "test_em", "epoch_best", "n_train")


def run_sweep(role, cfgs, seeds, dev, max_epochs, patience, force):
    """Train every (config, seed) for one role; write per-role results; return its summary rows."""
    D = load_data(role, dev, to_gpu=True)
    out = out_dir(role); os.makedirs(out, exist_ok=True)
    print(f"\n=== {role} sweep | emb={D['emb']} d_in={D['d_in']} | "
          f"split {role}_role: train={len(D['tr'])} val={len(D['va'])} test={len(D['te'])} "
          f"| N={D['N']} Lmax={D['Lmax']} ===", flush=True)

    per_fit = []
    for ci, spec in enumerate(cfgs):
        for seed in seeds:
            path = os.path.join(out, f"{label(spec)}_s{seed}.pt")
            if os.path.exists(path) and not force:                          # resume: reuse completed fit
                ck = torch.load(path, map_location="cpu", weights_only=False)
                m = {k: ck[k] for k in MK}; tag = "cached "
            else:
                m, base = train_eval(spec, seed, D, dev, max_epochs, patience)
                pm.save_model(path, base, {**spec, "d_in": D["d_in"], "out": 2, "seed": seed,
                                           "role": role, "emb": D["emb"],
                                           "mean": D["mean"], "std": D["std"], **m})
                tag = "trained"
            per_fit.append({"role": role, "emb": D["emb"],
                            **{k: spec.get(k) for k in ("arch", "pool", "depth")}, "seed": seed, **m,
                            "label": label(spec), "ckpt": os.path.relpath(path, HERE)})
            print(f"[{role[:4]} {ci+1:2}/{len(cfgs)}] {tag} {label(spec):22} s{seed} "
                  f"val {m['val_mae']:5.2f} (ex {m['val_ex']:4.1f} em {m['val_em']:4.1f})  "
                  f"test {m['test_mae']:5.2f}  @ep{m['epoch_best']}", flush=True)

    # aggregate per config (mean +/- std over seeds), rank by mean val MAE
    agg = defaultdict(list)
    for r in per_fit:
        agg[r["label"]].append(r)
    summary = []
    for lab, rs in agg.items():
        vm = np.array([r["val_mae"] for r in rs]); tm = np.array([r["test_mae"] for r in rs])
        r0 = rs[0]
        summary.append({"role": role, "emb": D["emb"], "label": lab,
                        "arch": r0["arch"], "pool": r0["pool"], "depth": r0["depth"],
                        "val_mae_mean": float(vm.mean()), "val_mae_std": float(vm.std()),
                        "test_mae_mean": float(tm.mean()), "test_mae_std": float(tm.std()),
                        "n_seeds": len(rs)})
    summary.sort(key=lambda s: s["val_mae_mean"])

    with open(os.path.join(out, "results_per_fit.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_fit[0].keys())); w.writeheader(); w.writerows(per_fit)
    with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    json.dump({"role": role, "emb": D["emb"], "summary": summary, "per_fit": per_fit},
              open(os.path.join(out, "results.json"), "w"), indent=2)

    print(f"\n--- {role} leaderboard (by mean val MAE, nm) ---")
    print(f"{'config':24} {'val_MAE':>14} {'test_MAE':>14}")
    for s in summary[:15]:
        print(f"{s['label']:24} {s['val_mae_mean']:6.2f} ± {s['val_mae_std']:4.2f}   "
              f"{s['test_mae_mean']:6.2f} ± {s['test_mae_std']:4.2f}")
    best = summary[0]
    print(f"best {role} ({D['emb']}): {best['label']}  "
          f"val {best['val_mae_mean']:.2f} ± {best['val_mae_std']:.2f} nm | "
          f"test {best['test_mae_mean']:.2f} ± {best['test_mae_std']:.2f} nm")
    return summary


def write_combined():
    """Merge whatever per-role results.csv files exist into a combined leaderboard."""
    rows = []
    for role in ROLES:
        p = os.path.join(out_dir(role), "results.csv")
        if os.path.exists(p):
            rows.extend(list(csv.DictReader(open(p))))
    if not rows:
        return
    csv_path = os.path.join(OUT_BASE, "peak_sweep_results_all.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(os.path.join(OUT_BASE, "peak_sweep_results_all.json"), "w"), indent=2)
    print(f"\ncombined results ({len(rows)} configs across roles) -> {os.path.relpath(csv_path, HERE)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", required=True, choices=ROLES + ["both"],
                    help="which sweep to run: surrogate (ESM-2), oracle (ProstT5), or both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="only the first N configs per role (smoke)")
    ap.add_argument("--pools", nargs="+", default=None,
                    help="only configs whose pool is in this list (e.g. --pools cov to add just the new "
                         "covariance-probe CNNs without touching the rest of the sweep)")
    ap.add_argument("--force", action="store_true", help="retrain fits even if a checkpoint exists")
    ap.add_argument("--dry-run", action="store_true", help="list configs, no training")
    a = ap.parse_args()

    roles = ROLES if a.role == "both" else [a.role]
    cfgs = make_configs()
    if a.pools:
        cfgs = [c for c in cfgs if c["pool"] in a.pools]
    if a.limit:
        cfgs = cfgs[:a.limit]

    if a.dry_run:
        for c in cfgs:
            print(" ", label(c), c)
        for role in roles:
            print(f"[{role:9}] emb={ROLE_CFG[role]['emb']:8} split={ROLE_CFG[role]['split_col']:14} "
                  f"-> {len(cfgs)} configs x {len(a.seeds)} seeds = {len(cfgs) * len(a.seeds)} fits")
        print(f"total: {len(cfgs) * len(a.seeds) * len(roles)} fits across {len(roles)} role(s)")
        return

    dev = device(); print(f"device: {dev}")
    for role in roles:
        run_sweep(role, cfgs, a.seeds, dev, a.max_epochs, a.patience, a.force)
    write_combined()


if __name__ == "__main__":
    main()
