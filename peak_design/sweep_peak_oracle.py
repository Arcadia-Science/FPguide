#!/usr/bin/env python3
"""Architectural sweep for ex/em peak regression on the ORACLE split.

Reads the curated peak dataset + precomputed ESM-2 embeddings from the dataset_pipeline, trains on
the oracle train/val/test split (dual_splits.csv `oracle_role`), and sweeps three axes:

  backbone (how per-residue info is read):  mlp (pool-only) | cnn (Conv1d) | transformer (self-attn)
  pooling readout:                          mean | max | concat | concatstd | attn
  depth:                                    cnn n_conv in {1..4}; transformer nlayers in {1..4}

= 5 pools x (1 mlp + 4 cnn + 4 transformer) = 45 configs, x 3 seeds = 135 fits.

Protocol: targets standardized on oracle_train; standardized-peak MSE; Adam(1e-3, wd 1e-4), batch 32;
early stopping on val peak-MAE (nm, patience 20, max 200 epochs, best weights restored).
Selection is by mean val MAE across seeds; oracle_test is reported for reference only.

ALL fits are saved: trained_models/oracle_sweep/<arch>-<pool>-d<depth>_s<seed>.pt
Results: trained_models/oracle_sweep/{results_per_fit.csv, results.csv, results.json} + a printed leaderboard.

Usage:
    python sweep_peak_oracle.py                 # full sweep
    python sweep_peak_oracle.py --dry-run       # list the 45 configs, no training
    python sweep_peak_oracle.py --limit 2       # first 2 configs (smoke)
    python sweep_peak_oracle.py --max-epochs 50 --seeds 0
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os

import numpy as np
import torch

import peak_models as pm

HERE = os.path.dirname(os.path.abspath(__file__))
CUR = os.path.join(HERE, "..", "dataset_pipeline", "data", "peak", "curated")
OUT = os.path.join(HERE, "trained_models", "oracle_sweep")
POOLS = ["mean", "max", "concat", "concatstd", "attn"]
BS, LR, WD = 32, 1e-3, 1e-4


def device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def make_configs():
    cfgs = []
    for pool in POOLS:
        cfgs.append({"arch": "mlp", "pool": pool, "depth": 0})
        for d in (1, 2, 3, 4):
            cfgs.append({"arch": "cnn", "pool": pool, "n_conv": d, "depth": d})
        for d in (1, 2, 3, 4):
            cfgs.append({"arch": "transformer", "pool": pool, "nlayers": d, "depth": d})
    return cfgs


def label(spec):
    return f"{spec['arch']}-{spec['pool']}-d{spec['depth']}"


def load_data(dev, cur=CUR):
    rows = list(csv.DictReader(open(os.path.join(cur, "peaks_assignments.csv"))))
    rows.sort(key=lambda r: int(r["index"]))
    names = [r["name"] for r in rows]
    peaks = np.load(os.path.join(cur, "peaks.npy")).astype(np.float32)          # (N,2) nm
    H = np.load(os.path.join(cur, "esm_residue_fp16.npy"))                       # (N,Lmax,1280) fp16
    Ls = np.load(os.path.join(cur, "esm_residue_len.npy")).astype(np.int64)
    role = {int(r["index"]): r["oracle_role"] for r in csv.DictReader(open(os.path.join(cur, "dual_splits.csv")))}
    N = len(rows)
    idx = {k: np.array([i for i in range(N) if role[i] == k]) for k in ("train", "val", "test")}
    mean = peaks[idx["train"]].mean(0); std = peaks[idx["train"]].std(0) + 1e-6
    Lmax = int(H.shape[1])
    return {
        "names": names, "N": N, "Lmax": Lmax,
        "Ht": torch.tensor(H), "Ls": Ls, "ar": torch.arange(Lmax),
        "Pk": torch.tensor(peaks, device=dev),
        "mean": mean.astype(np.float32), "std": std.astype(np.float32),
        "sd": torch.tensor(std, device=dev),
        "tr": idx["train"], "va": idx["val"], "te": idx["test"],
    }


def batches(D, idx, dev, shuffle=False, rng=None):
    idx = np.array(idx)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), BS):
        b = idx[i:i + BS]
        mk = (D["ar"].unsqueeze(0) < torch.tensor(D["Ls"][b]).unsqueeze(1)).to(dev)
        yield D["Ht"][b].float().to(dev), mk, b


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
    base = pm.build_base(spec, dev, out=2, drop=spec.get("drop", 0.2))
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="only the first N configs (smoke)")
    ap.add_argument("--force", action="store_true", help="retrain fits even if a checkpoint exists")
    ap.add_argument("--dry-run", action="store_true", help="list configs, no training")
    a = ap.parse_args()

    cfgs = make_configs()
    if a.limit:
        cfgs = cfgs[:a.limit]
    if a.dry_run:
        for c in cfgs:
            print(" ", label(c), c)
        print(f"{len(cfgs)} configs x {len(a.seeds)} seeds = {len(cfgs) * len(a.seeds)} fits")
        return

    dev = device(); print(f"device: {dev}")
    os.makedirs(OUT, exist_ok=True)
    D = load_data(dev)
    print(f"oracle split: train={len(D['tr'])} val={len(D['va'])} test={len(D['te'])} | N={D['N']} Lmax={D['Lmax']}")

    MK = ("val_mae", "val_ex", "val_em", "test_mae", "test_ex", "test_em", "epoch_best", "n_train")
    per_fit = []
    for ci, spec in enumerate(cfgs):
        for seed in a.seeds:
            path = os.path.join(OUT, f"{label(spec)}_s{seed}.pt")
            if os.path.exists(path) and not a.force:                       # resume: reuse completed fit
                ck = torch.load(path, map_location="cpu", weights_only=False)
                m = {k: ck[k] for k in MK}; tag = "cached "
            else:
                m, base = train_eval(spec, seed, D, dev, a.max_epochs, a.patience)
                pm.save_model(path, base, {**spec, "out": 2, "seed": seed,
                                           "mean": D["mean"], "std": D["std"], **m})
                tag = "trained"
            per_fit.append({**{k: spec.get(k) for k in ("arch", "pool", "depth")}, "seed": seed, **m,
                            "label": label(spec), "ckpt": os.path.relpath(path, HERE)})
            print(f"[{ci+1:2}/{len(cfgs)}] {tag} {label(spec):22} s{seed} "
                  f"val {m['val_mae']:5.2f} (ex {m['val_ex']:4.1f} em {m['val_em']:4.1f})  "
                  f"test {m['test_mae']:5.2f}  @ep{m['epoch_best']}", flush=True)

    # aggregate per config (mean +/- std over seeds), rank by mean val MAE
    from collections import defaultdict
    agg = defaultdict(list)
    for r in per_fit:
        agg[r["label"]].append(r)
    summary = []
    for lab, rs in agg.items():
        vm = np.array([r["val_mae"] for r in rs]); tm = np.array([r["test_mae"] for r in rs])
        r0 = rs[0]
        summary.append({"label": lab, "arch": r0["arch"], "pool": r0["pool"], "depth": r0["depth"],
                        "val_mae_mean": float(vm.mean()), "val_mae_std": float(vm.std()),
                        "test_mae_mean": float(tm.mean()), "test_mae_std": float(tm.std()),
                        "n_seeds": len(rs)})
    summary.sort(key=lambda s: s["val_mae_mean"])

    with open(os.path.join(OUT, "results_per_fit.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_fit[0].keys())); w.writeheader(); w.writerows(per_fit)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    json.dump({"summary": summary, "per_fit": per_fit}, open(os.path.join(OUT, "results.json"), "w"), indent=2)

    print("\n=== leaderboard (by mean val MAE, nm) ===")
    print(f"{'config':24} {'val_MAE':>14} {'test_MAE':>14}")
    for s in summary[:15]:
        print(f"{s['label']:24} {s['val_mae_mean']:6.2f} ± {s['val_mae_std']:4.2f}   "
              f"{s['test_mae_mean']:6.2f} ± {s['test_mae_std']:4.2f}")
    best = summary[0]
    print(f"\nbest by val MAE: {best['label']}  "
          f"val {best['val_mae_mean']:.2f} ± {best['val_mae_std']:.2f} nm | "
          f"test {best['test_mae_mean']:.2f} ± {best['test_mae_std']:.2f} nm")


if __name__ == "__main__":
    main()
