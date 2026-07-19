#!/usr/bin/env python3
"""Retrain the peak Oracle on ProstT5 embeddings (drop-in replacement for the ESM-2 oracle).

The oracle regresses ``(ex_max, em_max)`` (nm) from frozen per-residue embeddings. The surrogate that
*guides* design still runs on ESM-2; only the independent *evaluator* (oracle) is switched here to the
ProstT5 encoder (1024-dim, structure-aware) instead of ESM-2 (1280-dim).

Same architecture and training protocol as the ESM-2 oracle selected by ``sweep_peak_oracle.py``
(``cnn-concatstd-d1``): standardized-peak MSE, Adam(1e-3, wd 1e-4), batch 32, early stopping on val
peak-MAE (patience 20, max 200 epochs, best weights restored). Trained on the dual **oracle** split
(``dual_splits.csv`` ``oracle_role``). ``--emb {prostt5,esm}`` selects the embedding cache and input
dim, so the two backends can be compared head-to-head under an identical protocol.

Saves the best-val-seed model:
    trained_models/dual_oracle_<emb>_net.pt        (checkpoint; carries d_in so it reconstructs)
    trained_models/dual_oracle_<emb>_scaler.npz    (train-split peak mean/std, for StandardizedPeaks)

Usage:
    python train_oracle_prostt5.py                     # ProstT5 oracle, seeds 0 1 2
    python train_oracle_prostt5.py --emb esm           # ESM-2 baseline, identical protocol
    python train_oracle_prostt5.py --seeds 0 --max-epochs 50   # quick smoke
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
OUT = os.path.join(HERE, "trained_models")

# cache file + input dim per embedding backend
EMB = {
    "prostt5": {"cache": "prostt5_residue_fp16.npy", "len": "prostt5_residue_len.npy", "d_in": 1024},
    "esm":     {"cache": "esm_residue_fp16.npy",     "len": "esm_residue_len.npy",     "d_in": 1280},
}
# the oracle architecture selected by sweep_peak_oracle.py (best val MAE on ESM-2)
ORACLE_SPEC = dict(arch="cnn", pool="concatstd", conv_ch=128, k=5, n_conv=1, hidden=256, nl=2)
BS, LR, WD, DROP = 32, 1e-3, 1e-4, 0.2


def device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_data(emb, dev):
    cfg = EMB[emb]
    rows = list(csv.DictReader(open(os.path.join(CUR, "peaks_assignments.csv"))))
    rows.sort(key=lambda r: int(r["index"]))
    peaks = np.load(os.path.join(CUR, "peaks.npy")).astype(np.float32)             # (N,2) nm
    H = np.load(os.path.join(CUR, cfg["cache"]))                                   # (N,Lmax,d_in) fp16
    Ls = np.load(os.path.join(CUR, cfg["len"])).astype(np.int64)
    role = {int(r["index"]): r["oracle_role"]
            for r in csv.DictReader(open(os.path.join(CUR, "dual_splits.csv")))}
    N = len(rows)
    assert H.shape[0] == N == len(peaks) == len(Ls), "cache/target/split row mismatch"
    assert H.shape[2] == cfg["d_in"], f"{emb} cache dim {H.shape[2]} != expected {cfg['d_in']}"
    idx = {k: np.array([i for i in range(N) if role[i] == k]) for k in ("train", "val", "test")}
    mean = peaks[idx["train"]].mean(0).astype(np.float32)
    std = (peaks[idx["train"]].std(0) + 1e-6).astype(np.float32)
    Lmax = int(H.shape[1])
    return {
        "d_in": cfg["d_in"], "N": N, "Lmax": Lmax,
        "Ht": torch.tensor(H), "Ls": Ls, "ar": torch.arange(Lmax),
        "Pk": torch.tensor(peaks, device=dev), "peaks": peaks,
        "mean": mean, "std": std, "sd": torch.tensor(std, device=dev),
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
    P = np.concatenate(ps); T = D["peaks"][idx]
    ad = np.abs(P - T)
    return float(ad.mean()), float(ad[:, 0].mean()), float(ad[:, 1].mean())


def train_one(spec, seed, D, dev, max_epochs=200, patience=20):
    torch.manual_seed(seed); rng = np.random.default_rng(100 + seed)
    base = pm.build_base(spec, dev, out=2, drop=DROP)
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
    va = eval_mae(net, D, D["va"], dev); te = eval_mae(net, D, D["te"], dev)
    m = {"val_mae": va[0], "val_ex": va[1], "val_em": va[2],
         "test_mae": te[0], "test_ex": te[1], "test_em": te[2], "epoch_best": best["epoch"]}
    return m, net.base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", choices=list(EMB), default="prostt5")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    a = ap.parse_args()

    dev = device()
    D = load_data(a.emb, dev)
    spec = {**ORACLE_SPEC, "d_in": D["d_in"], "sid": f"cnn-concatstd-d1-{a.emb}"}
    print(f"emb={a.emb} d_in={D['d_in']} | device={dev} | "
          f"oracle split: train={len(D['tr'])} val={len(D['va'])} test={len(D['te'])} Lmax={D['Lmax']}")

    fits = []
    best = {"val": float("inf")}
    for seed in a.seeds:
        m, base = train_one(spec, seed, D, dev, a.max_epochs, a.patience)
        fits.append({"seed": seed, **m})
        print(f"  seed {seed}: val {m['val_mae']:5.2f} (ex {m['val_ex']:4.1f} em {m['val_em']:4.1f})  "
              f"test {m['test_mae']:5.2f} (ex {m['test_ex']:4.1f} em {m['test_em']:4.1f})  @ep{m['epoch_best']}",
              flush=True)
        if m["val_mae"] < best["val"]:
            best = {"val": m["val_mae"], "seed": seed, "base": copy.deepcopy(base), "m": m}

    os.makedirs(OUT, exist_ok=True)
    net_path = os.path.join(OUT, f"dual_oracle_{a.emb}_net.pt")
    scaler_path = os.path.join(OUT, f"dual_oracle_{a.emb}_scaler.npz")
    pm.save_model(net_path, best["base"], {**spec, "out": 2, "seed": best["seed"],
                                           "mean": D["mean"], "std": D["std"], **best["m"]})
    np.savez(scaler_path, mean=D["mean"], std=D["std"])

    vm = np.array([f["val_mae"] for f in fits]); tm = np.array([f["test_mae"] for f in fits])
    print(f"\n=== {a.emb} oracle (cnn-concatstd-d1) over {len(fits)} seed(s) ===")
    print(f"val  MAE {vm.mean():.2f} ± {vm.std():.2f} nm   test MAE {tm.mean():.2f} ± {tm.std():.2f} nm")
    print(f"saved best-val (seed {best['seed']}, val {best['val']:.2f}) -> "
          f"{os.path.relpath(net_path, HERE)}  test MAE {best['m']['test_mae']:.2f} nm")
    json.dump({"emb": a.emb, "spec": spec, "fits": fits,
               "val_mae_mean": float(vm.mean()), "val_mae_std": float(vm.std()),
               "test_mae_mean": float(tm.mean()), "test_mae_std": float(tm.std()),
               "best_seed": best["seed"]},
              open(os.path.join(OUT, f"dual_oracle_{a.emb}_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
