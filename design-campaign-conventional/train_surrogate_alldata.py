#!/usr/bin/env python
"""Train the BEST peak surrogate architecture (cnn-max-d1, the surrogate-sweep winner:
ESM-2 650M layer-33 residue embeddings -> 1x Conv1d(1280->128,k5) -> max-pool -> MLP head)
on **ALL** available curated FP data (no train/val/test split), for a conventional
design campaign.

Differences vs peak_design/sweep_peak_oracle.py (which trains on the surrogate TRAIN
split with val early-stopping):
  * training index = every one of the N=758 curated FPs;
  * target (ex_max, em_max) standardization mean/std computed over ALL N;
  * no held-out set, so no val-based early stopping -> train a fixed epoch budget
    (default 60, ~ the sweep's best epoch for this arch) with the identical
    optimizer/loss (Adam 1e-3, wd 1e-4, batch 32, standardized-peak MSE).

Reuses the precomputed ESM-2 residue cache and the sweep's data/loss helpers, so only
the ~0.9M-param CNN head is fit (fast, a few minutes on GPU).

Output: models/surrogate_cnn-max-d1_alldata.pt  (loadable via peak_models.load_model,
then peak_models.wrap(base, ck["mean"], ck["std"], dev) for nm predictions).

Usage:
    python train_surrogate_alldata.py                 # 60 epochs on all 758 FPs
    python train_surrogate_alldata.py --epochs 80 --seed 0
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "peak_design"))

import peak_models as pm
import sweep_peak_oracle as swp

BEST_SPEC = {"arch": "cnn", "pool": "max", "n_conv": 1, "depth": 1}   # surrogate-sweep winner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60,
                    help="fixed training epochs (no val early-stopping; ~sweep best epoch)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "models" / "surrogate_cnn-max-d1_alldata.pt"))
    args = ap.parse_args()

    dev = swp.device()
    print(f"device: {dev}")

    # Load peaks + ESM-2 residue cache + (unused) split via the sweep helper.
    D = swp.load_data("surrogate", dev, to_gpu=(dev.type == "cuda"))
    N = D["N"]

    # ---- ALL-DATA overrides: train on every FP; standardize on all N --------------
    peaks_all = D["Pk"].detach().cpu().numpy()
    mean = peaks_all.mean(0).astype(np.float32)
    std = (peaks_all.std(0) + 1e-6).astype(np.float32)
    D["mean"], D["std"] = mean, std
    D["sd"] = torch.tensor(std, device=dev)
    all_idx = np.arange(N)
    print(f"training cnn-max-d1 on ALL {N} FPs (no split) | "
          f"peak mean {mean.round(1)} std {std.round(1)} nm | {args.epochs} epochs")

    # ---- build + train (identical optimizer/loss to the sweep) --------------------
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(100 + args.seed)
    base = pm.build_base({**BEST_SPEC, "d_in": D["d_in"]}, dev, out=2, drop=0.2)
    net = pm.wrap(base, mean, std, dev)
    opt = torch.optim.Adam(net.parameters(), swp.LR, weight_decay=swp.WD)

    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        for Hb, mk, b in swp.batches(D, all_idx, dev, shuffle=True, rng=rng):
            opt.zero_grad()
            (((net(Hb, mk) - D["Pk"][torch.as_tensor(b, device=dev)]) / D["sd"]) ** 2).mean().backward()
            opt.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            mae = swp.eval_mae(net, D, all_idx, dev)
            print(f"  ep {ep+1:3}/{args.epochs} | train MAE {mae[0]:5.2f} nm "
                  f"(ex {mae[1]:4.1f} em {mae[2]:4.1f}) | {time.time()-t0:.0f}s", flush=True)

    train_mae = swp.eval_mae(net, D, all_idx, dev)
    print(f"final train MAE {train_mae[0]:.2f} nm (ex {train_mae[1]:.1f}, em {train_mae[2]:.1f}) "
          f"over all {N} FPs | {time.time()-t0:.0f}s")

    # ---- save checkpoint (compatible with peak_models.load_model) -----------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pm.save_model(args.out, net.base, {
        **BEST_SPEC, "d_in": D["d_in"], "out": 2, "drop": 0.2, "seed": args.seed,
        "role": "surrogate", "emb": D["emb"], "mean": mean, "std": std,
        "train_on": "all", "n_train": int(N), "epochs": args.epochs,
        "train_mae": train_mae[0], "train_ex": train_mae[1], "train_em": train_mae[2],
        "sweep_val_mae_ref": 13.81,   # same arch on the sweep's held-out val split, for reference
    })
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
