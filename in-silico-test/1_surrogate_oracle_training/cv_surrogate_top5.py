#!/usr/bin/env python3
"""3-fold CV on the top-5 surrogate configs from the nested-split sweep (naive random folds).

Pool: surrogate train ∪ val (515 rows, everything the sweep used for fitting+early-stopping).
The surrogate's held-out test (91 rows) is left completely untouched throughout -- it's used
only as a FIXED common yardstick evaluated once per fold-model, so all 5 configs x 3 folds are
compared against the exact same 91 sequences in addition to their own fold's held-out slice.

Naive random KFold (no sequence-identity clustering) -- consistent with this experiment's
already-naive-random dual_splits.csv. Note (flagged to the user before running this): FP
variants in this dataset are frequently near-identical point mutants of the same parent
scaffold, so naive random folds can place near-duplicates on both sides of a fold, which can
optimistically bias the held-out MAE. See esm2_design/cluster_split/run_oracle_cv.py for the
cluster-grouped alternative (not used here per the user's choice).

Per fold: mean/std standardization is recomputed from THAT fold's train indices only (proper
per-fold normalization, not leaking the pooled statistics). Same training protocol as the sweep
(Adam 1e-3, wd 1e-4, batch 32, early stop on fold-val MAE, patience 20, max_epochs 200).

Usage:
    python cv_surrogate_top5.py
"""
import copy
import csv
import json
import os

import numpy as np
import torch
from sklearn.model_selection import KFold

# --- stage-folder bootstrap: put the experiment root (design_common), lib/ (vendored
# --- modules) and msa/ (family alignment code) on the import path.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, "lib"), _os.path.join(_ROOT, "msa")]

import peak_models as pm
import sweep_peak_oracle as swp

TOP5 = [
    {"arch": "cnn", "pool": "concat",    "n_conv": 2, "depth": 2},
    {"arch": "cnn", "pool": "max",       "n_conv": 2, "depth": 2},
    {"arch": "cnn", "pool": "concat",    "n_conv": 1, "depth": 1},
    {"arch": "cnn", "pool": "max",       "n_conv": 1, "depth": 1},
    {"arch": "cnn", "pool": "concatstd", "n_conv": 2, "depth": 2},
]
N_FOLDS = 3
FOLD_SEED = 0
SEED = 0
MAX_EPOCHS, PATIENCE = 200, 20
OUT = os.path.join(_ROOT, "trained_models", "surrogate_cv3.csv")


def train_eval_idx(spec, tr, va, D, dev, seed=SEED, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """Same protocol as sweep_peak_oracle.train_eval, but on explicit (tr, va) index arrays with
    per-fold standardization, instead of the sweep's fixed D['tr']/D['va']/D['mean']/D['std']."""
    torch.manual_seed(seed); rng = np.random.default_rng(100 + seed)
    peaks_np = D["Pk"].cpu().numpy()
    mean = peaks_np[tr].mean(0).astype(np.float32); std = (peaks_np[tr].std(0) + 1e-6).astype(np.float32)
    base = pm.build_base({**spec, "d_in": D["d_in"]}, dev, out=2, drop=spec.get("drop", 0.2))
    net = pm.wrap(base, mean, std, dev)
    sd = torch.tensor(std, device=dev)
    opt = torch.optim.Adam(net.parameters(), swp.base.LR, weight_decay=swp.base.WD)
    best = {"val": float("inf"), "state": None, "epoch": -1}
    bad = 0
    for ep in range(max_epochs):
        net.train()
        for Hb, mk, b in swp.base.batches(D, tr, dev, shuffle=True, rng=rng):
            opt.zero_grad()
            (((net(Hb, mk) - D["Pk"][torch.as_tensor(b, device=dev)]) / sd) ** 2).mean().backward()
            opt.step()
        v = swp.base.eval_mae(net, D, va, dev)[0]
        if v < best["val"] - 1e-4:
            best = {"val": v, "state": copy.deepcopy({k: x.cpu() for k, x in net.base.state_dict().items()}),
                    "epoch": ep}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.base.load_state_dict(best["state"])
    return net, best["epoch"]


def main():
    dev = swp.device()
    D = swp.load_data("surrogate", dev, to_gpu=True)
    pool = np.concatenate([D["tr"], D["va"]])          # 515: everything the sweep fit+early-stopped on
    test_idx = D["te"]                                  # 91: fixed, untouched, common yardstick
    print(f"CV pool (train+val) = {len(pool)} | fixed held-out test = {len(test_idx)}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    rows = []
    for spec in TOP5:
        lab = swp.label(spec)
        fold_va_mae, fold_te_mae = [], []
        for k, (tri, vai) in enumerate(kf.split(pool)):
            tr, va = pool[tri], pool[vai]
            net, ep = train_eval_idx(spec, tr, va, D, dev)
            va_mae = swp.base.eval_mae(net, D, va, dev)[0]
            te_mae = swp.base.eval_mae(net, D, test_idx, dev)[0]
            fold_va_mae.append(va_mae); fold_te_mae.append(te_mae)
            rows.append(dict(label=lab, fold=k, n_tr=len(tr), n_va=len(va), epoch_best=ep,
                             fold_va_mae=va_mae, fixed_test_mae=te_mae))
            print(f"[{lab:18}] fold {k} n_tr={len(tr)} n_va={len(va)} @ep{ep} "
                  f"fold-va {va_mae:5.2f} nm | fixed-test {te_mae:5.2f} nm", flush=True)
        print(f"  {lab:18} CV fold-va {np.mean(fold_va_mae):5.2f} +/- {np.std(fold_va_mae):4.2f} nm | "
              f"fixed-test {np.mean(fold_te_mae):5.2f} +/- {np.std(fold_te_mae):4.2f} nm\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
