#!/usr/bin/env python3
"""3-fold CV on ALL 48 surrogate configs from the nested-split sweep (naive random folds).

Generalizes cv_surrogate_top5.py from the top-5 to the full grid. Same protocol exactly (see
that file's docstring for the full rationale/caveats -- not duplicated here): pool = surrogate
train ∪ val (515 rows), 3 naive-random KFold folds, per-fold standardization, fixed 91-row test
evaluated as a common yardstick. Same FOLD_SEED=0, so the 5 configs already in
trained_models/surrogate_cv3.csv (from cv_surrogate_top5.py) used the IDENTICAL folds -- this
script reuses those rows instead of recomputing them, and only runs the remaining 43 configs.

Usage:
    python cv_all_surrogate.py
    python cv_all_surrogate.py --force   # ignore existing rows, recompute all 48
"""
import argparse
import csv
import os

import numpy as np
from sklearn.model_selection import KFold

# --- stage-folder bootstrap: put the experiment root (design_common), lib/ (vendored
# --- modules) and msa/ (family alignment code) on the import path.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, "lib"), _os.path.join(_ROOT, "msa")]

import sweep_peak_oracle as swp
from cv_surrogate_top5 import FOLD_SEED, N_FOLDS, train_eval_idx

OUT = os.path.join(_ROOT, "trained_models", "surrogate_cv3.csv")


def load_existing():
    if not os.path.exists(OUT):
        return {}
    rows = list(csv.DictReader(open(OUT)))
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    return {lab: rs for lab, rs in by_label.items() if len(rs) == N_FOLDS}   # only complete labels are reusable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute all 48 configs, ignoring existing rows")
    args = ap.parse_args()

    dev = swp.device()
    D = swp.load_data("surrogate", dev, to_gpu=True)
    pool = np.concatenate([D["tr"], D["va"]])
    test_idx = D["te"]
    print(f"CV pool (train+val) = {len(pool)} | fixed held-out test = {len(test_idx)}")

    cfgs = swp.make_configs()
    existing = {} if args.force else load_existing()
    if existing:
        print(f"reusing {len(existing)} already-complete config(s) from {OUT}: {sorted(existing)}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    all_rows = [r for rs in existing.values() for r in rs]
    todo = [c for c in cfgs if swp.label(c) not in existing]
    print(f"{len(todo)}/{len(cfgs)} configs to run x {N_FOLDS} folds = {len(todo) * N_FOLDS} fits")

    for ci, spec in enumerate(todo):
        lab = swp.label(spec)
        fold_va_mae, fold_te_mae = [], []
        for k, (tri, vai) in enumerate(kf.split(pool)):
            tr, va = pool[tri], pool[vai]
            net, ep = train_eval_idx(spec, tr, va, D, dev)
            va_mae = swp.base.eval_mae(net, D, va, dev)[0]
            te_mae = swp.base.eval_mae(net, D, test_idx, dev)[0]
            fold_va_mae.append(va_mae); fold_te_mae.append(te_mae)
            all_rows.append(dict(label=lab, fold=k, n_tr=len(tr), n_va=len(va), epoch_best=ep,
                                 fold_va_mae=va_mae, fixed_test_mae=te_mae))
            print(f"[{ci+1:2}/{len(todo)}] [{lab:22}] fold {k} n_tr={len(tr)} n_va={len(va)} @ep{ep} "
                  f"fold-va {va_mae:5.2f} nm | fixed-test {te_mae:5.2f} nm", flush=True)
        print(f"  {lab:22} CV fold-va {np.mean(fold_va_mae):5.2f} +/- {np.std(fold_va_mae):4.2f} nm | "
              f"fixed-test {np.mean(fold_te_mae):5.2f} +/- {np.std(fold_te_mae):4.2f} nm\n", flush=True)

    all_rows.sort(key=lambda r: (r["label"], int(r["fold"])))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows ({len(all_rows)//N_FOLDS} configs) -> {OUT}")


if __name__ == "__main__":
    main()
