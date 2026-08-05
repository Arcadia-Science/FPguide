#!/usr/bin/env python3
"""Nested oracle/surrogate train/val/test split for the peak (ex/em) curated dataset.

Inverted structure vs. dataset_pipeline/make_dual_split.py: there, the SURROGATE split is
primary (70/15/15 over the whole pool) and the oracle is a sub-split carved from the
surrogate's train remainder. Here the ORACLE split is primary instead:

  1. oracle_role: 80/10/10 train/val/test over the ENTIRE curated dataset (naive random,
     no clustering/stratification -- same style as the source script).
  2. surrogate_role: 70/15/15 train/val/test carved ONLY from oracle_role=="train" rows.
     Rows outside oracle-train (oracle val/test) get surrogate_role="excluded" -- the
     surrogate never sees them, so the oracle's val/test are held out from both models.

This gives a strict nesting (S_train/S_val/S_test subset O_train) rather than the source
script's disjoint-overlap invariants (S_test subset O_train, O_test subset S_train, S_val
disjoint O_val) -- those don't apply here since O_val/O_test are entirely off-limits to the
surrogate by construction. No fixed per-protein pins (unlike the source script's FIXED
dict) -- this is a fresh naive-random split for a standalone experiment.

Writes dual_splits.csv into ./data/ (alongside the symlinked peaks_assignments.csv /
peaks.npy / embedding caches, which are shared read-only with dataset_pipeline's curated
dir since embeddings don't depend on the split).
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # experiment root
CUR = os.path.join(HERE, "data")
ORACLE_SEED = 0
SURR_SEED = 1

rows = list(csv.DictReader(open(os.path.join(CUR, "peaks_assignments.csv"))))
rows.sort(key=lambda r: int(r["index"]))
names = [r["name"] for r in rows]
Nc = len(rows)

# ---- 1) oracle: 80/10/10 over the whole pool -----------------------------------------
n_o_te = int(round(0.10 * Nc))
n_o_va = int(round(0.10 * Nc))
operm = np.random.default_rng(ORACLE_SEED).permutation(Nc)
Orole = np.array(["train"] * Nc, dtype="<U8")
Orole[operm[:n_o_te]] = "test"
Orole[operm[n_o_te:n_o_te + n_o_va]] = "val"

# ---- 2) surrogate: 70/15/15 carved from oracle-train only ----------------------------
o_train = np.where(Orole == "train")[0]
No = len(o_train)
n_s_te = int(round(0.15 * No))
n_s_va = int(round(0.15 * No))
sperm = np.random.default_rng(SURR_SEED).permutation(o_train)
Srole = np.array(["excluded"] * Nc, dtype="<U8")     # default: outside oracle-train -> surrogate never sees it
Srole[sperm] = "train"                                # everyone in oracle-train starts as surrogate-train...
Srole[sperm[:n_s_te]] = "test"                        # ...then carve out surrogate test/val from within it
Srole[sperm[n_s_te:n_s_te + n_s_va]] = "val"

assert np.all(np.isin(np.where(np.isin(Srole, ("train", "val", "test")))[0], o_train)), \
    "surrogate train/val/test must be a subset of oracle-train"

with open(os.path.join(CUR, "dual_splits.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["index", "name", "surrogate_role", "oracle_role"])
    for i in range(Nc):
        w.writerow([i, names[i], Srole[i], Orole[i]])


def cnt(a, keys):
    return {k: int((a == k).sum()) for k in keys}


print(f"nested oracle/surrogate split written -> {os.path.relpath(os.path.join(CUR, 'dual_splits.csv'), HERE)}")
print(f"  N={Nc}")
print(f"  oracle (80/10/10 of N={Nc}):            {cnt(Orole, ('train', 'val', 'test'))}")
print(f"  surrogate (70/15/15 of oracle-train={No}): {cnt(Srole, ('train', 'val', 'test', 'excluded'))}")
print("  invariant OK: surrogate train/val/test subset oracle-train "
      "(oracle val/test held out from the surrogate entirely)")
