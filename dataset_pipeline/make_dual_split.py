#!/usr/bin/env python3
"""Coordinated surrogate/oracle train/val/test split for the peak (ex/em) curated dataset.

Identical logic to the original peak_design/curate_split_visualize.ipynb (seed 0, 70/15/15):
two splits with disjoint val sets, arranged so each side's test set is fully seen during the
other's training -- S_test subset O_train and O_test subset S_train. Writes dual_splits.csv.
"""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CUR = os.path.join(HERE, "data", "peak", "curated")
SEED = 0

rows = list(csv.DictReader(open(os.path.join(CUR, "peaks_assignments.csv"))))
names = [r["name"] for r in rows]
Nc = len(rows)

perm = np.random.default_rng(SEED).permutation(Nc)
n_te = int(round(0.15 * Nc))
n_va = int(round(0.15 * Nc))

# surrogate split
Srole = np.array(["train"] * Nc)
Srole[perm[:n_te]] = "test"
Srole[perm[n_te:n_te + n_va]] = "val"

# oracle split: reuse the surrogate TRAIN pool, reshuffled with SEED+1, so the oracle's
# test/val come from what the surrogate trained on (and vice versa)
s_train = perm[n_te + n_va:]
st = np.random.default_rng(SEED + 1).permutation(s_train)
Orole = np.array(["train"] * Nc)
Orole[st[:n_te]] = "test"
Orole[st[n_te:n_te + n_va]] = "val"

assert np.all(Orole[Srole == "test"] == "train"), "S_test must be subset of O_train"
assert np.all(Srole[Orole == "test"] == "train"), "O_test must be subset of S_train"

with open(os.path.join(CUR, "dual_splits.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["index", "name", "surrogate_role", "oracle_role"])
    for i in range(Nc):
        w.writerow([i, names[i], Srole[i], Orole[i]])


def cnt(a):
    return {b: int((a == b).sum()) for b in ("train", "val", "test")}


print(f"peak dual split written -> {os.path.relpath(os.path.join(CUR, 'dual_splits.csv'), HERE)}")
print(f"  N={Nc}")
print(f"  surrogate: {cnt(Srole)}")
print(f"  oracle:    {cnt(Orole)}")
print(f"  invariants OK: S_test subset O_train, O_test subset S_train")
