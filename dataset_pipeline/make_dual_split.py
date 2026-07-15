#!/usr/bin/env python3
"""Coordinated surrogate/oracle train/val/test split for the peak (ex/em) curated dataset.

Same dual-split logic as before (seed 0, 70/15/15): two splits with disjoint val sets,
arranged so each side's test set is fully seen during the other's training --
S_test subset O_train and O_test subset S_train.

On top of that, a set of **fixed per-protein role assignments** (requested 2026-07-14)
is pinned first; every remaining protein follows the standard dual split. All fixed
S_test proteins are pinned to O_train so the S_test subset O_train invariant holds.
Writes dual_splits.csv.
"""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CUR = os.path.join(HERE, "data", "peak", "curated")
SEED = 0

# Fixed assignments keyed by curated `name` -> (surrogate_role, oracle_role).
# Every S_test entry here is also O_train to keep S_test subset O_train.
FIXED = {
    # S-train, O-train
    "DsRed":      ("train", "train"),
    "ZsYellow1":  ("train", "train"),
    "LSSmOrange": ("train", "train"),
    "mAmetrine":  ("train", "train"),
    "Sumire":     ("train", "train"),
    "eqFP578":    ("train", "train"),
    "LSS-mKate1": ("train", "train"),   # "LSSmKate1"
    "LSS-mKate2": ("train", "train"),   # "LSSmKate2"
    # S-test, O-train
    "ZsGreen":    ("test",  "train"),
    "mRuby":      ("test",  "train"),
    "avGFP":      ("test",  "train"),
    "sREACh2":    ("test",  "train"),
    "Sirius":     ("test",  "train"),   # "Srius"
    "mBeRFP":     ("test",  "train"),
}

rows = list(csv.DictReader(open(os.path.join(CUR, "peaks_assignments.csv"))))
names = [r["name"] for r in rows]
Nc = len(rows)
name_to_idx = {n: i for i, n in enumerate(names)}

missing = [n for n in FIXED if n not in name_to_idx]
assert not missing, f"fixed names not found in peaks_assignments.csv: {missing}"
fixed_idx = {name_to_idx[n]: roles for n, roles in FIXED.items()}

Srole = np.array(["train"] * Nc, dtype="<U5")
Orole = np.array(["train"] * Nc, dtype="<U5")

# 1) pin fixed proteins
for i, (s, o) in fixed_idx.items():
    Srole[i], Orole[i] = s, o

# 2) standard dual split over the remaining pool
rest = np.array([i for i in range(Nc) if i not in fixed_idx])
Nr = len(rest)
n_te = int(round(0.15 * Nr))
n_va = int(round(0.15 * Nr))

perm = np.random.default_rng(SEED).permutation(rest)
Srole[perm[:n_te]] = "test"
Srole[perm[n_te:n_te + n_va]] = "val"

# oracle: reuse the rest's surrogate-TRAIN pool, reshuffled with SEED+1, so the oracle's
# test/val come from what the surrogate trained on (and vice versa)
s_train_rest = perm[n_te + n_va:]
st = np.random.default_rng(SEED + 1).permutation(s_train_rest)
Orole[st[:n_te]] = "test"
Orole[st[n_te:n_te + n_va]] = "val"

assert np.all(Orole[Srole == "test"] == "train"), "S_test must be subset of O_train"
assert np.all(Srole[Orole == "test"] == "train"), "O_test must be subset of S_train"
assert np.sum((Srole == "val") & (Orole == "val")) == 0, "S_val and O_val must be disjoint"

with open(os.path.join(CUR, "dual_splits.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["index", "name", "surrogate_role", "oracle_role"])
    for i in range(Nc):
        w.writerow([i, names[i], Srole[i], Orole[i]])


def cnt(a):
    return {b: int((a == b).sum()) for b in ("train", "val", "test")}


print(f"peak dual split written -> {os.path.relpath(os.path.join(CUR, 'dual_splits.csv'), HERE)}")
print(f"  N={Nc}  (fixed={len(fixed_idx)}, rest={Nr})")
print(f"  surrogate: {cnt(Srole)}")
print(f"  oracle:    {cnt(Orole)}")
print(f"  invariants OK: S_test subset O_train, O_test subset S_train, S_val ∩ O_val = ∅")
print("  fixed placements:")
for n, (s, o) in FIXED.items():
    i = name_to_idx[n]
    print(f"    {n:12s} (idx {i:3d}): S={Srole[i]:5s} O={Orole[i]:5s}")
