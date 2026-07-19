#!/usr/bin/env python
"""Split the full known-structure manifest (pairs_knownstruct_Otrain.csv) into two
role-based sub-cohorts and cap each to N pairs with an EVEN SPREAD across the
target-identity band:

  * knownstruct_Strain_Otrain : scaffold in surrogate TRAIN split
  * knownstruct_Stest_Otrain  : scaffold in surrogate TEST  split (generalization)

"Even spread" = N evenly spaced identity targets between the subset's min and max
identity; greedily assign the closest still-unused pair to each target. Deterministic.

Usage
-----
    python select_knownstruct.py                 # 20 train + 20 test
    python select_knownstruct.py --n-train 20 --n-test 20
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C

FULL = C.pairs_csv_path("knownstruct_Otrain")
COLS = C.PAIRS_COLS + ["scaffold_pdb"]


def spread_select(rows, n):
    """Pick n rows with identities evenly spread across [min, max]. rows: list of dicts."""
    if len(rows) <= n:
        return sorted(rows, key=lambda r: float(r["identity"]))
    ids = np.array([float(r["identity"]) for r in rows])
    targets = np.linspace(ids.min(), ids.max(), n)
    used = set()
    picked = []
    for t in targets:
        order = np.argsort(np.abs(ids - t))
        for j in order:
            if j not in used:
                used.add(j)
                picked.append(rows[j])
                break
    return sorted(picked, key=lambda r: float(r["identity"]))


def write_manifest(cohort, rows):
    fn = C.pairs_csv_path(cohort)
    with open(fn, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in COLS})
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--roles", nargs="*", default=["train", "val", "test"],
                    help="which surrogate-split roles to (re)generate manifests for")
    args = ap.parse_args()

    if not os.path.exists(FULL):
        raise SystemExit(f"missing {FULL}; run curate_knownstruct.py first")
    allrows = list(csv.DictReader(open(FULL)))
    by_role = {"train": [], "val": [], "test": []}
    for r in allrows:
        by_role.setdefault(r["scaffold_surr_role"], []).append(r)
    print(f"available: train={len(by_role['train'])} val={len(by_role['val'])} test={len(by_role['test'])}")

    specs = {"train": (args.n_train, "knownstruct_Strain_Otrain"),
             "val": (args.n_val, "knownstruct_Sval_Otrain"),
             "test": (args.n_test, "knownstruct_Stest_Otrain")}
    for role in args.roles:
        n, cohort = specs[role]
        pool = by_role.get(role, [])
        sel = spread_select(pool, n)
        fn = write_manifest(cohort, sel)
        ids = np.array([float(r["identity"]) for r in sel])
        npdb = len({r["scaffold_pdb"] for r in sel})
        print(f"\n{cohort}: {len(sel)} pairs  ({npdb} unique PDBs)")
        print(f"  identity: min {ids.min():.0%} med {np.median(ids):.0%} max {ids.max():.0%}")
        print(f"  -> {fn}")
        for r in sel:
            print(f"    {r['scaffold_name']:22}[{r['scaffold_pdb']}] SS{r['scaffold_SS']:>3} "
                  f"-> {r['target_name']:22} SS{r['target_SS']:>3}  id {float(r['identity']):.0%}")
    print("\nselect_knownstruct done.")


if __name__ == "__main__":
    main()
