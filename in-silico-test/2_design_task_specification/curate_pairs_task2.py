#!/usr/bin/env python
"""Curate design task set #2: a RANDOM qualifying target per scaffold, not the furthest one.

This is ``2_design_task_specification/curate_pairs.py`` with one thing changed and two cohorts
instead of three. Every filter is identical -- identity floor/cap, minimum ex/em distance,
length tolerance, oracle-train target pool with a moderate Stokes shift and a chromophore motif,
and the ``validate_structures.py`` gate on scaffolds. What changes is which of the qualifying
targets a scaffold is paired with:

    task 1 (stage 2)   argmax over distance -- the scaffold's hardest legitimate target
    task 2 (HERE)      a uniform random draw over the same qualifying set

WHY. Task 1's cohorts are extreme by construction: taking the furthest target of ~85 candidates
puts every task 100-300 nm from its scaffold (median 190), so the reported gains are measured on
the tail of the task distribution and nothing says the same algorithm behaves the same way on an
ordinary target. Drawing uniformly from the identical candidate set holds the eligibility rules
fixed and moves only the difficulty, which makes the two task sets a controlled pair: any
difference between them is attributable to where in the distance distribution the targets sit.

TWO COHORTS, NOT THREE
----------------------
Task 1 selected per surrogate ROLE (train/val/test) and merged train+val into the ``seen``
condition only at analysis time. That was to keep the per-role distance spread balanced against
a val pool that caps at ~37 pairs. Here the pools are merged UP FRONT, because the deployed
surrogate does not distinguish train from val -- ``train_final_surrogate.py`` refits on the
train UNION val pool (n_train=515), so an S-val scaffold sits inside its training data exactly
as an S-train scaffold does (see ``design_common``). So:

    knownstruct_Spool   scaffold in the refit surrogate's training pool (S-train + S-val)  -> 36
    knownstruct_Stest   scaffold the surrogate has never been trained on                   -> 36

which is the ``seen`` / ``held-out`` contrast the analysis actually uses, at 36 tasks a side.
``scaffold_surr_role`` still records whether an S-pool scaffold came from train or val.

THE CANDIDATE POOL CACHE
------------------------
Random selection needs the whole qualifying set per scaffold, so this stage caches
``pairs_task2/_candidate_pool_cache.json`` -- EVERY (scaffold, target) pair passing the criteria,
not just the winner. Stage 2's ``_full_pool_cache.json`` is one row per scaffold (its argmax)
and cannot be re-drawn from, which is why the ~10 min all-pairs identity scan is redone once here
rather than reused. Afterwards ``--from-cache`` re-selects instantly.

Draws are reproducible from identity alone: a scaffold's target is drawn from a stream seeded by
``(SEED, scaffold_idx)``, so it does not depend on scan order, on which cohort the scaffold is
in, or on how many scaffolds were selected.

Usage:
    python 4_design_task2/curate_pairs_task2.py                 # ~10 min scan, then select
    python 4_design_task2/curate_pairs_task2.py --from-cache    # re-select, instant
    python 4_design_task2/curate_pairs_task2.py --from-cache --select spread
"""
import argparse
import csv
import json
import os
import time

import biotite.sequence as bseq
import biotite.sequence.align as balign
import numpy as np

# --- stage-folder bootstrap: put the experiment root (design_common) and lib/ on the import path.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, "lib")]

import design_common as C

PAIRS_DIR = str(C.PAIRS_DIR_T2)
POOL_CACHE = C.PAIRS_DIR_T2 / "_candidate_pool_cache.json"
VALIDATION_JSON = C.HERE / "structure_validation.json"

CHROMO_AA2 = set("YWHF")
SS_MAX = 80.0
_MAT = balign.SubstitutionMatrix.std_protein_matrix()

COLS = ["scaffold_idx", "scaffold_name", "scaffold_SS", "scaffold_surr_role",
        "target_idx", "target_name", "target_SS", "target_orac_role",
        "identity", "dist_nm", "delta_ex_nm", "delta_em_nm", "scaffold_pdb",
        "n_candidates", "dist_rank", "max_dist_nm"]

# (cohort, surrogate roles pooled into it) -- train and val are ONE pool here, see the docstring
COHORT_ROLES = [("knownstruct_Spool", ("train", "val")), ("knownstruct_Stest", ("test",))]


def has_chromo(seq, lo=50, hi=85):
    return any(seq[i + 1] in CHROMO_AA2 and seq[i + 2] == "G" and lo <= i <= hi for i in range(len(seq) - 2))


def seq_identity(a, b):
    aln = balign.align_optimal(bseq.ProteinSequence(a), bseq.ProteinSequence(b), _MAT, gap_penalty=(-10, -1))[0]
    return float(balign.get_sequence_identity(aln))


def spread_select(rows, n, key="dist"):
    """Stage 2's rule, kept for --select spread: n rows spread evenly across [min, max] of `key`."""
    if len(rows) <= n:
        return sorted(rows, key=lambda r: r[key])
    vals = np.array([r[key] for r in rows])
    targets = np.linspace(vals.min(), vals.max(), n)
    used, picked = set(), []
    for t in targets:
        for j in np.argsort(np.abs(vals - t)):
            if j not in used:
                used.add(j); picked.append(rows[j]); break
    return sorted(picked, key=lambda r: r[key])


def random_select(rows, n, salt):
    """Uniform sample of n scaffolds, seeded by (SEED, salt) so the cohort is reproducible."""
    if len(rows) <= n:
        return sorted(rows, key=lambda r: r["dist"])
    rows = sorted(rows, key=lambda r: r["si"])       # order-independent of the scan
    rng = np.random.default_rng(np.random.SeedSequence([C.SEED, salt]))
    idx = rng.choice(len(rows), size=n, replace=False)
    return sorted((rows[int(j)] for j in idx), key=lambda r: r["dist"])


def draw_target(si, cands):
    """Uniform draw over a scaffold's qualifying targets, from a stream seeded by (SEED, si)."""
    cands = sorted(cands, key=lambda c: c["ti"])     # order-independent of the scan
    rng = np.random.default_rng(np.random.SeedSequence([C.SEED, si]))
    return cands[int(rng.integers(len(cands)))]


def load_validated_names(require=True):
    """Names of scaffolds validate_structures.py confirmed can yield a window (None if unavailable)."""
    if not VALIDATION_JSON.exists():
        msg = (f"missing {VALIDATION_JSON.name}; run 2_design_task_specification/validate_structures.py "
               f"first so scaffolds that cannot yield a window are excluded up front")
        if require:
            raise SystemExit(msg)
        print(f"WARNING: {msg} -- proceeding unfiltered")
        return None
    v = json.loads(VALIDATION_JSON.read_text())["scaffolds"]
    return {nm for nm, r in v.items() if r["ok"]}


def scan(scaffolds, pool, seqs, peaks, split, args):
    """Every (scaffold, target) pair passing the criteria -> {scaffold_idx: [candidates]}."""
    cands, t0 = {}, time.time()
    for k, si in enumerate(scaffolds):
        Ls = len(seqs[si])
        hits = []
        for ti in pool:
            if ti == si or abs(len(seqs[ti]) - Ls) > args.len_tol:
                continue
            idv = seq_identity(seqs[si], seqs[ti])
            if not (args.id_lo <= idv <= args.id_hi):
                continue
            dist = float(np.linalg.norm(peaks[ti] - peaks[si]))
            if dist < args.min_dist:
                continue
            hits.append(dict(ti=ti, idv=idv, dist=dist))
        if hits:
            cands[si] = hits
        if (k + 1) % 100 == 0:
            print(f"  scanned {k+1}/{len(scaffolds)} scaffolds ({len(cands)} with >=1 valid target, "
                  f"{sum(len(v) for v in cands.values())} candidate pairs) | {time.time()-t0:.0f}s",
                  flush=True)
    return cands


def load_pool_cache(criteria):
    if not POOL_CACHE.exists():
        raise SystemExit(f"missing {POOL_CACHE}; run without --from-cache first")
    raw = json.loads(POOL_CACHE.read_text())
    if raw.get("criteria") != criteria:
        raise SystemExit(f"{POOL_CACHE.name} was built with criteria {raw.get('criteria')}, but this "
                         f"run asks for {criteria}; drop --from-cache to rescan")
    return {int(si): v for si, v in raw["candidates"].items()}


def write_cohort(fn, sel, rows, peaks, SS, hits):
    with open(fn, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for p in sel:
            si, ti = p["si"], p["ti"]
            w.writerow(dict(scaffold_idx=si, scaffold_name=rows[si]["name"], scaffold_SS=f"{SS[si]:.0f}",
                            scaffold_surr_role=p["srole"], target_idx=ti, target_name=rows[ti]["name"],
                            target_SS=f"{SS[ti]:.0f}", target_orac_role="train",
                            identity=f"{p['idv']:.4f}", dist_nm=f"{p['dist']:.1f}",
                            delta_ex_nm=f"{peaks[ti,0]-peaks[si,0]:.1f}",
                            delta_em_nm=f"{peaks[ti,1]-peaks[si,1]:.1f}",
                            scaffold_pdb=hits[si],
                            n_candidates=p["n_cand"], dist_rank=p["dist_rank"],
                            max_dist_nm=f"{p['max_dist']:.1f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=36, help="pairs per cohort (S-pool and S-test)")
    ap.add_argument("--id-lo", type=float, default=0.50, help="identity floor (family plausibility)")
    ap.add_argument("--id-hi", type=float, default=0.98, help="identity cap (excludes near-duplicates)")
    ap.add_argument("--min-dist", type=float, default=40.0, help="min scaffold-target ex/em distance (nm)")
    ap.add_argument("--len-tol", type=int, default=30)
    ap.add_argument("--select", choices=["random", "spread"], default="random",
                    help="how the --n scaffolds per cohort are chosen: a uniform sample (default, "
                         "keeps the natural distance distribution of the random targets), or stage "
                         "2's even spread across distance")
    ap.add_argument("--from-cache", action="store_true",
                    help="re-select from pairs_task2/_candidate_pool_cache.json instead of redoing "
                         "the ~10min all-pairs identity scan")
    ap.add_argument("--no-validation", action="store_true",
                    help="skip the validate_structures.py filter (not recommended)")
    args = ap.parse_args()

    d = C.load_dataset()
    rows, seqs, N, peaks = d["rows"], d["seqs"], d["N"], d["peaks"]
    SS = peaks[:, 1] - peaks[:, 0]

    split = C.load_split()
    hits = C.load_hits()
    print(f"structure-known entries (structure_hits.csv): {len(hits)}")

    scaffolds = [i for i in hits if split.get(i, ("excluded",))[0] in ("train", "val", "test")]
    print(f"usable scaffolds: {len(scaffolds)} ({len(hits) - len(scaffolds)} more are structure-known "
          f"but sit in OUR oracle's own held-out val/test -- excluded)")

    validated = None if args.no_validation else load_validated_names()
    if validated is not None:
        n_ok = sum(1 for i in scaffolds if rows[i]["name"] in validated)
        print(f"structure-validated scaffolds: {n_ok} ({len(scaffolds) - n_ok} will be dropped)")

    criteria = dict(id_lo=args.id_lo, id_hi=args.id_hi, min_dist=args.min_dist, len_tol=args.len_tol)
    os.makedirs(PAIRS_DIR, exist_ok=True)

    if args.from_cache:
        cands = load_pool_cache(criteria)
        print(f"candidate pool from cache: {len(cands)} scaffolds, "
              f"{sum(len(v) for v in cands.values())} qualifying pairs")
    else:
        pool = [i for i in range(N) if SS[i] < SS_MAX and split.get(i, ("", "excluded"))[1] == "train"
                and has_chromo(seqs[i])]
        print(f"target pool (our oracle-train, SS<{SS_MAX:.0f}, motif): {len(pool)}")
        print(f"identity floor {args.id_lo:.0%} / cap {args.id_hi:.0%} | min ex/em distance "
              f"{args.min_dist:.0f}nm | length tolerance {args.len_tol}")
        cands = scan(scaffolds, pool, seqs, peaks, split, args)
        # cached BEFORE the validation filter, so the pool survives changes in structure verdicts
        json.dump({"criteria": criteria,
                   "candidates": {str(si): [dict(ti=c["ti"], idv=c["idv"], dist=round(c["dist"], 4))
                                            for c in v] for si, v in cands.items()}},
                  open(POOL_CACHE, "w"))
        print(f"scaffolds with >=1 valid target: {len(cands)}/{len(scaffolds)} | "
              f"{sum(len(v) for v in cands.values())} candidate pairs -> {POOL_CACHE.name}")

    # ---- one random target per scaffold, drawn from that scaffold's own qualifying set ----
    pairs = []
    for si, cl in cands.items():
        if validated is not None and rows[si]["name"] not in validated:
            continue
        c = draw_target(si, cl)
        dists = sorted((x["dist"] for x in cl), reverse=True)
        pairs.append(dict(si=si, ti=c["ti"], idv=c["idv"], dist=c["dist"], srole=split[si][0],
                          n_cand=len(cl), dist_rank=1 + dists.index(c["dist"]), max_dist=dists[0]))
    print(f"selectable scaffolds after the structure-validation filter: {len(pairs)} | "
          f"candidates per scaffold min/med/max = "
          f"{min(p['n_cand'] for p in pairs)}/{int(np.median([p['n_cand'] for p in pairs]))}/"
          f"{max(p['n_cand'] for p in pairs)}")

    selected = {}
    for ci, (cohort, roles) in enumerate(COHORT_ROLES):
        bucket = [dict(p) for p in pairs if p["srole"] in roles]
        sel = (random_select(bucket, args.n, ci) if args.select == "random"
               else spread_select(bucket, args.n, key="dist"))
        selected[cohort] = sel
        fn = os.path.join(PAIRS_DIR, f"pairs_{cohort}.csv")
        write_cohort(fn, sel, rows, peaks, SS, hits)
        ids = np.array([p["idv"] for p in sel]); dd = np.array([p["dist"] for p in sel])
        mx = np.array([p["max_dist"] for p in sel])
        print(f"{cohort} ({'+'.join(roles)}): {len(sel)}/{len(bucket)} available | "
              f"identity min {ids.min():.0%} med {np.median(ids):.0%} max {ids.max():.0%} | "
              f"dist min {dd.min():.0f} med {np.median(dd):.0f} max {dd.max():.0f} nm "
              f"(task 1's argmax rule would give med {np.median(mx):.0f}) -> {fn}")

    allsel = [p for s in selected.values() for p in s]
    dd = np.array([p["dist"] for p in allsel]); mx = np.array([p["max_dist"] for p in allsel])
    print(f"\ntotal: {len(allsel)} tasks | random-target distance med {np.median(dd):.0f} nm vs "
          f"{np.median(mx):.0f} nm for the same scaffolds' furthest target "
          f"({np.mean(dd < 100) * 100:.0f}% of tasks now start under 100 nm)")


if __name__ == "__main__":
    main()
