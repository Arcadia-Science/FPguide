#!/usr/bin/env python
"""Curate scaffold->target design pairs for this experiment, selecting for spectral DISTANCE.

Cohort membership follows THIS experiment's own nested split (surrogate train/val/test from
``data/dual_splits.csv``), and everything it reads is local to this folder (``structure_hits.csv``
for which entries have an experimental structure; ``data/`` for the curated dataset).

THREE FILES, TWO CONDITIONS
---------------------------
Selection runs per surrogate ROLE and writes one manifest per role, because a role's pool is what
the even distance spread is computed over (and the val pool is the binding constraint -- it caps
at ~37 pairs, so merging pools before selecting would silently let train scaffolds crowd val ones
out). But the DEPLOYED surrogate does not distinguish train from val: ``train_final_surrogate.py``
refits on the train UNION val pool (n_train=515) after 3-fold CV, so an S-val scaffold sits inside
its training data exactly as an S-train scaffold does. The train/val boundary belongs to the
sweep's single-split protocol, not to the model these tasks are run against.

So ``--n`` pairs are selected per role, and the resulting cohorts are REPORTED as two conditions
(``design_common.COHORT_CONDITION``):

    seen      = knownstruct_Strain + knownstruct_Sval   -> 2 x --n tasks, scaffold in the pool
    held-out  = knownstruct_Stest                       ->     --n tasks, scaffold never trained on

At the default ``--n 36`` that is 72 seen + 36 held-out = 108 tasks. The three CSVs and the
per-cohort design output directories are unchanged -- the conditions are a grouping over them, so
switching to this framing never requires re-running a design campaign.

WHY DISTANCE, NOT IDENTITY-CLOSENESS
------------------------------------
The obvious pairing rule -- and the one the original esm2_design pipeline used -- is to match
each scaffold to the target closest to 80% identity within a [70%,90%] band. That guarantees
"same-family homolog", but says nothing about whether the target's ex/em actually DIFFERS from
the scaffold's: two 80%-identity relatives can easily have near-identical spectra, in which case
guided design barely has to move anywhere to "succeed". Since the point of this experiment is to
demonstrate the algorithm can guide designs TOWARDS OTHER PLACES in ex/em space, spectral
distance is the axis that matters and identity is only a plausibility guardrail.

So instead:
  * identity is a FLOOR plus a high cap (``--id-lo`` / ``--id-hi``), not a tight band centered on
    80% -- a scaffold and target may be very close relatives as long as their spectra genuinely
    differ. A near-identical sequence pair with a large spectral shift is exactly the kind of
    case that best demonstrates directed movement (few edits available, far to travel).
  * a minimum ex/em Euclidean distance is required (``--min-dist``), so every kept pair asks the
    algorithm to move well beyond model noise -- the default 40nm is >2x the final surrogate's
    ~17.5nm held-out test MAE (see train_final_surrogate.py).
  * per scaffold, among all targets passing both filters, the one with the LARGEST spectral
    distance is kept (that scaffold's hardest legitimate within-family target).
  * the N-per-cohort selection is spread evenly across DISTANCE, so each cohort covers a
    deliberate range of task difficulty (near/mid/far) instead of an arbitrary identity tie-break.

VALIDATED SCAFFOLDS ONLY
------------------------
Candidate scaffolds are restricted up front to those ``validate_structures.py`` has confirmed can
actually yield a design window (the deposited structure maps cleanly onto the dataset sequence,
and the sequence is in the family alignment at its own length). ``structure_hits.csv`` alone can't
tell us that -- it only knows a matching PDB entity EXISTS, not that its coordinates are usable --
so without this filter, failures surface later during build_windows.py and cohorts have to be
patched round by round, which degrades the even distance spread. Filtering here keeps selection to
a single clean pass.

Usage:
    python curate_pairs.py
    python curate_pairs.py --n 10 --id-lo 0.50 --id-hi 0.98 --min-dist 40
    python curate_pairs.py --from-cache    # re-select from the cached pool (skips the ~9min scan)
"""
import argparse
import csv
import json
import os
import time

import biotite.sequence as bseq
import biotite.sequence.align as balign
import numpy as np

# --- stage-folder bootstrap: put the experiment root (design_common), lib/ (vendored
# --- modules) and msa/ (family alignment code) on the import path.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, "lib"), _os.path.join(_ROOT, "msa")]

import design_common as C

PAIRS_DIR = str(C.PAIRS_DIR)
CHROMO_AA2 = set("YWHF")
SS_MAX = 80.0

_MAT = balign.SubstitutionMatrix.std_protein_matrix()


def has_chromo(seq, lo=50, hi=85):
    return any(seq[i + 1] in CHROMO_AA2 and seq[i + 2] == "G" and lo <= i <= hi for i in range(len(seq) - 2))


def seq_identity(a, b):
    aln = balign.align_optimal(bseq.ProteinSequence(a), bseq.ProteinSequence(b), _MAT, gap_penalty=(-10, -1))[0]
    return float(balign.get_sequence_identity(aln))


def spread_select(rows, n, key="dist"):
    """Pick n rows spread evenly across [min, max] of `key`: place n linspace targets over the
    observed range, then greedily assign each target its nearest not-yet-used row."""
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


COLS = ["scaffold_idx", "scaffold_name", "scaffold_SS", "scaffold_surr_role",
        "target_idx", "target_name", "target_SS", "target_orac_role",
        "identity", "dist_nm", "delta_ex_nm", "delta_em_nm", "scaffold_pdb"]
ROLE_COHORT = [("train", "knownstruct_Strain"), ("val", "knownstruct_Sval"), ("test", "knownstruct_Stest")]
VALIDATION_JSON = C.HERE / "structure_validation.json"
POOL_CACHE = C.PAIRS_DIR / "_full_pool_cache.json"


def load_validated_names(require=True):
    """Names of scaffolds validate_structures.py confirmed can yield a window (None if unavailable)."""
    if not VALIDATION_JSON.exists():
        msg = (f"missing {VALIDATION_JSON.name}; run validate_structures.py first so scaffolds that "
               f"cannot yield a window are excluded up front")
        if require:
            raise SystemExit(msg)
        print(f"WARNING: {msg} -- proceeding unfiltered")
        return None
    v = json.loads(VALIDATION_JSON.read_text())["scaffolds"]
    return {nm for nm, r in v.items() if r["ok"]}


def load_pool_cache():
    """Full per-role valid-pairs pool cached by a previous scan -> (pools, criteria|None).

    Tolerates the pre-criteria cache layout (a bare {role: [...]} mapping)."""
    if not POOL_CACHE.exists():
        raise SystemExit(f"missing {POOL_CACHE}; run curate_pairs.py without --from-cache first")
    raw = json.loads(POOL_CACHE.read_text())
    if "pools" in raw:
        return raw["pools"], raw.get("criteria")
    return raw, None


def _pairs_from_cache(rows, validated, criteria):
    """Rebuild the flat pairs list from the cached pool, applying the validation filter."""
    pools, cached_criteria = load_pool_cache()
    if cached_criteria is None:
        print(f"WARNING: {POOL_CACHE.name} predates criteria tracking -- assuming it was built with "
              f"the current pairing criteria {criteria}")
    elif cached_criteria != criteria:
        raise SystemExit(f"{POOL_CACHE.name} was built with criteria {cached_criteria}, but this run "
                         f"asks for {criteria}; drop --from-cache to rescan")
    pairs, n_drop = [], 0
    for role, _ in ROLE_COHORT:
        for p in pools.get(role, []):
            if validated is not None and rows[p["si"]]["name"] not in validated:
                n_drop += 1
                continue
            pairs.append(dict(si=p["si"], ti=p["ti"], idv=p["idv"], dist=p["dist"], srole=role))
    print(f"re-selecting from cached pool: {len(pairs)} valid pairs "
          f"({n_drop} dropped by the structure-validation filter)")
    return pairs


def write_cohort(fn, sel, rows, peaks, SS, hits, role):
    with open(fn, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for p in sel:
            si, ti = p["si"], p["ti"]
            w.writerow(dict(scaffold_idx=si, scaffold_name=rows[si]["name"], scaffold_SS=f"{SS[si]:.0f}",
                            scaffold_surr_role=role, target_idx=ti, target_name=rows[ti]["name"],
                            target_SS=f"{SS[ti]:.0f}", target_orac_role="train",
                            identity=f"{p['idv']:.4f}", dist_nm=f"{p['dist']:.1f}",
                            delta_ex_nm=f"{peaks[ti,0]-peaks[si,0]:.1f}",
                            delta_em_nm=f"{peaks[ti,1]-peaks[si,1]:.1f}",
                            scaffold_pdb=hits[si]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10,
                    help="pairs per ROLE cohort (train/val/test). The two reporting conditions get "
                         "2n (seen = train+val) and n (held-out = test)")
    ap.add_argument("--id-lo", type=float, default=0.50, help="identity floor (family plausibility)")
    ap.add_argument("--id-hi", type=float, default=0.98, help="identity cap (excludes near-duplicates)")
    ap.add_argument("--min-dist", type=float, default=40.0, help="min scaffold-target ex/em distance (nm)")
    ap.add_argument("--len-tol", type=int, default=30)
    ap.add_argument("--from-cache", action="store_true",
                    help="re-select from pairs/_full_pool_cache.json instead of redoing the ~9min "
                         "all-pairs identity scan (only valid if the pairing criteria are unchanged)")
    ap.add_argument("--no-validation", action="store_true",
                    help="skip the validate_structures.py filter (not recommended -- failures then "
                         "surface during build_windows.py and cohorts need patching)")
    args = ap.parse_args()

    d = C.load_dataset()
    rows, seqs, N, peaks = d["rows"], d["seqs"], d["N"], d["peaks"]
    SS = peaks[:, 1] - peaks[:, 0]

    split = C.load_split()
    hits = C.load_hits()
    print(f"structure-known entries (structure_hits.csv): {len(hits)}")

    # scaffold candidates: structure-known AND our surrogate role in {train,val,test}
    scaffolds = [i for i in hits if split.get(i, ("excluded",))[0] in ("train", "val", "test")]
    n_excl = len(hits) - len(scaffolds)
    print(f"usable scaffolds: {len(scaffolds)} ({n_excl} more are structure-known but sit in OUR "
          f"oracle's own held-out val/test -- excluded, since this cohort is about the surrogate's roles)")

    # Scaffolds a window can actually be built for (see validate_structures.py). Applied when
    # SELECTING, not when scanning -- so the cached pool below stays a complete record of which
    # scaffold/target pairs satisfy the sequence+spectral criteria, independent of structure verdicts.
    validated = None if args.no_validation else load_validated_names()
    if validated is not None:
        n_ok = sum(1 for i in scaffolds if rows[i]["name"] in validated)
        print(f"structure-validated scaffolds: {n_ok} ({len(scaffolds) - n_ok} will be dropped -- their "
              f"deposited structure does not map cleanly onto the sequence, or the sequence is absent "
              f"from the family alignment)")

    criteria = dict(id_lo=args.id_lo, id_hi=args.id_hi, min_dist=args.min_dist, len_tol=args.len_tol)
    os.makedirs(PAIRS_DIR, exist_ok=True)

    if args.from_cache:
        pairs = _pairs_from_cache(rows, validated, criteria)
    else:
        # target pool: OUR oracle-train, moderate Stokes shift, chromophore motif present
        pool = [i for i in range(N) if SS[i] < SS_MAX and split.get(i, ("", "excluded"))[1] == "train" and has_chromo(seqs[i])]
        print(f"target pool (our oracle-train, SS<{SS_MAX:.0f}, motif): {len(pool)}")
        print(f"identity floor {args.id_lo:.0%} / cap {args.id_hi:.0%} | min ex/em distance {args.min_dist:.0f}nm "
              f"| length tolerance {args.len_tol}")

        pairs = []
        t0 = time.time()
        for k, si in enumerate(scaffolds):
            Ls = len(seqs[si])
            best = None
            for ti in pool:
                if ti == si or abs(len(seqs[ti]) - Ls) > args.len_tol:
                    continue
                idv = seq_identity(seqs[si], seqs[ti])
                if not (args.id_lo <= idv <= args.id_hi):
                    continue
                dist = float(np.linalg.norm(peaks[ti] - peaks[si]))
                if dist < args.min_dist:
                    continue
                if best is None or dist > best[2]:       # keep the MOST spectrally distant valid target
                    best = (ti, idv, dist)
            if best is not None:
                ti, idv, dist = best
                pairs.append(dict(si=si, ti=ti, idv=idv, dist=dist, srole=split[si][0]))
            if (k + 1) % 100 == 0:
                print(f"  matched {k+1}/{len(scaffolds)} scaffolds ({len(pairs)} with a valid target) | "
                      f"{time.time()-t0:.0f}s", flush=True)
        print(f"scaffolds with a valid target: {len(pairs)}/{len(scaffolds)}")

        # Cache the FULL valid-pairs pool per role (not just the N selected), so re-selecting later
        # (--from-cache) never has to redo the ~9min all-pairs identity scan. Criteria are stored
        # alongside it, since a cached pool is only reusable for the criteria that produced it.
        # Cached BEFORE the validation filter, so the pool survives changes in structure verdicts.
        json.dump({"criteria": criteria,
                   "pools": {role: [dict(si=p["si"], ti=p["ti"], idv=p["idv"], dist=p["dist"])
                                    for p in pairs if p["srole"] == role]
                             for role, _ in ROLE_COHORT}},
                  open(POOL_CACHE, "w"), indent=1)

        if validated is not None:
            n_before = len(pairs)
            pairs = [p for p in pairs if rows[p["si"]]["name"] in validated]
            print(f"after the structure-validation filter: {len(pairs)} selectable pairs "
                  f"({n_before - len(pairs)} dropped)")

    selected = {}
    for role, cohort in ROLE_COHORT:
        bucket = [dict(p) for p in pairs if p["srole"] == role]
        sel = spread_select(bucket, args.n, key="dist")
        selected[cohort] = sel
        fn = os.path.join(PAIRS_DIR, f"pairs_{cohort}.csv")
        write_cohort(fn, sel, rows, peaks, SS, hits, role)
        if sel:
            ids = np.array([p["idv"] for p in sel])
            dists = np.array([p["dist"] for p in sel])
            print(f"{cohort}: {len(sel)}/{len(bucket)} available pairs | "
                  f"identity min {ids.min():.0%} med {np.median(ids):.0%} max {ids.max():.0%} | "
                  f"dist min {dists.min():.0f} med {np.median(dists):.0f} max {dists.max():.0f} nm -> {fn}")
        else:
            print(f"{cohort}: 0 pairs available -> {fn}")

    # The manifests are per role; the analysis unit is the CONDITION (see the module docstring --
    # the deployed surrogate was refit on train+val, so S-train and S-val are one condition).
    print("\nreporting conditions (the grouping every analysis should use):")
    for cond in C.CONDITIONS:
        merged = [p for coh in C.CONDITION_COHORTS[cond] for p in selected.get(coh, [])]
        if not merged:
            print(f"  {cond:9} ({C.CONDITION_LABEL[cond]}): 0 tasks"); continue
        dists = np.array([p["dist"] for p in merged])
        print(f"  {cond:9} ({C.CONDITION_LABEL[cond]}): {len(merged):3d} tasks from "
              f"{'+'.join(c.replace('knownstruct_', '') for c in C.CONDITION_COHORTS[cond])} | "
              f"dist min {dists.min():.0f} med {np.median(dists):.0f} max {dists.max():.0f} nm")
    print(f"  total: {sum(len(s) for s in selected.values())} tasks")


if __name__ == "__main__":
    main()
