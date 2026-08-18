#!/usr/bin/env python
"""Build Tier-B design windows FROM SCRATCH for this experiment's own scaffold/target pairs
(``curate_pairs_task2.py`` -> ``pairs_task2/``).

Everything is computed locally and nothing is reused from other pipeline folders: the pocket
geometry comes from ``pockets.py`` against the local ``structures/experimental/`` RCSB cache
(self-populating -- a cache miss is fetched from RCSB). Building from scratch keeps this folder
standalone and makes the windows reproducible from its own inputs alone, at the cost of
recomputing geometry for scaffolds other folders happen to share.

The window rule:
  * editable  = chromophore positions 1-2 + every residue with an atom within 5 A of the
                chromophore ("Tier-B" pocket)
  * fixed     = chromophore position 3 + the catalytic residues
  * per-position alphabet is constrained to aromatics at chromophore position 2 and to H-bond
    capable residues at positions H-bonding the chromophore.

This used to also emit a per-scaffold family PSSM -- each position's alphabet intersected with
what a 763-sequence family alignment supported there -- for the MSA-proposal design arm. That arm
is archived and neither live arm consults the PSSM, so the family profile was dropped along with
the alignment (see ``archive/msa/``). Windows built before the removal carried a ``pssm`` block;
it is gone from ``design_windows.json`` too.

Every scaffold reaching this point should already be buildable: ``validate_structures.py``
(beside it) checks the structure-quality gate (>=90% local identity, >=70% coverage) for every
candidate, and curation selects only from the ones that passed.
A skip here therefore means the validation cache is stale relative to ``structures/experimental/``,
or curation was run with ``--no-validation``.

Re-running is resumable: any scaffold already present in ``design_windows.json`` is kept as-is
and only missing ones are computed. Use ``--rebuild`` to recompute everything from zero.

``design_windows.json`` is a UNION over every task set built so far, not a snapshot of one
cohort's scaffolds: a window is a property of the scaffold (its structure and its sequence),
independent of which target it was paired with. The file on disk therefore holds 137 scaffolds --
task 2's (the default here) plus the 108 of the archived task 1, which were built first and are
left in place. Both task sets read the same file and share the scaffolds they have in common
instead of recomputing their geometry.

This script and ``validate_structures.py`` came from task 1's task-specification stage, now under
``archive/``; they live here because task 2's curation is their only live caller. See
``archive/README.md``.

Usage
-----
    python 2_design_task_specification/build_windows.py            # task 2's cohorts (the default)
    python 2_design_task_specification/build_windows.py --rebuild
    python 2_design_task_specification/build_windows.py --pairs-dir archive/pairs \
        --cohorts knownstruct_Strain knownstruct_Sval knownstruct_Stest   # archived task 1
"""
import argparse
import json
import sys
import time

import numpy as np

# --- stage-folder bootstrap: put the experiment root (design_common) and lib/ (vendored
# --- modules) on the import path.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, "lib")]

import design_common as C

import pockets

CUTOFF, HBOND_CUTOFF = 5.0, 3.5
AROMATIC = ["Y", "W", "H", "F"]
HBOND_AA = ["S", "T", "Y", "N", "Q", "D", "E", "H", "K", "R", "W"]
OUT = C.WINDOWS_JSON


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="recompute every window from zero")
    ap.add_argument("--pairs-dir", default=str(C.PAIRS_DIR_T2),
                    help=f"manifest directory -- the task set to build for (default "
                         f"{C.PAIRS_DIR_T2.name}; archive/pairs is the archived task 1)")
    ap.add_argument("--cohorts", nargs="*", default=C.TASK2_COHORTS,
                    help="cohorts to read from --pairs-dir")
    args = ap.parse_args()

    d = C.load_dataset()
    rows, seqs = d["rows"], d["seqs"]

    scaffolds = {}
    for coh in args.cohorts:
        for r in C.read_pairs(coh, args.pairs_dir):
            scaffolds.setdefault(int(r["scaffold_idx"]), r["scaffold_pdb"])
    print(f"{len(scaffolds)} unique scaffolds across {len(args.cohorts)} cohorts "
          f"in {_os.path.basename(str(args.pairs_dir).rstrip('/'))}")

    prior = {} if args.rebuild or not OUT.exists() else json.loads(OUT.read_text())["windows"]
    windows = dict(prior)          # union across task sets -- see the module docstring
    todo = {si: pdb for si, pdb in scaffolds.items() if rows[si]["name"] not in windows}
    print(f"already in {OUT.name}: {len(windows)} window(s), "
          f"{len(scaffolds) - len(todo)} of them this task set's | to build: {len(todo)}")

    if todo:
        t0 = time.time()
        n_done = 0
        for si, pdb in todo.items():
            nm = rows[si]["name"]; seq = seqs[si]
            try:
                c1, catal, pocket, q, hbond = pockets.experimental_window(
                    nm, seq, pdb, cutoff=CUTOFF, hbond_cutoff=HBOND_CUTOFF,
                    return_quality=True, return_hbond=True, structdir=str(C.STRUCT_DIR))
            except Exception as e:
                print(f"  ! {nm} [{pdb}]: {type(e).__name__}: {str(e)[:120]} -- SKIPPED")
                continue

            editable0 = sorted([c1, c1 + 1] + list(pocket))
            fixed0 = sorted([c1 + 2] + list(catal))
            pc = {c1 + 1: list(AROMATIC)}
            for p in hbond:
                pc[p] = list(HBOND_AA)

            windows[nm] = {
                "scaffold_idx": si, "scaffold_pdb": pdb, "seq_len": len(seq),
                "chromophore": {"pos1_0based": c1, "pos2_0based": c1 + 1, "pos3_0based": c1 + 2},
                "catalytic_0based": list(catal),
                "pocket_0based": list(pocket),
                "hbond_partners_0based": list(hbond),
                "editable_0based": editable0,
                "fixed_0based": fixed0,
                "position_constraints": {str(k): v for k, v in pc.items()},
                "n_editable": len(editable0), "n_hbond": len(hbond),
                "structure_match": {"chain": q["chain"], "local_identity": round(q["local_id"], 3),
                                    "coverage": round(q["coverage"], 3)},
                "scaffold_seq": seq,
            }
            n_done += 1
            print(f"  [{n_done}/{len(todo)}] {nm} [{pdb}]: {len(editable0)} editable, "
                  f"{len(hbond)} hbond | {time.time()-t0:.0f}s", flush=True)
        print(f"built {n_done} windows in {time.time()-t0:.0f}s")

    built_here = [rows[si]["name"] for si in scaffolds if rows[si]["name"] in windows]
    meta_out = {
        "description": "Per-scaffold Tier-B design window (5 A chromophore pocket + H-bond "
                       "alphabet), built from scratch for this experiment's own cohorts. A union "
                       "over every task set built so far -- a window is a property of the "
                       "scaffold, not of the pair.",
        "cutoff_angstrom": CUTOFF, "hbond_cutoff_angstrom": HBOND_CUTOFF,
        "aromatic_alphabet": AROMATIC, "hbond_alphabet": HBOND_AA,
        "structure_cache": "structures/experimental",
        "n_scaffolds": len(windows),
        "generated_by": "in-silico-test/build_windows.py",
    }
    OUT.write_text(json.dumps({"meta": meta_out, "windows": windows}, indent=1))

    ned = [wv["n_editable"] for wv in windows.values()]
    print(f"\nwrote {len(windows)} scaffold windows -> {OUT} "
          f"({len(built_here)}/{len(scaffolds)} of them this task set's)")
    if ned:
        print(f"editable min/med/max = {min(ned)}/{int(np.median(ned))}/{max(ned)}")
    missing = [rows[si]["name"] for si in scaffolds if rows[si]["name"] not in windows]
    if missing:
        print(f"\n{len(missing)} scaffold(s) have no window "
              f"(failed the structure quality gate):\n  {missing}")
        print("this should not happen when curation filters on structure_validation.json -- refresh it "
              "with `python validate_structures.py --revalidate`, then re-run curate_pairs.py --from-cache")


if __name__ == "__main__":
    main()
