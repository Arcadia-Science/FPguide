#!/usr/bin/env python
"""Pre-validate every candidate scaffold's structure so curate_pairs.py can pick valid pairs in
ONE pass, with no reselect/rebuild loop.

THE PROBLEM THIS SOLVES
-----------------------
``structure_hits.csv`` answers a purely SEQUENCE-level question: does some PDB entry exist whose
declared entity sequence matches this protein at >=97%? It never looks at coordinates. But a
usable design window needs a coordinate-level property: the atoms actually modelled in one chain
must map cleanly onto this exact dataset sequence, because the window is a set of 0-based indices
INTO that sequence, derived from a structure->sequence residue mapping. ``experimental_window``
therefore gates on local identity >=90% over >=70% coverage and raises rather than emit a
silently mis-numbered pocket.

Those two questions diverge routinely:
  * SEQRES describes the construct that went into the crystallization tube; the model describes
    what was resolved from the density. Disordered termini/loops or a truncated construct give
    high identity but ~65% coverage.
  * an entry may be a complex or fusion where the best-matching chain isn't the FP at all
    (e.g. 22MM's best chain matches mCherry at 28% identity / 45% coverage).
  * PDB entries are SHARED across near-identical dataset entries -- 23 of the usable scaffolds all
    point at 2G2S -- so one structure that matches none of its claimants eliminates a whole
    cluster at once, which is why failures otherwise arrive in family-shaped batches.

None of that is knowable without fetching, parsing, picking the best chain and aligning, i.e.
without attempting the build. So this script attempts it once for every candidate scaffold and
caches the verdict; curation then filters on the cache instead of discovering failures later.

It also checks the OTHER precondition build_windows.py can fail on: the scaffold sequence must be
present in the family alignment and ungap to its own length, since the PSSM is read through that
alignment row.

Cost: ~360 candidate scaffolds collapse to ~154 unique PDB entries, so this is a one-time cost
dominated by fetching the uncached ones (the RCSB cache in structures/experimental/ is reused).

Writes ``structure_validation.json``. Resumable: already-recorded scaffolds are skipped.

Usage
-----
    python validate_structures.py
    python validate_structures.py --revalidate       # recompute every verdict from zero
    python validate_structures.py --limit 20         # probe timing on a subset
"""
import argparse
import json
import sys
import time

import numpy as np

# --- stage-folder bootstrap: put the experiment root (design_common), lib/ (vendored
# --- modules) and msa/ (family alignment code) on the import path.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, "lib"), _os.path.join(_ROOT, "msa")]

import design_common as C

import pockets
from conservation import load_alignment  # noqa: E402

CUTOFF, HBOND_CUTOFF = 5.0, 3.5          # must match build_windows.py
MIN_ID, MIN_COV = 0.90, 0.70             # experimental_window's gate defaults, recorded for provenance
OUT = C.HERE / "structure_validation.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revalidate", action="store_true", help="recompute every verdict from zero")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N unvalidated scaffolds")
    args = ap.parse_args()

    d = C.load_dataset()
    rows, seqs = d["rows"], d["seqs"]
    split, hits = C.load_split(), C.load_hits()

    # candidates: structure-known AND our surrogate role in {train,val,test} -- the same scaffold
    # universe curate_pairs.py draws from.
    cand = sorted(i for i in hits if split.get(i, ("excluded",))[0] in ("train", "val", "test"))
    n_pdb = len({hits[i] for i in cand})
    print(f"{len(cand)} candidate scaffolds across {n_pdb} unique PDB entries")

    prior = {} if args.revalidate or not OUT.exists() else json.loads(OUT.read_text())["scaffolds"]
    todo = [i for i in cand if rows[i]["name"] not in prior]
    if args.limit:
        todo = todo[:args.limit]
    print(f"already validated: {len(cand) - len([i for i in cand if rows[i]['name'] not in prior])} "
          f"| to validate now: {len(todo)}")

    results = dict(prior)
    if todo:
        print("loading family alignment (for the PSSM-precondition check) ...")
        A, meta = load_alignment()
        aln_seqs = {}
        for r, s in zip(meta.index, meta.seq):
            aln_seqs.setdefault(s, int(r))

        t0 = time.time()
        for k, si in enumerate(todo):
            nm, pdb, seq = rows[si]["name"], hits[si], seqs[si]
            rec = {"scaffold_idx": si, "pdb_id": pdb, "seq_len": len(seq)}

            # ---- precondition 1: structure maps cleanly onto this sequence -------------------
            try:
                c1, catal, pocket, q, hbond = pockets.experimental_window(
                    nm, seq, pdb, cutoff=CUTOFF, hbond_cutoff=HBOND_CUTOFF,
                    return_quality=True, return_hbond=True, structdir=str(C.STRUCT_DIR))
                rec.update(structure_ok=True, chain=q["chain"],
                           local_identity=round(float(q["local_id"]), 3),
                           coverage=round(float(q["coverage"]), 3),
                           n_editable=len(sorted([c1, c1 + 1] + list(pocket))), n_hbond=len(hbond))
            except Exception as e:
                rec.update(structure_ok=False, reason=f"{type(e).__name__}: {str(e)[:200]}")

            # ---- precondition 2: sequence is in the family alignment, at its own length ------
            row = aln_seqs.get(seq)
            if row is None:
                rec.update(alignment_ok=False, alignment_reason="sequence absent from family alignment")
            else:
                n_ungapped = int((A[row] != "-").sum())
                if n_ungapped != len(seq):
                    rec.update(alignment_ok=False,
                               alignment_reason=f"alignment row ungaps to {n_ungapped}, sequence has {len(seq)}")
                else:
                    rec.update(alignment_ok=True)

            rec["ok"] = bool(rec.get("structure_ok") and rec.get("alignment_ok"))
            results[nm] = rec

            if not rec["ok"] or (k + 1) % 25 == 0:
                tag = "ok " if rec["ok"] else "BAD"
                why = "" if rec["ok"] else f" -- {rec.get('reason') or rec.get('alignment_reason')}"
                print(f"  [{k+1}/{len(todo)}] {tag} {nm} [{pdb}]{why[:150]} | {time.time()-t0:.0f}s", flush=True)

            if (k + 1) % 50 == 0:      # checkpoint so a long run is never lost
                OUT.write_text(json.dumps({"meta": _meta(results), "scaffolds": results}, indent=1))

    OUT.write_text(json.dumps({"meta": _meta(results), "scaffolds": results}, indent=1))

    ok = [r for r in results.values() if r["ok"]]
    bad = {nm: r for nm, r in results.items() if not r["ok"]}
    print(f"\n{len(ok)}/{len(results)} scaffolds usable -> {OUT}")

    struct_fail = {nm: r for nm, r in bad.items() if not r.get("structure_ok")}
    aln_fail = {nm: r for nm, r in bad.items() if r.get("structure_ok") and not r.get("alignment_ok")}
    print(f"  {len(struct_fail)} failed the structure gate (local id >= {MIN_ID:.0%}, coverage >= {MIN_COV:.0%})")
    print(f"  {len(aln_fail)} failed the family-alignment precondition")

    if struct_fail:
        by_pdb = {}
        for nm, r in struct_fail.items():
            by_pdb.setdefault(r["pdb_id"], []).append(nm)
        print("\n  worst offending PDB entries (one bad structure can eliminate a whole family):")
        for pdb, names in sorted(by_pdb.items(), key=lambda kv: -len(kv[1]))[:8]:
            print(f"    {pdb}: {len(names)} scaffold(s) -- e.g. {names[:3]}")

    per_role = {}
    for nm, r in results.items():
        role = split[r["scaffold_idx"]][0]
        per_role.setdefault(role, [0, 0])
        per_role[role][1] += 1
        per_role[role][0] += int(r["ok"])
    print("\n  usable per surrogate role: " +
          " | ".join(f"{role} {a}/{b}" for role, (a, b) in sorted(per_role.items())))


def _meta(results):
    return {
        "description": "Per-scaffold verdict on whether a usable design window can be built: the "
                       "deposited structure must map cleanly onto the dataset sequence, and the "
                       "sequence must be present in the family alignment at its own length. Lets "
                       "curate_pairs.py select valid pairs in one pass instead of discovering "
                       "failures during build_windows.py.",
        "gate_min_local_identity": MIN_ID, "gate_min_coverage": MIN_COV,
        "cutoff_angstrom": CUTOFF, "hbond_cutoff_angstrom": HBOND_CUTOFF,
        "n_scaffolds": len(results), "n_ok": sum(1 for r in results.values() if r["ok"]),
        "generated_by": "in-silico-test/validate_structures.py",
    }


if __name__ == "__main__":
    main()
