#!/usr/bin/env python
"""Build Tier-B design windows + per-scaffold family PSSMs FROM SCRATCH for this experiment's
own scaffold/target pairs (``curate_pairs.py`` -> ``pairs/``).

Everything is computed locally and nothing is reused from other pipeline folders: the pocket
geometry comes from ``pockets.py`` against the local ``structures/experimental/`` RCSB cache
(self-populating -- a cache miss is fetched from RCSB), and the family PSSM comes from
``msa/conservation.py`` against the local ``msa/data/fp_all.aln.fasta`` alignment. Building
from scratch keeps this folder standalone and makes the windows reproducible from its own
inputs alone, at the cost of recomputing geometry for scaffolds other folders happen to share.

The window rule itself (unchanged from the Tier-B + family-support design it was adapted from):
  * editable  = chromophore positions 1-2 + every residue with an atom within 5 A of the
                chromophore ("Tier-B" pocket)
  * fixed     = chromophore position 3 + the catalytic residues
  * per-position alphabet is constrained to aromatics at chromophore position 2 and to H-bond
    capable residues at positions H-bonding the chromophore, then intersected with the residues
    the whole-family alignment actually supports at that column (Henikoff-weighted frequencies).
    A position with empty intersection falls back to the constraint alone (or the wild-type).

Every scaffold reaching this point should already be buildable: ``validate_structures.py`` checks
the structure-quality gate (>=90% local identity, >=70% coverage) and the family-alignment
precondition for every candidate, and ``curate_pairs.py`` selects only from the ones that passed.
A skip here therefore means the validation cache is stale relative to ``structures/experimental/``,
or curation was run with ``--no-validation``.

Re-running is resumable: any scaffold already present in ``design_windows.json`` is kept as-is
and only missing ones are computed. Use ``--rebuild`` to recompute everything from zero.

Usage
-----
    python build_windows.py
    python build_windows.py --rebuild
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
from conservation import AAS, OCC_MIN, encode, henikoff_weights, load_alignment, weighted_freqs  # noqa: E402

CUTOFF, HBOND_CUTOFF = 5.0, 3.5
AROMATIC = ["Y", "W", "H", "F"]
HBOND_AA = ["S", "T", "Y", "N", "Q", "D", "E", "H", "K", "R", "W"]
OUT = C.WINDOWS_JSON


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="recompute every window from zero")
    args = ap.parse_args()

    d = C.load_dataset()
    rows, seqs = d["rows"], d["seqs"]

    scaffolds = {}
    for coh in C.DEFAULT_COHORTS:
        for r in C.read_pairs(coh):
            scaffolds.setdefault(int(r["scaffold_idx"]), r["scaffold_pdb"])
    print(f"{len(scaffolds)} unique scaffolds across {len(C.DEFAULT_COHORTS)} cohorts")

    prior = {} if args.rebuild or not OUT.exists() else json.loads(OUT.read_text())["windows"]
    windows = {rows[si]["name"]: prior[rows[si]["name"]] for si in scaffolds if rows[si]["name"] in prior}
    todo = {si: pdb for si, pdb in scaffolds.items() if rows[si]["name"] not in windows}
    print(f"already built (resumed from {OUT.name}): {len(windows)} | to build: {len(todo)}")

    if todo:
        print(f"loading family alignment from {C.MSA_DIR / 'data' / 'fp_all.aln.fasta'} ...")
        A, meta = load_alignment()
        code_full = encode(A)
        occ = (code_full >= 0).mean(0)
        core = np.nonzero(occ >= OCC_MIN)[0]
        w = henikoff_weights(code_full[:, core])
        F = weighted_freqs(code_full, w)
        print(f"  alignment {A.shape[0]} seq x {A.shape[1]} col | core {len(core)} col "
              f"| N_eff(Henikoff) {float(w.sum()**2 / (w**2).sum()):.1f}")

        t0 = time.time()
        n_fallback, n_done = 0, 0
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

            row = meta.index[meta.seq == seq]
            if not len(row):
                print(f"  ! {nm}: scaffold sequence not found in the family alignment -- SKIPPED")
                continue
            row = int(row[0])
            col_of = np.nonzero(A[row] != "-")[0]
            if len(col_of) != len(seq):
                print(f"  ! {nm}: alignment row ungaps to {len(col_of)} residues, "
                      f"scaffold has {len(seq)} -- SKIPPED")
                continue

            pssm = {}
            for p in editable0:
                f = F[int(col_of[p])]
                constraint = pc.get(p)
                support = [a for i, a in enumerate(AAS) if f[i] > 0]
                keep = [a for a in support if constraint is None or a in constraint]
                used_fallback = False
                if not keep:
                    keep = list(constraint) if constraint else [seq[p]]
                    probs = np.full(len(keep), 1.0 / len(keep))
                    used_fallback = True
                    n_fallback += 1
                else:
                    probs = np.array([f[AAS.index(a)] for a in keep])
                    probs = probs / probs.sum()
                pssm[str(p)] = {"alphabet": "".join(keep),
                                "probs": [round(float(x), 8) for x in probs],
                                "fallback": used_fallback}

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
                "pssm": pssm,
            }
            n_done += 1
            print(f"  [{len(windows)}/{len(scaffolds)}] {nm} [{pdb}]: {len(editable0)} editable, "
                  f"{len(hbond)} hbond | {time.time()-t0:.0f}s", flush=True)
        print(f"built {n_done} windows ({n_fallback} position fallbacks) in {time.time()-t0:.0f}s")

    meta_out = {
        "description": "Per-scaffold Tier-B design window (5 A chromophore pocket + H-bond "
                       "alphabet) intersected with whole-family MSA support, built from scratch "
                       "for this experiment's own distance-spread cohorts "
                       "(knownstruct_Strain/Sval/Stest from curate_pairs.py).",
        "cutoff_angstrom": CUTOFF, "hbond_cutoff_angstrom": HBOND_CUTOFF,
        "aromatic_alphabet": AROMATIC, "hbond_alphabet": HBOND_AA,
        "source_alignment": "msa/data/fp_all.aln.fasta",
        "structure_cache": "structures/experimental",
        "n_scaffolds": len(windows),
        "generated_by": "in-silico-test/build_windows.py",
    }
    OUT.write_text(json.dumps({"meta": meta_out, "windows": windows}, indent=1))

    ned = [wv["n_editable"] for wv in windows.values()]
    print(f"\nwrote {len(windows)}/{len(scaffolds)} scaffold windows -> {OUT}")
    if ned:
        print(f"editable min/med/max = {min(ned)}/{int(np.median(ned))}/{max(ned)}")
    missing = [rows[si]["name"] for si in scaffolds if rows[si]["name"] not in windows]
    if missing:
        print(f"\n{len(missing)} scaffold(s) have no window (failed the structure quality gate or "
              f"are absent from the family alignment):\n  {missing}")
        print("this should not happen when curation filters on structure_validation.json -- refresh it "
              "with `python validate_structures.py --revalidate`, then re-run curate_pairs.py --from-cache")


if __name__ == "__main__":
    main()
