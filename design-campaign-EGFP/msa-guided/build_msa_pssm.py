#!/usr/bin/env python
"""Per-position family residue distributions for the EGFP Tier-B window.

Turns the 763-sequence family alignment into the generative prior this campaign samples
from, replacing ESM-2's masked-LM conditional. For every window position we emit the
weighted frequency of each of the 20 amino acids in the aligned column that EGFP's own
residue occupies.

Three choices carry the weight here:

* **Henikoff weights, not raw counts.** The dataset is a mutant library (hundreds of
  near-identical avGFP and DsRed descendants), so unweighted frequencies would report
  whatever the avGFP lineage happens to share. Weights are computed over the alignment
  core exactly as ``msa_conservation/conservation.py`` does, dropping the effective
  sample size from 763 to ~272.

* **Zero means forbidden.** A residue with weighted frequency 0 -- absent from every
  aligned FP at that column -- is dropped from the alphabet, not smoothed. The campaign
  can then never place it. Note this is a statement about 763 observed sequences at
  N_eff 272, so it means "the family never does this", not "this cannot fold".

* **EGFP's own column, found by alignment.** Positions map through EGFP's row in the
  MSA rather than through avGFP numbering plus an offset, so indels anywhere in the
  family cannot silently shift the mapping.

Writes ``msa_pssm_egfp.json``: per 0-based scaffold position, the surviving alphabet and
its renormalized frequencies, already intersected with the window's Tier-B constraints.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MSA = REPO / "msa_conservation"
sys.path.insert(0, str(MSA))
from conservation import (AAS, CLASS_OF, OCC_MIN, encode,  # noqa: E402
                          henikoff_weights, load_alignment, n_eff, weighted_freqs)

WINDOWS_JSON = HERE / "design_windows_egfp_tierB.json"
OUT = HERE / "msa_pssm_egfp.json"
SCAFFOLD = "EGFP"


def main():
    A, meta = load_alignment()
    code_full = encode(A)
    occ = (code_full >= 0).mean(0)
    core = np.nonzero(occ >= OCC_MIN)[0]

    # weights come from the core columns (same basis as the conservation analysis), but
    # frequencies are read off the FULL alignment so no window position can be missing.
    w = henikoff_weights(code_full[:, core])
    F = weighted_freqs(code_full, w)
    print(f"alignment {A.shape[0]} seq x {A.shape[1]} col | core {len(core)} | "
          f"N_eff(Henikoff) {n_eff(w):.1f}")

    wins = json.load(open(WINDOWS_JSON))["windows"]
    win = wins[SCAFFOLD]
    scaf = win["scaffold_seq"]

    row = meta.index[meta.seq == scaf]
    if not len(row):
        raise SystemExit(f"{SCAFFOLD} scaffold sequence is not in the MSA; cannot map positions")
    row = int(row[0])
    col_of = np.nonzero(A[row] != "-")[0]          # 0-based scaffold position -> alignment column
    if len(col_of) != len(scaf):
        raise SystemExit(f"row {row} ungaps to {len(col_of)} residues, scaffold has {len(scaf)}")
    print(f"{SCAFFOLD} is msa_id {meta.loc[row, 'msa_id']} (slug {meta.loc[row, 'slug']}), "
          f"{len(scaf)} residues mapped to alignment columns")

    constraints = {int(k): set(v) for k, v in win["position_constraints"].items()}
    core_set = set(core.tolist())

    out, report = {}, []
    for p in sorted(win["editable_0based"]):
        c = int(col_of[p])
        f = F[c]
        allowed = constraints.get(p)
        keep = [a for i, a in enumerate(AAS)
                if f[i] > 0 and (allowed is None or a in allowed)]
        if not keep:
            raise SystemExit(f"position {p+1} has an empty alphabet after intersecting "
                             f"the family support with its Tier-B constraint {allowed}")
        probs = np.array([f[AAS.index(a)] for a in keep])
        probs = probs / probs.sum()
        out[str(p)] = {"alphabet": "".join(keep),
                       "probs": [round(float(x), 8) for x in probs]}

        order = np.argsort(-probs)
        scaf_aa = scaf[p]
        report.append(dict(
            pos_0based=p, pos_1based=p + 1, scaffold_aa=scaf_aa,
            in_core=c in core_set, occupancy=round(float(occ[c]), 3),
            constraint=("aromatic" if allowed and len(allowed) == 4 else
                        "hbond" if allowed else "none"),
            n_family_support=int((f > 0).sum()),
            n_allowed=len(keep),
            n_blocked=int((f == 0).sum()) if allowed is None else 20 - len(keep),
            scaffold_aa_kept=scaf_aa in keep,
            scaffold_aa_prob=round(float(probs[keep.index(scaf_aa)]), 4) if scaf_aa in keep else 0.0,
            alphabet="".join(keep[i] for i in order),
            top3=" ".join(f"{keep[i]}:{probs[i]:.2f}" for i in order[:3]),
        ))

    meta_out = {
        "description": "Henikoff-weighted family residue frequencies per EGFP window position, "
                       "intersected with the Tier-B alphabet constraints. Zero-frequency "
                       "residues are dropped, so the campaign can never select them.",
        "source_alignment": "msa_conservation/data/fp_all.aln.fasta",
        "n_sequences": int(A.shape[0]), "n_eff_henikoff": round(n_eff(w), 1),
        "occ_min_core": OCC_MIN, "scaffold": SCAFFOLD,
        "windows_json": WINDOWS_JSON.name,
        "n_positions": len(out),
    }
    json.dump({"meta": meta_out, "pssm": out}, open(OUT, "w"), indent=1)

    R = pd.DataFrame(report)
    pd.set_option("display.width", 220)
    print(f"\n=== EGFP window: family alphabet per position ===")
    print(R[["pos_1based", "scaffold_aa", "constraint", "occupancy", "n_family_support",
             "n_allowed", "n_blocked", "scaffold_aa_prob", "alphabet", "top3"]].to_string(index=False))

    print(f"\nall {len(R)} positions in alignment core: {bool(R.in_core.all())}")
    print(f"scaffold residue survives everywhere: {bool(R.scaffold_aa_kept.all())}")
    print(f"alphabet size: min {R.n_allowed.min()}  median {R.n_allowed.median():.0f}  "
          f"max {R.n_allowed.max()}  (mean {R.n_allowed.mean():.1f} of 20)")
    print(f"residues blocked outright across the window: {int(R.n_blocked.sum())} "
          f"of {20*len(R)} position-residue combinations")
    print(f"-> {OUT.name}")


if __name__ == "__main__":
    main()
