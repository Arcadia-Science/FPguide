#!/usr/bin/env python
"""Compute the design (edit) window for each of the 24 campaign scaffolds using the
SAME rules as peak_design/ (structure-based 5 A contact shell) and save them to a single
portable JSON so any design algorithm can load them without re-deriving.

Window rule (identical to esm2_design/pockets.py + guided_design_peak_structure_*):
  * chromophore = X-[YWHF]-G motif near seq position 50-85 (closest to ~65) -> pos1,pos2,pos3;
  * EDITABLE = chromophore pos1 & pos2  +  every residue with a heavy atom within 5.0 A of the
    chromophore reference atoms (fused hetero-chromophore if modelled, else the tripeptide),
    read off the scaffold's EXPERIMENTAL RCSB structure and mapped to the dataset sequence;
  * pos2 is constrained to aromatics {Y,W,H,F};
  * FIXED / protected = pos3 (Gly) and the maturation-catalytic Arg+Glu (nearest to the
    chromophore) -- never edited.

TIER-B (this variant): residues whose side-chain polar (N/O) atom is within 3.5 A of a
chromophore polar (N/O) atom are flagged as chromophore H-bond partners. They stay EDITABLE
but their alphabet is restricted to the H-bond-capable set {S,T,Y,N,Q,D,E,H,K,R,W} via
position_constraints, so a design can retune the wavelength while preserving H-bond
CAPABILITY (not identity). This is layered on top of the pos2->aromatic constraint and does
not change the editable set (H-bond partners are already inside the 5 A pocket).

Output: design_windows_24_tierB.json
  { "meta": {...}, "windows": { "<scaffold>": { chromophore, catalytic, pocket, editable,
    fixed, position_constraints, scaffold_seq, ... } } }
All position lists are 0-based into scaffold_seq; *_1based mirrors are included for convenience.
"""
import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent      # .../fpdesign
REPO = HERE.parent                           # .../spectrum-to-fp-design
CAMPAIGN = REPO / "design-campaign-conventional"   # default home for pairs list + output JSON
CUR = REPO / "dataset_pipeline" / "data" / "peak" / "curated"
sys.path.insert(0, str(REPO / "esm2_design"))
import pockets                     # noqa: E402

# reuse the established experimental-structure cache so PDBs aren't re-downloaded
STRUCTDIR = str(REPO / "esm2_design" / "structures" / "experimental")
AROMATIC = ["Y", "W", "H", "F"]
# Tier-B: residues that can donate/accept a hydrogen bond via their side chain. A chromophore
# H-bond partner is kept EDITABLE but restricted to this set, so the design can retune the
# wavelength without silently deleting a donor/acceptor (which would shift protonation / kill
# brightness / impair maturation -- failure modes the (ex, em) surrogate cannot see).
HBOND_AA = ["S", "T", "Y", "N", "Q", "D", "E", "H", "K", "R", "W"]
CUTOFF = pockets.CUTOFF            # 5.0 A
HBOND_CUTOFF = 3.5                 # heavy-atom donor..acceptor distance for an H-bond


def load_seqs():
    rows = list(csv.DictReader(open(CUR / "peaks_assignments.csv")))
    seqs = [None] * len(rows)
    h = None
    for line in open(CUR / "sequences.fasta"):
        line = line.strip()
        if line.startswith(">"):
            h = int(line[1:].split("|")[0])
        elif line:
            seqs[h] = line
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(CAMPAIGN / "pairs" / "campaign_pairs_24.csv"))
    ap.add_argument("--out", default=str(CAMPAIGN / "design_windows_24_tierB.json"))
    ap.add_argument("--cutoff", type=float, default=CUTOFF)
    ap.add_argument("--hbond-cutoff", type=float, default=HBOND_CUTOFF,
                    help="heavy-atom N/O donor..acceptor distance to flag a Tier-B H-bond partner")
    args = ap.parse_args()

    seqs = load_seqs()
    pairs = list(csv.DictReader(open(args.pairs)))
    print(f"computing {args.cutoff:.0f} A windows + {args.hbond_cutoff:.1f} A Tier-B H-bond partners "
          f"for {len(pairs)} scaffolds ...")

    windows = {}
    t0 = time.time()
    for r in pairs:
        si = int(r["scaffold_idx"]); nm = r["scaffold_name"]; pdb = r["scaffold_pdb"]
        seq = seqs[si]
        try:
            c1, catal, pocket, q, hbond = pockets.experimental_window(
                nm, seq, pdb, cutoff=args.cutoff, hbond_cutoff=args.hbond_cutoff,
                return_quality=True, return_hbond=True, structdir=STRUCTDIR)
        except Exception as e:
            print(f"  ! {nm} [{pdb}]: {type(e).__name__}: {str(e)[:110]}")
            continue
        tri0 = [c1, c1 + 1, c1 + 2]
        editable0 = sorted([c1, c1 + 1] + list(pocket))          # pos1, pos2, and the 5 A pocket
        fixed0 = sorted([c1 + 2] + list(catal))                  # Gly + catalytic Arg/Glu
        # position_constraints keys are 0-BASED (the campaign looks them up by 0-based index):
        #   pos2 -> aromatics; each Tier-B H-bond partner -> H-bond-capable residues only.
        pc = {str(c1 + 1): AROMATIC}
        for p in hbond:
            pc[str(p)] = HBOND_AA
        windows[nm] = {
            "scaffold_idx": si,
            "scaffold_pdb": pdb,
            "seq_len": len(seq),
            "chromophore": {
                "pos1_0based": c1, "pos2_0based": c1 + 1, "pos3_0based": c1 + 2,
                "pos1_1based": c1 + 1, "pos2_1based": c1 + 2, "pos3_1based": c1 + 3,
                "residues": f"{seq[c1]}{c1+1}-{seq[c1+1]}{c1+2}-{seq[c1+2]}{c1+3}",
            },
            "catalytic_0based": list(catal),
            "catalytic_1based": [c + 1 for c in catal],
            "catalytic_residues": [f"{seq[c]}{c+1}" for c in catal],
            "pocket_0based": list(pocket),
            "pocket_1based": [p + 1 for p in pocket],
            "hbond_partners_0based": list(hbond),
            "hbond_partners_1based": [p + 1 for p in hbond],
            "hbond_residues": [f"{seq[p]}{p+1}" for p in hbond],
            "editable_0based": editable0,
            "editable_1based": [p + 1 for p in editable0],
            "fixed_0based": fixed0,
            "fixed_1based": [p + 1 for p in fixed0],
            "position_constraints": pc,                          # pos2->aromatic + Tier-B H-bond
            "n_editable": len(editable0),
            "n_pocket": len(pocket),
            "n_hbond": len(hbond),
            "structure_match": {"chain": q["chain"], "local_identity": round(q["local_id"], 3),
                                "coverage": round(q["coverage"], 3)},
            "scaffold_seq": seq,
        }

    meta = {
        "description": "Structure-based 5 A chromophore edit window per campaign scaffold "
                       "(same rule as esm2_design/pockets.py) + Tier-B H-bond-partner alphabet "
                       "restriction.",
        "rule": "editable = chromophore pos1 & pos2 + residues with a heavy atom within CUTOFF of the "
                "chromophore; pos2 constrained to aromatics {Y,W,H,F}; pos3 (Gly) and catalytic Arg+Glu "
                "fixed; Tier-B chromophore H-bond partners (side-chain N/O within HBOND_CUTOFF of a "
                "chromophore N/O) stay editable but restricted to H-bond-capable residues.",
        "cutoff_angstrom": args.cutoff,
        "hbond_cutoff_angstrom": args.hbond_cutoff,
        "chromophore_motif": "X-[YWHF]-G near sequence position 50-85 (closest to ~65)",
        "structure_source": "experimental RCSB structure (>=97% identity), per-scaffold PDB id",
        "position_indexing": "0-based into scaffold_seq; *_1based mirrors provided; "
                             "position_constraints keys are 0-based",
        "aromatic_alphabet": AROMATIC,
        "hbond_alphabet": HBOND_AA,
        "tier_b_note": "H-bond partners are a CAPABILITY proxy (heavy-atom N/O distance only; no "
                       "explicit H, angle, or water-mediated bridges) -- not ground-truth H-bonds.",
        "n_scaffolds": len(windows),
        "generated_by": "fpdesign/build_design_windows.py",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(args.out, "w") as fh:
        json.dump({"meta": meta, "windows": windows}, fh, indent=2)

    npk = [w["n_pocket"] for w in windows.values()]
    ned = [w["n_editable"] for w in windows.values()]
    nhb = [w["n_hbond"] for w in windows.values()]
    print(f"wrote {len(windows)} windows in {time.time()-t0:.0f}s -> {args.out}")
    print(f"pocket size min/med/max = {min(npk)}/{sorted(npk)[len(npk)//2]}/{max(npk)} | "
          f"editable min/med/max = {min(ned)}/{sorted(ned)[len(ned)//2]}/{max(ned)} | "
          f"Tier-B hbond min/med/max = {min(nhb)}/{sorted(nhb)[len(nhb)//2]}/{max(nhb)}")
    print(f"\n{'scaffold':22}{'PDB':6}{'chromo':>12}{'catalytic':>14}{'n_edit':>7}{'n_hb':>6}  hbond_partners")
    for nm, w in windows.items():
        print(f"{nm[:21]:22}{w['scaffold_pdb']:6}{w['chromophore']['residues']:>12}"
              f"{','.join(w['catalytic_residues']):>14}{w['n_editable']:>7}{w['n_hbond']:>6}  "
              f"{','.join(w['hbond_residues'])}")


if __name__ == "__main__":
    main()
