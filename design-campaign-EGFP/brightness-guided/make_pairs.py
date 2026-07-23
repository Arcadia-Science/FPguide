#!/usr/bin/env python
"""Build the pairs CSV for the EGFP brightness-guided design campaign.

Identical scaffold + target set to design-campaign-EGFP/make_pairs.py (the brightness campaign
mirrors that one and differs ONLY in its guidance signal): scaffold = EGFP, four commercially
available (patent-expired -> open_likely in licensing/commercial_use_flags.csv), non-LSS targets
spanning distinct spectral bands blue / green / orange / red -- deliberately NO large-Stokes-shift
proteins.

    EGFP (idx 171, PDB 4EUL, ex 488 / em 507) ->
      EBFP     (156, ex 380 / em 440)  blue    avGFP lineage (Clontech, US5,777,079 expired)
      mEmerald (407, ex 487 / em 509)  green   avGFP lineage (Clontech, expired) -- near neighbour
      mOrange  (479, ex 548 / em 562)  orange  DsRed/mFruit lineage (Tsien, US7,687,614 expired 2021)
      mCherry  (389, ex 587 / em 610)  red     DsRed/mFruit lineage (Tsien, US7,687,614 expired 2021)

Output columns match design-campaign-conventional/pairs/campaign_pairs_24.csv so the shared
fpdesign engine loads it unchanged. ``identity`` is the global scaffold<->target sequence identity
(metadata only; the edit window lives entirely on the EGFP scaffold, so no identity band is imposed).
"""
import csv
from pathlib import Path

import biotite.sequence as bseq
import biotite.sequence.align as balign

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CUR = REPO / "dataset_pipeline" / "data" / "peak" / "curated"
OUT = HERE / "pairs" / "campaign_pairs_egfp.csv"

SCAFFOLD_IDX = 171          # EGFP
SCAFFOLD_PDB = "4EUL"       # experimental structure used for the current campaign's EGFP window
TARGET_IDX = [156, 407, 479, 389]   # EBFP, mEmerald, mOrange, mCherry (spectral order blue->red)

_MAT = balign.SubstitutionMatrix.std_protein_matrix()


def seq_identity(a, b):
    aln = balign.align_optimal(bseq.ProteinSequence(a), bseq.ProteinSequence(b),
                               _MAT, gap_penalty=(-10, -1))[0]
    return float(balign.get_sequence_identity(aln))


def main():
    recs = {}
    for r in csv.DictReader(open(CUR / "peaks_assignments.csv")):
        i = int(r["index"])
        recs[i] = dict(name=r["name"], ex=float(r["ex_max"]), em=float(r["em_max"]), seq=r["seq"])
    s = recs[SCAFFOLD_IDX]
    s_ss = s["em"] - s["ex"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scaffold_idx", "scaffold_name", "scaffold_SS", "scaffold_pdb",
                    "target_idx", "target_name", "target_SS", "identity", "selection"])
        for ti in TARGET_IDX:
            t = recs[ti]
            idv = seq_identity(s["seq"], t["seq"])
            w.writerow([SCAFFOLD_IDX, s["name"], f"{s_ss:.0f}", SCAFFOLD_PDB,
                        ti, t["name"], f"{t['em']-t['ex']:.0f}", f"{idv:.4f}", "forced"])
            print(f"  EGFP -> {t['name']:10} ex {t['ex']:.0f} em {t['em']:.0f} "
                  f"SS {t['em']-t['ex']:.0f} | identity {idv:.0%}")
    print(f"wrote {len(TARGET_IDX)} pairs -> {OUT}")


if __name__ == "__main__":
    main()
