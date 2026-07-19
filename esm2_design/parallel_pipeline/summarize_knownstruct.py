#!/usr/bin/env python
"""Summarize the known-structure design run, split by cohort (S-train vs S-test).
Per task the "best" round is the round>=1 with the lowest ORACLE peak error. Writes
summary.csv per cohort dir and prints an aggregate table + the S-train/S-test contrast.
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C

COHORTS = ["knownstruct_Strain_Otrain", "knownstruct_Sval_Otrain", "knownstruct_Stest_Otrain"]


def summarize_file(fn):
    H = sorted(csv.DictReader(open(fn)), key=lambda h: int(h["round"]))
    scaf = H[0]
    gen = [h for h in H if int(h["round"]) >= 1]
    best = min(gen, key=lambda h: float(h["peak_err"]))
    return dict(
        name=best["example"], cohort=best["cohort"], pdb=best["scaffold_pdb"],
        seq_id=float(scaf["seq_id_scaf_target"]),
        scaf_err=float(scaf["peak_err"]), best_round=int(best["round"]),
        best_err=float(best["peak_err"]), d_err=float(best["peak_err"]) - float(scaf["peak_err"]),
        ppl0=float(scaf["ppl"]), ppl=float(best["ppl"]), id_scaf=float(best["ident_to_scaffold"]),
    )


def main():
    all_summ = []
    for coh in COHORTS:
        outdir = os.path.join(C.PIPE_OUT, coh)
        files = sorted(glob.glob(os.path.join(outdir, "design_*.csv")))
        summ = [summarize_file(fn) for fn in files]
        summ.sort(key=lambda d: d["d_err"])
        if summ:
            with open(os.path.join(outdir, "summary.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
        all_summ.append((coh, summ))

    for coh, summ in all_summ:
        if not summ:
            print(f"\n### {coh}: no CSVs"); continue
        imp = sum(1 for s in summ if s["d_err"] < 0)
        print(f"\n### {coh}  (n={len(summ)})")
        print(f"  improved (oracle err down): {imp}/{len(summ)}")
        print(f"  mean oracle err: scaffold {np.mean([s['scaf_err'] for s in summ]):.1f} -> "
              f"best {np.mean([s['best_err'] for s in summ]):.1f} nm  (mean Delta {np.mean([s['d_err'] for s in summ]):+.1f})")
        print(f"  mean identity-to-scaffold retained: {np.mean([s['id_scaf'] for s in summ]):.1%}")
        print(f"  mean ppl: {np.mean([s['ppl0'] for s in summ]):.1f} -> {np.mean([s['ppl'] for s in summ]):.1f}")
        print(f"  {'task':42}{'id':>5}{'scaf':>7}{'bestR':>6}{'best':>7}{'Delta':>7}{'id_sc':>7}{'ppl':>6}")
        for s in summ:
            print(f"  {s['name'][:41]:42}{s['seq_id']:>5.0%}{s['scaf_err']:>7.1f}{s['best_round']:>6}"
                  f"{s['best_err']:>7.1f}{s['d_err']:>+7.1f}{s['id_scaf']:>7.0%}{s['ppl']:>6.1f}")

    flat = [s for _, summ in all_summ for s in summ]
    print(f"\n=== combined (n={len(flat)}) : improved {sum(1 for s in flat if s['d_err']<0)}/{len(flat)} | "
          f"mean Delta {np.mean([s['d_err'] for s in flat]):+.1f} nm | "
          f"mean id retained {np.mean([s['id_scaf'] for s in flat]):.0%} ===")
    print("summarize done.")


if __name__ == "__main__":
    main()
