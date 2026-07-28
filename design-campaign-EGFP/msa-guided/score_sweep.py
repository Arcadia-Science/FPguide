#!/usr/bin/env python
"""Score every cell of the MSA-guided sweep and report the grid.

Reads ``designs/lam-*/design_EGFP-*.csv`` and writes ``sweep_metrics.csv`` (one row per
cell x target) plus ``sweep_designs.csv`` (one row per final-round design). The lambda
weights come from the FOLDER NAME, which is the only place all four are recorded -- the
engine has no lam_edit column.

Two checks run alongside the scoring, because they are the claims this strategy rests on:
every edit is verified against the family support it was supposed to be confined to, and
the family log-likelihood of each design is computed under the same PSSM (the cheap,
GPU-free analogue of the pseudo-perplexity this campaign skips).
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CELL_RE = re.compile(r"lam-ex([\d.]+)_lam-em([\d.]+)_lam-bright([\d.]+)_lam-edit([\d.]+)")
BRIGHT_T = 0.0        # classifier logit > 0 == "bright" class


def main():
    blob = json.load(open(HERE / "msa_pssm_egfp.json"))["pssm"]
    allowed = {int(p): set(e["alphabet"]) for p, e in blob.items()}
    logp = {int(p): {a: float(np.log(x)) for a, x in zip(e["alphabet"], e["probs"])}
            for p, e in blob.items()}

    cells, designs, n_edit, n_bad = [], [], 0, 0
    for f in sorted(glob.glob(str(HERE / "designs" / "*" / "design_EGFP-*.csv"))):
        cell = Path(f).parent.name
        m = CELL_RE.match(cell)
        if not m:
            continue
        lam_ex, lam_em, lam_b, lam_e = (float(x) for x in m.groups())
        tgt = re.search(r"EGFP-(\w+)\.csv", f).group(1)
        d = pd.read_csv(f)
        fin = d[d["round"] == d["round"].max()].copy()

        muts, viol, fam_ll = [], [], []
        for r in fin.itertuples():
            ms = [(i, a, b) for i, (a, b) in enumerate(zip(r.scaffold_seq, r.designed_seq)) if a != b]
            muts.append(len(ms))
            viol.append(sum(1 for i, _, b in ms if b not in allowed.get(i, set())))
            fam_ll.append(np.mean([logp[i][r.designed_seq[i]] for i in allowed]))
        fin["n_mut"] = muts
        fin["n_violation"] = viol
        fin["fam_logp"] = fam_ll
        n_edit += int(np.sum(muts)); n_bad += int(np.sum(viol))

        fin["cell"] = cell
        fin["lam_peaks"] = lam_ex; fin["lam_bright_"] = lam_b; fin["lam_edit"] = lam_e
        designs.append(fin[["cell", "target_name", "lam_peaks", "lam_bright_", "lam_edit",
                            "trial", "peak_err", "pred_ex", "pred_em", "pred_bright",
                            "n_mut", "n_violation", "fam_logp", "designed_seq"]])

        cells.append(dict(
            cell=cell, target=tgt, lam_peaks=lam_ex, lam_bright=lam_b, lam_edit=lam_e,
            err_mean=fin.peak_err.mean(), err_best=fin.peak_err.min(), err_sd=fin.peak_err.std(),
            bright_mean=fin.pred_bright.mean(),
            n_bright=int((fin.pred_bright > BRIGHT_T).sum()), n=len(fin),
            n_mut=fin.n_mut.mean(), fam_logp=fin.fam_logp.mean(),
            n_distinct=fin.designed_seq.nunique(), n_violation=int(fin.n_violation.sum())))

    C = pd.DataFrame(cells).sort_values(["target", "err_mean"])
    D = pd.concat(designs, ignore_index=True)
    C.to_csv(HERE / "sweep_metrics.csv", index=False)
    D.to_csv(HERE / "sweep_designs.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"{len(C)} cell x target results, {len(D)} final-round designs")
    print(f"family-support violations: {n_bad} / {n_edit} edits "
          f"({'CLEAN' if n_bad == 0 else 'FAILED'})\n")

    print("mean peak error (nm) by lambda_peaks:")
    print(C.pivot_table(index="target", columns="lam_peaks", values="err_mean").round(1).to_string())
    print("\nfraction of designs predicted bright, by lambda_bright:")
    print((C.pivot_table(index="target", columns="lam_bright", values="n_bright") / 12)
          .round(2).to_string())
    print("\nmean mutations from scaffold, by lambda_edit:")
    print(C.pivot_table(index="target", columns="lam_edit", values="n_mut").round(1).to_string())

    show = ["lam_peaks", "lam_bright", "lam_edit", "err_mean", "err_best", "bright_mean",
            "n_bright", "n_mut", "fam_logp", "n_distinct"]
    for t in sorted(C.target.unique()):
        s = C[C.target == t]
        print(f"\n=== {t}: top 5 by peak error (accuracy-optimal) ===")
        print(s.nsmallest(5, "err_mean")[show].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        ok = s[s.n_bright >= 6]
        print(f"=== {t}: top 5 with >=6/12 predicted bright (usable) ===")
        print(ok.nsmallest(5, "err_mean")[show].to_string(index=False, float_format=lambda x: f"{x:.2f}")
              if len(ok) else "  none")

    print("\ncorrelations across cells (why the two optima differ):")
    for t in sorted(C.target.unique()):
        s = C[C.target == t]
        print(f"  {t}: n_mut vs bright {s.n_mut.corr(s.bright_mean):+.2f} | "
              f"n_mut vs err {s.n_mut.corr(s.err_mean):+.2f} | "
              f"fam_logp vs bright {s.fam_logp.corr(s.bright_mean):+.2f} | "
              f"fam_logp vs err {s.fam_logp.corr(s.err_mean):+.2f}")
    print("\n-> sweep_metrics.csv, sweep_designs.csv")


if __name__ == "__main__":
    main()
