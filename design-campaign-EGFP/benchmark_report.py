#!/usr/bin/env python
"""Equal-budget benchmark across the five surviving design strategies.

Every strategy is given a comparable RAW-DESIGN budget of >= 1,125 designs per target
(raw = cells x trials x 3 iterations), so the cross-strategy comparison is not confounded by how
many designs each one was allowed to draw:

    1 gibbs        gibbs-sampling/designs_benchmark375/       375 trials x 3   = 1,125
    2 spectra      esm2_guided/, lam_bright=0 lam_edit=0        5 x  75 x 3   = 1,125
    4 DMS guide    esm2_guided/, lam_bright>0                 100 x   4 x 3   = 1,200
    5 MSA guide    msa-guided/designs/ (UNCHANGED)            125 x  12 x 3   = 4,500
    6 MSA gibbs    msa-gibbs/designs_benchmark375/            375 trials x 3   = 1,125

STRATEGY 3 (constrained spectra guide, lam_bright=0 & lam_edit>0) WAS RETIRED from the campaign.
Its 20 matched-sweep cells were run and are still on disk, but they feed no analysis -- here, in
``esm2_guided/analyze.py``, or anywhere else. The numbering is left with a gap on purpose
so that "strategy 4" keeps meaning what it has always meant in this campaign's write-ups.

STRATEGY 5 IS DELIBERATELY NOT EQUALISED -- it is left at its full 12 trials/cell, i.e. FOUR TIMES
the budget of every other row, and is flagged as such in the output. Read its row as a
best-achievable reference, not as a like-for-like competitor. Its per-trial seeding
(``SEED + si*131 + trial*17``) would make a 3-trial subsample exact if an equal-budget row is ever
wanted.

Strategies 1 and 6 are TARGET-FREE: one run serves both targets, so their per-target pool is the
whole 1,125-design run scored against each target in turn. That is the correct parity with the
guided strategies' 1,125 per target, not half of it.

Filters are imported from ``make_shortlist_case.py`` rather than reimplemented -- the ID test
(ESM max-pool NN-distance to the 40k GFP-DMS reference <= p99), the confident-brightness bar
(BRIGHT_T = 0.5), the greedy diverse top-10 (N=10, MIN_HD=5) and the on-disk embedding cache. The
target-free CSVs carry no ``pred_bright`` column, so their brightness is scored with the same
classifier every other strategy is judged by, exactly as ``build()`` does for the gibbs cases.

READ-ONLY. Touches no design tree, no shortlist, and nothing under fpdesign/.

    python benchmark_report.py                # both targets
    python benchmark_report.py mOrange        # one
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CAMP = Path(__file__).resolve().parent
sys.path.insert(0, str(CAMP))
import make_shortlist_case as msc  # noqa: E402

ESM2 = CAMP / "esm2_guided" / "designs"
CELL = re.compile(r"lam-ex([\d.]+)_lam-em[\d.]+_lam-bright([\d.]+)_lam-edit([\d.]+)$")


def esm2_slice(target, keep):
    out = []
    for f in sorted(ESM2.glob(f"*/design_EGFP-{target}.csv")):
        _, b, e = (float(x) for x in CELL.match(f.parent.name).groups())
        if keep(b, e):
            out.append(f)
    return out


def pools(target):
    """(label, paths, target_free, note) per strategy, in campaign order."""
    return [
        ("1 gibbs (ESM-2)",
         [CAMP / "gibbs-sampling" / "designs_benchmark375" / "design_EGFP.csv"], True, ""),
        ("2 spectra guide",
         esm2_slice(target, lambda b, e: b == 0 and e == 0), False, ""),
        # Strategy 3 (constrained spectra guide, lam_bright=0 & lam_edit>0) was retired from the
        # campaign. Its 20 cells were run and remain on disk, but they are excluded from every
        # analysis -- so the 105 cells reported here are 5 (strategy 2) + 100 (strategy 4), not 125.
        ("4 DMS guide",
         esm2_slice(target, lambda b, e: b > 0), False, ""),
        ("5 MSA guide",
         msc._msa_pool(target), False, "4x budget"),
        ("6 MSA gibbs",
         [CAMP / "msa-gibbs" / "designs_benchmark375" / "design_EGFP.csv"], True, ""),
    ]


def score(paths, target, target_free, tex, tem):
    frames = []
    for p in paths:
        frames.append(pd.read_csv(p).assign(_cell=Path(p).parent.name))
    d = pd.concat(frames, ignore_index=True)
    cells = d["_cell"].nunique()
    tpc = sorted(d.groupby("_cell").trial.nunique().unique())
    d = d[d["round"] >= 1].copy()
    raw = len(d)
    d = d.drop_duplicates("designed_seq").copy()
    if target_free:
        # no target columns in a target-free CSV: recompute peak_err per target, as build() does
        d["peak_err"] = 0.5 * ((d["pred_ex"] - tex).abs() + (d["pred_em"] - tem).abs())
    d["_blog"] = (d["pred_bright"].to_numpy() if "pred_bright" in d.columns
                  else msc.bright_logit(d["designed_seq"].tolist()))
    d["_idist"] = msc.id_dist(d["designed_seq"].tolist())
    d["_is_id"] = d["_idist"] <= msc.p99
    d["_ok"] = d["_is_id"] & (d["_blog"] > msc.BRIGHT_T)
    scaf = d["scaffold_seq"].iloc[0]
    d["_nmut"] = [sum(a != b for a, b in zip(s, scaf)) for s in d["designed_seq"]]
    return d, cells, "/".join(map(str, tpc)), raw


def main():
    targets = [sys.argv[1]] if len(sys.argv) > 1 else ["mOrange", "EBFP"]
    for target in targets:
        ref = pd.read_csv(msc.REF_CSV[target]).iloc[0]
        tex, tem = float(ref["target_ex"]), float(ref["target_em"])
        print("\n" + "=" * 118)
        print(f"### EQUAL-BUDGET BENCHMARK -- {target} (true peaks {tex:.0f}/{tem:.0f} nm)")
        print("=" * 118)
        rows = []
        for label, paths, tfree, note in pools(target):
            missing = [p for p in paths if not Path(p).exists()]
            if missing or not paths:
                print(f"  [skip] {label}: {len(missing)} missing path(s)")
                continue
            d, cells, tpc, raw = score(paths, target, tfree, tex, tem)
            ok = d[d["_ok"]]
            r = dict(strategy=label, cells=cells, trials_cell=tpc, raw=raw, unique=len(d),
                     best_err=round(d["peak_err"].min(), 1),
                     mean_err=round(d["peak_err"].mean(), 1),
                     id_pct=round(100 * d["_is_id"].mean(), 1),
                     bright_pct=round(100 * (d["_blog"] > msc.BRIGHT_T).mean(), 1),
                     n_ok=len(ok),
                     best_ok=round(ok["peak_err"].min(), 1) if len(ok) else np.nan,
                     nmut_ok=round(ok["_nmut"].mean(), 1) if len(ok) else np.nan,
                     note=note)
            if len(ok):
                top = msc.diverse_topk(ok, msc.N, msc.MIN_HD)
                r["top10_ex"] = round(top["pred_ex"].mean(), 1)
                r["top10_em"] = round(top["pred_em"].mean(), 1)
                r["top10_nmut"] = round(top["_nmut"].mean(), 1)
            rows.append(r)
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print("\n  raw = designs at round>=1 before dedupe (cells x trials x 3 iterations).")
        print("  n_ok / best_ok = in-distribution AND classifier logit > 0.5, the shortlist's own bar.")
        print("  '4x budget' = strategy 5 left at its full 12 trials/cell by design; not equal-budget.")
        lo = df[df["raw"] < 1125]
        print(f"  BUDGET CHECK (raw >= 1125): "
              f"{'PASS - all strategies' if lo.empty else 'FAIL: ' + ', '.join(lo.strategy)}")


if __name__ == "__main__":
    main()
