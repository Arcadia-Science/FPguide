#!/usr/bin/env python
"""Score the matched-lambda sweep against strategies 4 and 5 on identical criteria.

Every filter here is IMPORTED from ../make_shortlist_case.py rather than reimplemented -- the ID
test (ESM max-pool NN-distance to the 40k GFP-DMS reference <= p99), the confident-brightness bar
(BRIGHT_T = 0.5), the greedy diverse top-10 (N = 10, MIN_HD = 5) and the on-disk embedding cache.
Reimplementing any of them would make the comparison this effort exists to run meaningless.

Three views, all on the same designs:

  1. PER STRATEGY SLICE. lam_ex/lam_em never take 0 in this grid while lam_bright and lam_edit both
     do, so the cells partition onto campaign strategies 2 (peaks only) and 4 (peaks + brightness
     + edit). The 20 cells that used to be read as strategy 3 (peaks + edit, lam_bright=0 and
     lam_edit>0) are EXCLUDED -- that strategy was retired from the campaign. They are still on
     disk; nothing reads them. See ../esm2_guided/README.md.
  2. LAMBDA HEATMAP. Best ID-and-bright peak error per (lam_bright, lam_edit), minimised over
     lam_peaks. The question is whether the ESM-2 proposal has a good region at lam_edit ~ 0 that
     strategy 4's T=10 grid (lam_edit in {10,15,20}) structurally could not reach. Excluding the
     strategy-3 cells blanks four entries in the lam_bright=0 row only; all 100 lam_bright>0 cells
     keep full lam_edit coverage, so that question is unaffected.
  3. HEAD TO HEAD. The same numbers for strategy 5's pool (../msa-guided/), pooled and filtered
     exactly as the shortlists do, alongside this effort's own row. Strategy 4's original T=10
     pool was retired to ../archive/superseded-unmatched-runs/ and is no longer read; its
     comparator is the "4 DMS guide" slice in view 1.

lam_edit is NOT a CSV column in any sweep -- the engine records lam_ex/lam_em/lam_bright only --
so it is parsed from the cell folder name, which is the same thing the shortlists' `source`
carries. The other three are cross-checked against the folder name; a mismatch is fatal.

    python analyze.py                 # both targets
    python analyze.py mOrange         # one
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CAMP = HERE.parent
sys.path.insert(0, str(CAMP))
import make_shortlist_case as msc  # noqa: E402  (loads ESM, the DMS reference, the classifier)

CELL_RE = re.compile(r"lam-ex([\d.]+)_lam-em([\d.]+)_lam-bright([\d.]+)_lam-edit([\d.]+)$")


def parse_cell(name):
    m = CELL_RE.match(name)
    if not m:
        raise SystemExit(f"cell folder does not carry four lambdas: {name}")
    return tuple(float(x) for x in m.groups())


def is_retired(lam_bright, lam_edit):
    """Cells belonging to the RETIRED constrained spectra guide (strategy 3).

    lam_bright=0 with lam_edit>0 is peaks + edit penalty and no brightness term -- exactly what
    strategy 3 was. That strategy was dropped from the campaign, so its 20 cells are excluded here
    before anything is pooled: they feed no slice table, no pooled row and no heatmap cell. The
    CSVs stay on disk as run output; nothing reads them.
    """
    return lam_bright == 0 and lam_edit > 0


def slice_of(lam_bright, lam_edit):
    """Which campaign strategy a cell reproduces (see README, 'One sweep covers 2 and 4')."""
    return "2 spectra guide" if lam_bright == 0 else "4 DMS guide"


def load_pool(paths, target, tag_cell=True):
    """Pool design CSVs, dedupe on sequence, attach lambdas. Mirrors make_shortlist_case.build."""
    frames = []
    for p in paths:
        d = pd.read_csv(p).assign(source=Path(p).parent.name)
        if tag_cell:
            lp, _, lb, le = parse_cell(Path(p).parent.name)
            # the CSV records three of the four; a folder/CSV mismatch means the cell was written
            # by a different setting than its name claims, which would silently corrupt every
            # slice below (existing_pair validates temp/trials/k but never the lambdas).
            for col, want in (("lam_ex", lp), ("lam_em", lp), ("lam_bright", lb)):
                got = d[col].dropna().unique()
                if len(got) != 1 or not np.isclose(got[0], want):
                    raise SystemExit(f"{p}: {col}={got} but folder says {want}")
            d["lam_peaks"], d["lam_bright_v"], d["lam_edit_v"] = lp, lb, le
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d[d["round"] >= 1].drop_duplicates("designed_seq").copy()
    return d


def score(d):
    """Attach ID distance / brightness logit / mutation load, using make_shortlist_case's machinery."""
    d = d.copy()
    d["_blog"] = (d["pred_bright"].to_numpy() if "pred_bright" in d.columns
                  else msc.bright_logit(d["designed_seq"].tolist()))
    d["_idist"] = msc.id_dist(d["designed_seq"].tolist())
    d["_is_id"] = d["_idist"] <= msc.p99
    d["_ok"] = d["_is_id"] & (d["_blog"] > msc.BRIGHT_T)      # the shortlist's own admission bar
    scaf = d["scaffold_seq"].iloc[0]
    d["_nmut"] = [sum(a != b for a, b in zip(s, scaf)) for s in d["designed_seq"]]
    return d


def summarise(d, label):
    ok = d[d["_ok"]]
    row = dict(group=label, designs=len(d),
               best_err=round(d["peak_err"].min(), 1),
               mean_err=round(d["peak_err"].mean(), 1),
               id_pct=round(100 * d["_is_id"].mean(), 1),
               bright_pct=round(100 * (d["_blog"] > msc.BRIGHT_T).mean(), 1),
               n_ok=len(ok),
               best_err_ok=round(ok["peak_err"].min(), 1) if len(ok) else np.nan,
               mean_nmut_ok=round(ok["_nmut"].mean(), 1) if len(ok) else np.nan)
    if len(ok):
        top = msc.diverse_topk(ok, msc.N, msc.MIN_HD)
        row["top10_mean_ex"] = round(top["pred_ex"].mean(), 1)
        row["top10_mean_em"] = round(top["pred_em"].mean(), 1)
        row["top10_mean_nmut"] = round(top["_nmut"].mean(), 1)
    return row


def heatmap(d, target):
    print(f"\n  best ID-and-bright peak error (nm) by lam_bright x lam_edit, "
          f"min over lam_peaks -- {target}")
    piv = (d[d["_ok"]].groupby(["lam_bright_v", "lam_edit_v"])["peak_err"].min().unstack())
    piv = piv.reindex(index=sorted(d["lam_bright_v"].unique()),
                      columns=sorted(d["lam_edit_v"].unique()))
    hdr = "  lam_bright \\ lam_edit |" + "".join(f"{c:>8g}" for c in piv.columns)
    print("  " + "-" * (len(hdr) - 2))
    print(hdr)
    for b, r in piv.iterrows():
        cells = "".join("      --" if pd.isna(v) else f"{v:8.1f}" for v in r)
        print(f"  {b:>21g} |{cells}")
    print("  '--' = no design in that cell passed ID and logit > 0.5.")
    print("  Top row (lam_bright=0) is strategy 2, and only its lam_edit=0 column is CONSIDERED --")
    print("  the lam_edit>0 cells there were the retired strategy 3 and are excluded entirely.")
    print("  The rest is strategy 4, which keeps full lam_edit coverage.")
    print("  lam_edit 0 and 0.5 are the columns strategy 4's own T=10 grid could not express.")


def main():
    targets = [sys.argv[1]] if len(sys.argv) > 1 else ["mOrange", "EBFP"]
    for target in targets:
        all_cells = sorted(HERE.glob(f"designs/*/design_EGFP-{target}.csv"))
        if not all_cells:
            print(f"\n### {target}: no cells yet"); continue
        # Drop the retired strategy-3 cells BEFORE pooling, so they cannot reach the slice table,
        # the pooled row or the heatmap. Reported rather than silently dropped -- a shrinking cell
        # count with no explanation is exactly how a truncated analysis reads as a complete one.
        cells = [p for p in all_cells if not is_retired(*parse_cell(p.parent.name)[2:])]
        n_ret = len(all_cells) - len(cells)
        print("\n" + "=" * 100)
        print(f"### {target} -- matched-lambda ESM-2 sweep: {len(cells)} cells "
              f"({n_ret} retired strategy-3 cells excluded, of {len(all_cells)} on disk)")
        print("=" * 100)
        d = score(load_pool(cells, target))
        d["slice"] = [slice_of(b, e) for b, e in zip(d["lam_bright_v"], d["lam_edit_v"])]

        rows = [summarise(d[d["slice"] == s], s) for s in sorted(d["slice"].unique())]
        rows.append(summarise(d, f"ALL {len(cells)} cells pooled"))
        print("\n  per strategy slice (deduped designs; 'ok' = ID and logit > 0.5)")
        print(pd.DataFrame(rows).to_string(index=False))

        heatmap(d, target)

        # ---- the strategy this is a control for, pooled and filtered identically ----
        # Strategy 4's ORIGINAL T=10 pool used to be a row here. That run was retired to
        # archive/superseded-unmatched-runs/ (gitignored) and is no longer read by active code;
        # its comparator is now this file's own "4 DMS guide" slice above, which is the same
        # measurement made on the matched grid.
        print(f"\n  head to head -- same filters, same top-10 rule")
        ref = []
        s5 = score(load_pool(msc._msa_pool(target), target, tag_cell=False))
        ref.append(summarise(s5, "strategy 5 MSA guide (125 cells, 12 trials)"))
        ref.append(summarise(d, "THIS: ESM-2 at strategy 5's lambdas (3 trials)"))
        print(pd.DataFrame(ref).to_string(index=False))

        out = HERE / f"metrics_{target}.csv"
        (d.groupby(["lam_peaks", "lam_bright_v", "lam_edit_v"])
           .apply(lambda g: pd.Series(summarise(g, "")), include_groups=False)
           .drop(columns=["group"]).reset_index().to_csv(out, index=False))
        print(f"\n  per-cell metrics -> {out.name}")


if __name__ == "__main__":
    main()
