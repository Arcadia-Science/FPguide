#!/usr/bin/env python
"""Assemble the wet-lab order sheet for batch 1 from the per-strategy shortlists.

    python make_batch.py            # -> shortlists/FPdesign-batch1.xlsx

Batch 1 is 10 constructs, all from the two MSA-proposal strategies:

    5   MSA guide  (mOrange)   2 low / 2 medium / 1 high mutation load
    3   MSA guide  (EBFP)      1 low / 1 medium / 1 high
    2   MSA gibbs              the 2 closest to mOrange -- unguided control

The guided eight are stratified by MUTATION LOAD rather than taken as ranks 1-N, so the batch spans
the edit-distance range of each top-10 instead of clustering at whatever load the surrogate happens
to prefer. That turns "how many edits can the scaffold absorb" into something the batch can answer.
Within a tier the pick is the lowest surrogate peak error, so quality still decides among equals.

The guided eight also clear a stricter brightness bar than the classifier's own boundary: the
shortlists they come from require logit > 0.5 rather than > 0 (see BRIGHT_T in
make_shortlist_case.py), so no wet-lab slot goes to a design sitting on the decision boundary.

Every column of the source shortlist is carried through unchanged; this script only SELECTS rows and
adds `batch_id`, `mut_tier`, `shortlist_file` and `pred_peak_err_nm` (the last recomputed here
against the target's TRUE peaks, so the order sheet is self-contained). Re-running the shortlist
builders and then this script reproduces the file exactly.
"""
import sys
from pathlib import Path
import pandas as pd

CAMP = Path("/home/ubuntu/spectrum-to-fp-design/design-campaign-EGFP")
sys.path.insert(0, str(CAMP))
from xlsx_io import write_xlsx

SHORT = CAMP / "shortlists"
OUT = SHORT / "FPdesign-batch1.xlsx"

# (shortlist file, [(design name, mutation-load tier, expected edits)])
#
# CAREFUL: shortlist design names are RANK-DERIVED (`<target>_<code>_<NN>` is assigned in peak-error
# order), so a name does NOT pin a sequence -- rebuilding a shortlist under different selection
# criteria silently repoints the same name at a different design. The expected edit count is
# therefore recorded alongside each pick and asserted below, which turns a shifted shortlist into a
# loud failure instead of a quietly wrong order sheet.
#
# Tier boundaries are written out per target because the two top-10s are distributed differently:
#   mOrange  6,6 | 9,9,9 | 10,10,11,12,14  -- `high` takes the 14, the top of the range; a 10 sits
#            one edit above the medium cluster and would not test anything new.
#   EBFP     7,7,8,8 | 10,11,11,11 | 12,12  -- the low cluster is tight (7-8) and the upper half
#            crowds into 11-12, so the ladder can only span 8 -> 11 -> 12.
PICKS = [
    ("shortlist_mOrange_MSA-guide.xlsx", [("mOrange_MSA_03", "low",     6),
                                          ("mOrange_MSA_07", "low",     6),
                                          ("mOrange_MSA_01", "medium",  9),
                                          ("mOrange_MSA_04", "medium",  9),
                                          ("mOrange_MSA_10", "high",   14)]),
    ("shortlist_EBFP_MSA-guide.xlsx",    [("EBFP_MSA_01", "low",     8),
                                          ("EBFP_MSA_02", "medium", 11),
                                          ("EBFP_MSA_06", "high",   12)]),
    # MSA gibbs is target-free: one 288-design run backs both target files and only the ranking
    # differs. Both controls are taken from the mOrange ranking, so they are the two unguided
    # designs that drift furthest toward orange. This strategy produces no predicted-bright designs
    # at all, so the brightness bar that gates the guided eight does not apply to it.
    ("shortlist_mOrange_MSA-gibbs.xlsx", [("mOrange_MSAgib_01", "", 21),
                                          ("mOrange_MSAgib_02", "", 20)]),
]

TRUE_PEAKS = {}   # filled from the reference rows
refs, designs = {}, []

for fname, picks in PICKS:
    names = [n for n, _, _ in picks]
    d = pd.read_excel(SHORT / fname)
    for _, r in d[d.role == "reference"].iterrows():
        refs.setdefault(r["name"], r)
        TRUE_PEAKS[r["name"]] = (float(r.true_ex_nm), float(r.true_em_nm))
    missing = set(names) - set(d.name)
    if missing:
        raise SystemExit(f"{fname}: missing {sorted(missing)} -- rebuild the shortlist first")
    sel = d[d.name.isin(names)].set_index("name").loc[names].reset_index()
    drift = [(n, exp, int(got)) for (n, _, exp), got in zip(picks, sel.n_mut_vs_EGFP) if int(got) != exp]
    if drift:
        raise SystemExit(f"{fname}: these names no longer point at the design they were picked for "
                         f"(name, expected edits, found): {drift}. Shortlist names are rank-derived "
                         f"-- re-choose the batch against the rebuilt shortlist.")
    designs.append(sel.assign(shortlist_file=fname, mut_tier=[t for _, t, _ in picks]))

rows = [{**r.to_dict(), "batch_id": "", "mut_tier": "", "shortlist_file": "",
         "pred_peak_err_nm": ""} for r in refs.values()]

for i, r in enumerate(pd.concat(designs, ignore_index=True).to_dict("records"), 1):
    tex, tem = TRUE_PEAKS[r["target"]]
    rows.append({**r, "batch_id": f"B1_{i:02d}",
                 "pred_peak_err_nm": round(0.5 * (abs(r["pred_ex_nm"] - tex)
                                                  + abs(r["pred_em_nm"] - tem)), 1)})

COLS = ["batch_id", "name", "role", "target", "strategy", "mut_tier", "true_ex_nm", "true_em_nm",
        "pred_ex_nm", "pred_em_nm", "pred_peak_err_nm", "n_mut_vs_EGFP", "is_id", "is_bright",
        "bright_logit", "source", "shortlist_file", "aa_sequence", "dna_sequence"]
out = pd.DataFrame(rows)[COLS]
write_xlsx(out, OUT, sheet_name="batch1")

d = out[out.role == "design"]
print(f"{len(d)} designs + {len(out) - len(d)} references -> {OUT.name}")
for (s, t), g in d.groupby(["strategy", "target"], sort=False):
    tiers = "/".join(x or "-" for x in g.mut_tier)
    print(f"  {s:20} {t:8} n={len(g)} | muts {g.n_mut_vs_EGFP.tolist()} ({tiers}) | "
          f"err {g.pred_peak_err_nm.min():.1f}-{g.pred_peak_err_nm.max():.1f} nm | "
          f"ID {(g.is_id == 'yes').sum()}/{len(g)} | bright {(g.is_bright == 'yes').sum()}/{len(g)}")
