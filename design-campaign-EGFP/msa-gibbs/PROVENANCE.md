# ⚠ `designs/design_EGFP.csv` IS THE SHORTLIST SOURCE — DO NOT MODIFY OR EXTEND IT

This folder is **not** deprecated. Unlike the T=10 guided arm (`guided-design/`,
`guided-design-constraint/`, `brightness-guided/`, `lambda_sweep/`) — since superseded by
`../esm2_guided/` and retired to `../archive/superseded-unmatched-runs/` — strategy 6 runs
at `T = 1` and needs no λ at all (it is an unguided draw from the family profile), so nothing about
its scale was ever mismatched.

What this note protects is the **96-trial run in `designs/design_EGFP.csv`** — 96 trials × 3
iterations = 288 target-free designs.

## Why it is load-bearing

**Batch 1's two wet-lab controls came from this exact file.** `B1_09` (`mOrange_MSAgib_01`) and
`B1_10` (`mOrange_MSAgib_02`) in `../shortlists/FPdesign-batch1.xlsx` are the two designs from this
run that drift furthest toward orange. It also backs
`../shortlists/shortlist_{mOrange,EBFP}_MSA-gibbs.xlsx`.

Two downstream consumers read it:

- `../make_shortlist_case.py` as `MGIB`, for the `mOrange_MSAgibbs` / `EBFP_MSAgibbs` cases.
- `../visualize_campaign.ipynb`, for the campaign figures.

## Why appending trials here would be silently destructive

The engine is resumable (`trial_resume=True`), so re-running with `--trials 375` would **append**
trials 96–374 to this same CSV rather than complain. That would enlarge the shortlist pool, change
the top-10, and — because shortlist design names are **rank-derived** (`<target>_<code>_<NN>` is
assigned in peak-error order) — silently repoint `mOrange_MSAgib_01` at a different sequence.

`../make_batch.py` asserts the expected edit count next to each pick, so it would most likely fail
loudly rather than mis-order. But the shortlist xlsx itself would already have changed, and the
batch-1 provenance chain would be broken.

## Where the scaled-up benchmark run lives instead

`designs_benchmark375/design_EGFP.csv` — 375 trials × 3 iterations = 1,125 raw designs, matching
the equal-budget benchmark every other strategy is being scaled to. It is written by
`design_campaign_benchmark.py`, a thin wrapper that changes **only** the output directory. Nothing
downstream reads it; it exists purely for the cross-strategy comparison.

The two trees are directly comparable — per-trial seeding is `SEED + si*131 + trial*17`
(`fpdesign/campaign.py` line 419), so trials 0–95 of the benchmark run are byte-identical to this
folder's 96-trial run.
