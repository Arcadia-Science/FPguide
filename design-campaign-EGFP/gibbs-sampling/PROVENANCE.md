# ⚠ `designs/design_EGFP.csv` IS THE SHORTLIST SOURCE — DO NOT MODIFY OR EXTEND IT

This folder is **not** deprecated. Strategy 1 is a pure ESM-2 masked-LM Gibbs draw at `T = 1` with
no target term and no λ at all, so nothing about its scale was ever mismatched with the other
strategies.

What this note protects is the **96-trial run in `designs/design_EGFP.csv`** — 96 trials × 3
iterations = 288 target-free designs, with a populated `ppl` column.

## Why it is load-bearing

It backs `../shortlists/shortlist_{mOrange,EBFP}_gibbs.xlsx`. Two downstream consumers read it:

- `../make_shortlist_case.py` as `GIBBS`, for the `mOrange_gibbs` / `EBFP_gibbs` cases.
- `../visualize_campaign.ipynb`, for the campaign figures.

`../scale_to_96.sh` also targets this folder.

## Why appending trials here would be silently destructive

The engine is resumable (`trial_resume=True`), so re-running with `--trials 375` would **append**
trials 96–374 to this same CSV rather than complain — enlarging the shortlist pool and changing the
top-10. Shortlist design names are rank-derived (`<target>_<code>_<NN>`, assigned in peak-error
order), so the same name would silently point at a different sequence.

## Where the scaled-up benchmark run lives instead

`designs_benchmark375/design_EGFP.csv` — 375 trials × 3 iterations = 1,125 raw designs, matching
the equal-budget benchmark every other strategy is being scaled to. It is written by
`design_campaign_benchmark.py`, a thin wrapper that changes **only** the output directory and
disables ESM-2 pseudo-perplexity.

**The benchmark run leaves `ppl` blank.** `ppl_batched` costs about as much as a whole design
iteration and would roughly double a multi-hour run for a column nothing in this campaign reads;
`../msa-gibbs/`, `../msa-guided/` and `../esm2_guided/` all skip it
for the same reason. That is the one respect in which the benchmark CSV differs from this one
beyond depth. Run `--backfill-ppl` against the benchmark tree if it is ever wanted.

Apart from `ppl`, the two trees are directly comparable — per-trial seeding is
`SEED + si*131 + trial*17` (`fpdesign/campaign.py` line 419), so trials 0–95 of the benchmark run
reproduce this folder's designs exactly.
