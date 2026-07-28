# msa-gibbs/ — the family profile with the steering switched off (EGFP, target-free)

A sixth strategy, and the one that exists to be a **control**. It is to
[`../msa-guided/`](../msa-guided/) what [`../gibbs-sampling/`](../gibbs-sampling/) is to the
ESM-based guided strategies: the same proposal distribution, with the surrogate removed from
selection. Together the four close a 2×2 that separates the two things that actually differ between
strategies in this campaign — *where candidates come from* and *whether anything steers the choice*.

|  | unguided (gibbs) | surrogate-guided |
|---|---|---|
| **ESM-2 proposal** | [`../gibbs-sampling/`](../gibbs-sampling/) | [`../lambda_sweep/`](../lambda_sweep/), [`../brightness-guided/`](../brightness-guided/) |
| **family profile** | **this folder** | [`../msa-guided/`](../msa-guided/) |

Without this cell you cannot tell whether the MSA guide's advantage comes from the family profile or
from the λ retuning that came with it. With it, you can: read down the left column for the proposal
effect in isolation, and across the rows for the steering effect.

## What it does

Target-free, exactly like `gibbs-sampling/` — the search never sees a target, so one run is a single
design effort per scaffold and it writes the target-free `COLS_FREE` schema. At each visited
position the residue is drawn from `softmax(top-k family log-frequency / T)` at `T=1, k=10`, the
family analogue of a Gibbs draw from ESM-2's masked-LM conditional. The surrogate is loaded only to
record each round's (ex, em); it never enters selection.

It reads **the same PSSM file** as `msa-guided/` — `../msa-guided/msa_pssm_egfp.json`, referenced
rather than copied, so the two provably sample the identical distribution — and inherits the hard
support constraint (zero family frequency means unselectable) and the Tier-B window (local copy,
byte-identical to every other strategy's).

### One property to understand before reading the numbers

**A PSSM is context-independent.** ESM-2's conditional shifts as the sequence is edited, so its
Gibbs chain genuinely mixes and is pulled toward a self-consistent sequence. The family profile does
not move at all. Every iteration re-draws each editable position independently from the same fixed
distribution, so the chain has no memory: round 3 is one independent sample from the profile, not a
refinement of round 2.

That makes this a clean sample of *what the family profile alone proposes for this window*, which is
the baseline worth having — but it also means the mutation load is set by the profile rather than by
the scaffold. `--lam-edit` is exposed (default 0) if you want the opt-in penalty toward the scaffold
residue; at 0 this is an exact draw from the profile's top-k.

There is **no brightness term**: the engine only accepts `brightness_ckpt` with `strategy="guided"`,
and steering is exactly what this control removes. `pred_bright` is therefore absent from the CSV
and is scored afterwards by the same `cnn-max-d2` classifier every other strategy is judged by,
as is already done for ESM gibbs.

## Results — 96 trials × 3 iterations, 11 seconds

The whole run costs 11 s on one GPU, against roughly an hour for ESM gibbs at the same depth: Gibbs
never calls the surrogate on candidate residues, and a profile lookup replaces the ESM-2 forward
pass that dominated the original. Both rows below are 288 unique round≥1 designs, scored by the same
brightness classifier and the same in-distribution test (NN distance to the 40k GFP-DMS cloud ≤ its
99th percentile, `p99 ≈ 30.2`).

| | ESM-2 gibbs | **MSA gibbs** |
|---|---|---|
| mutations from EGFP | 22.7 (18–25) | **18.0 (11–23)** |
| in-distribution | 0 / 288 (0 %) | **24 / 288 (8 %)** |
| predicted bright | 0 / 288 (0 %) | 0 / 288 (0 %) |
| best peak error → mOrange | 44.7 nm | **29.2 nm** |
| best peak error → EBFP | 38.6 nm | **27.1 nm** |
| mean predicted (ex, em) | 463 / 496 | **479 / 503** |

EGFP itself is at 488 / 507, so the family profile's designs drift roughly half as far from the
scaffold's spectrum as ESM-2's do. The support constraint holds exactly as it does in the guided
case: **0 of 5,177 position-edits fall outside the family support**, against 22.6 % for the
ESM-based strategies on the same window.

**The proposal matters, and it is not enough.** Swapping ESM-2 for the family profile improves every
axis — fewer mutations, the first non-zero in-distribution fraction any unguided strategy has
produced, and a best-case peak error ~10 nm better on both targets. But **both unguided strategies
produce exactly zero predicted-bright designs out of 288.** No proposal distribution substitutes for
the brightness term; that result is unchanged by fixing the thing that was most obviously broken
about the proposal.

That is the cleanest available statement of what each half of the MSA guide contributes. The family
profile is why its designs are chemically plausible and in-distribution; the brightness term is why
any of them are usable at all.

Averaged over all designs the unguided run also moves *away* from mOrange (63.7 nm mean vs the
scaffold's 56.7 nm), which is the expected behaviour of sampling with no target in the objective —
the best-case numbers above are the tail of 288 draws, not a trend.

**One place the mean and the best disagree, and why.** In the notebook's strategy-comparison bar
chart (mean over each trial's best round) MSA gibbs looks *worse* than ESM gibbs against EBFP —
70 nm vs 57 nm — while its single best design is better (27.1 vs 38.6 nm). That is not a
contradiction: ESM gibbs blue-shifts further by accident (mean 463/496 against MSA gibbs' 479/503),
so drifting harder away from green happens to land it nearer a 380/440 target on average. Against
mOrange, where the shift is in the other direction, the ordering flips and MSA gibbs wins on the
mean too (54 vs 64 nm). Neither run is optimizing either target, so the mean is measuring which
direction the proposal happens to drift; the best-of-288 is the number that reflects what the pool
contains.

## Layout

```
design_campaign.py                 driver: MSAGibbsCampaign (target-free, no ppl, no brightness term)
design_windows_egfp_tierB.json     the Tier-B window (copy of the campaign's, md5 9ad48573…)
run_campaign.sh                    wrapper: timestamped log + .last_log, resumable
designs/design_EGFP.csv            96 trials × 3 rounds, target-free COLS_FREE schema
logs/design_campaign_<ts>.log      one log per pass
```

The PSSM is **not** duplicated here — it lives at `../msa-guided/msa_pssm_egfp.json` and is rebuilt
with `../msa-guided/build_msa_pssm.py`.

Reproduce, from this folder:

```bash
bash run_campaign.sh --trials 96 --iters 3     # ~11 s; complete trials are skipped on re-run
```

## Caveats

- **Zero brightness is a classifier statement, not a measurement.** `cnn-max-d2` (val AUROC 0.982)
  is confident that 18 random-ish substitutions in the chromophore pocket kill fluorescence. That is
  almost certainly right, but it is the same in-sample model used everywhere in this campaign.
- **The chain does not mix.** Because the profile is context-independent, `--iters` buys independent
  redraws, not refinement. Comparing convergence-vs-round against ESM gibbs is not meaningful; only
  the final distribution is.
- **Target-free means the peak errors are post-hoc.** They are computed by pairing each design with
  each target afterwards, exactly as the notebook does for ESM gibbs. Nothing in the run optimized
  them.
