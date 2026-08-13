# `esm2_guided/` — the ESM-2 guided arm (strategies 2 + 4), at strategy 5's λ grid

> Formerly `lambda_sweep_matched/`. The name records what it *is* — the surrogate-guided arm with
> ESM-2 as the proposal — rather than how it came about, which was as a matched-λ control against
> the T=10 sweep it has since replaced.

> **Path note.** This effort exists *because* of the T=10 sweep it is a control for, so
> `../lambda_sweep/`, `../guided-design/` and `../guided-design-constraint/` are referred to
> throughout. That whole arm has since been retired — **this effort is what superseded it** — and
> now lives in `../archive/superseded-unmatched-runs/`, which is gitignored and read by no active
> code. Every such reference below is a historical one; nothing here loads those folders.

The campaign's headline mOrange result is **MSA guide 3.4 nm vs DMS guide 23.7 nm**, both 10/10
in-distribution and confidently bright. This effort exists because those two numbers were produced
by λ grids that do not overlap where it matters, so the comparison could not distinguish "the
family profile is a better proposal" from "the ESM-based sweep looked in the wrong place".

|  | strategy 4 (`../lambda_sweep/`) | strategy 5 (`../msa-guided/`) |
|---|---|---|
| proposal | ESM-2 masked-LM | 763-seq family profile |
| λ_ex = λ_em | 10, 20, 30 | 0.25, 0.5, 1, 2, 4 |
| λ_bright | 40, 50, 60 | **0**, 0.5, 1, 2, 4 |
| λ_edit | 10, 15, 20 | **0**, 0.5, 1, 2, 4 |
| temperature | 10 | 1 |
| cells | 27 (+1 deep tuned run) | 125 |
| trials/cell | 12, but 24/48/96 in 3 imported cells | 12, uniform |
| unique designs (mOrange) | 1,110 | 2,635 |

Every one of strategy 5's mOrange top-10 came from **λ_edit ∈ {0, 0.5}** — four from λ_edit = 0
exactly — and λ_peaks ∈ {2, 4}; see the `source` column of
`../shortlists/shortlist_mOrange_MSA-guide.xlsx`. The T=10 grid cannot express either value: its
weakest edit penalty is 10, where the penalty outweighs the z-scored proposal 10:1. So strategy
5's entire winning shortlist sits in a region strategy 4's sweep structurally could not sample.

The campaign README defends the comparison on **pool depth** — quadrupling strategy 4's pool
(287 → 1,110) moved its best error 24.2 → 23.7 nm. That tests sampling density *inside* its grid.
It says nothing about the grid being centred in the wrong place, which is a different confound and
the one this effort tests.

## What this runs

Strategy 5's exact grid — `{0.25,0.5,1,2,4} × {0,0.5,1,2,4} × {0,0.5,1,2,4}` at T = 1 — on the
real ESM-2 masked-LM proposal. `design_campaign.py` is a near-copy of
`../lambda_sweep/design_campaign.py` differing in exactly three things (own output tree, T = 1,
unit λ defaults); everything else — engine, surrogate, brightness classifier, Tier-B window, pairs
CSV, per-cell folder naming, ppl-disabling subclass — is inherited unchanged. The separate output
tree is mandatory, not cosmetic: cell folders are named from the four λ but **not** the
temperature, and `existing_pair` validates temp but not λ, so T=1 cells written into
`../lambda_sweep/designs/` would be indistinguishable to the `designs/*/design_EGFP-*.csv` glob
that `visualize_campaign.ipynb` and `make_shortlist_case.py` both use.

**3 trials × 3 iterations per cell, both targets.** 125 cells × 3 trials × 3 rounds = 1,125
designs per target before dedupe.

## One sweep covers strategies 2 and 4

λ_ex/λ_em never take 0 in this grid while λ_bright and λ_edit both do, so the 125 cells partition
exactly onto the campaign's strategies — no overlap, no gaps:

| strategy | slice | cells | |
|---|---|---|---|
| **2** spectra guide (peaks only) | λ_bright = 0, λ_edit = 0 | 5 | analysed |
| ~~**3** constrained spectra guide (peaks + edit)~~ | λ_bright = 0, λ_edit ∈ {0.5,1,2,4} | 20 | **retired — excluded** |
| **4** DMS guide (peaks + brightness + edit) | λ_bright ∈ {0.5,1,2,4} × λ_edit ∈ {0,…,4} | 100 | analysed |

**Strategy 3 was retired from the campaign.** Its 20 cells were run and remain on disk, but they
feed nothing: `analyze.py` drops them before pooling and `../benchmark_report.py` has no row for
them, so both report **105 cells, not 125**. The numbering keeps its gap on purpose, so "strategy
4" still means what it means everywhere else in this campaign.

That exclusion costs the λ heatmap four entries in its λ_bright = 0 row and nothing else — all 100
λ_bright > 0 cells keep full λ_edit coverage, **including the λ_edit ∈ {0, 0.5} region this whole
effort exists to probe**, so the grid-coverage argument above is untouched.

In `fpdesign.campaign._select_guided` the brightness term is
`scores + lam_bright * _zc(Bk[sl, 0])`, so λ_bright = 0 contributes exactly nothing. Running the
strategies as separate 125-cell sweeps would be 375 cells, 250 of them recomputing identical
arithmetic.

The reason to keep the classifier **loaded** at λ_bright = 0, rather than running a separate
brightness-free driver, is that such a driver sets `brightness_ckpt=None` and therefore cannot
*record* `pred_bright`. Logging it across all cells is what lets the same ID-and-bright filter
every other strategy is judged by apply here too.

Note that the λ_bright > 0, λ_edit = 0 block (20 cells) is a setting strategy 4 never ran — its
grid fixed λ_edit ∈ {10,15,20}. That block is exactly where strategy 5's winners live, so it is
the most load-bearing part of the grid.

### λ_bright = 0 equivalence — verified, not assumed

`check_lam_bright0.py` runs the same driver at λ_bright = 0 twice — classifier loaded vs
`brightness_ckpt=None` — and compares `designed_seq` row for row:

```
rows compared        : 12
designed_seq IDENTICAL: True
VERDICT: PASS
```

It used to also diff this effort's `CampaignConfig` (with `brightness_ckpt=None`) against
`../guided-design-constraint/`'s field by field; that ran clean (no unexpected diffs) before the
constrained spectra guide was retired and its driver moved to
`../archive/superseded-unmatched-runs/`. That half is no longer re-runnable from active code and
has been removed; the empirical test above was always the load-bearing one.

Rerun it after any change to `_select_guided`.

## λ scale and temperature

Every term in the guided score is z-scored across the k candidates, so λ **is** the term's
relative weight — the argument `../msa-guided/design_campaign.py` makes in its point 3. That is
why the λ scale and T = 1 travel together: at the inherited λ = 20/20/60/10 with T = 10 the
effective weights are 0.1 / 2 / 2 / 6 / 1, and the proposal counts for a sixtieth of brightness.

`check_scale.py` measures where that actually puts the sampler, as normalized entropy of the
selection softmax (1.0 = uniform over the k allowed candidates, 0.0 = deterministic). Strategy 5's
λ scale had only ever been validated on the *family-profile* proposal, and the transfer is not
obvious: ESM-2 scores 12.6 % masked top-1 on EGFP, so `_zc(topv)` divides a nearly-flat log-prob
spread by a tiny standard deviation.

| setting | λ (peaks/bright/edit) | T | H/Hmax | max p |
|---|---|---|---|---|
| this grid, centre | 1/1/1 | 1 | 0.310 | 0.761 |
| this grid, corner | 4/4/4 | 1 | 0.040 | 0.967 |
| this grid, weakest cell (strategy 2) | 0.25/0/0 | 1 | 0.772 | 0.391 |
| this grid, where strategy 5's winners sat | 2/2/0 | 1 | 0.292 | 0.738 |
| **REF: strategy 4's own sweep centre** | 20/60/10 | 10 | **0.125** | 0.883 |
| REF: archived negative control | 1/1/1 | 10 | 0.984 | 0.161 |

**No cell of this grid is degenerate** at either end — the range 0.040–0.772 brackets strategy 5's
own 0.031–0.694 and stays clear of the ~0.98 failure mode `../archive/README.md` documents, where
the search collapses into random sampling from the proposal's top-k.

Two results worth carrying into the writeup:

- **Strategy 4's original grid was not degenerate either.** At 0.125 its sweep centre was *sharper*
  than this grid's centre (0.310), sitting between centre and corner. Its 23.7 nm was produced by a
  functioning, decided sampler — so that number cannot be explained away as a T=10 artifact, and
  the remaining candidate explanation is grid location.
- **ESM-2's top-k on its own sits at 0.97 — nearly uniform.** The proposal term contributes almost
  no preference of its own before the surrogate steers, which is the calibration problem
  `../msa-guided/` was built around, quantified on the same axis as everything else here.

## Result — the grid-coverage confound does not explain the gap

125 cells run, zero failures; **105 analysed** (the 20 retired strategy-3 cells are excluded).
Trials per cell were set per slice to equalise the raw-design budget — 75 for strategy 2, 4 for
strategy 4. Scored by `analyze.py`, which imports its filters from `../make_shortlist_case.py` so
they are identical by construction.

| target | pool | designs | ID % | bright % | n ID&bright | best ID&bright err |
|---|---|---|---|---|---|---|
| mOrange | strategy 4, its own T=10 grid + tuned run *(historical †)* | 1,110 | 63.7 | 15.9 | 131 | 23.7 nm |
| mOrange | strategy 5, MSA guide | 2,635 | 70.9 | 30.9 | 782 | **3.5 nm** |
| mOrange | **this: ESM-2 at strategy 5's λ** | 2,126 | 13.6 | 1.8 | 20 | **31.8 nm** |
| EBFP | strategy 4, its own T=10 grid + tuned run *(historical †)* | 948 | 63.4 | 54.9 | 394 | **7.9 nm** |
| EBFP | strategy 5, MSA guide | 2,835 | 69.1 | 20.5 | 572 | 12.5 nm |
| EBFP | **this: ESM-2 at strategy 5's λ** | 2,210 | 13.2 | 4.1 | 55 | **29.3 nm** |

† The strategy-4 T=10 rows are a **historical record**. That pool was retired to
`../archive/superseded-unmatched-runs/` and `analyze.py` no longer computes it — the live
comparator for strategy 4 is this grid's own `λ_bright > 0` slice (best ID&bright 31.8 nm mOrange,
29.3 nm EBFP). The numbers are kept here because they are what the confound test was run against.

**Giving the ESM-2 proposal strategy 5's exact λ grid makes it worse, not better.** It does not
approach the MSA guide on either target, and it does not even match strategy 4's own T=10 grid —
31.8 vs 23.7 nm on mOrange, 29.3 vs 7.9 nm on EBFP.

That comparison does not lean on the depth caveat below — it runs the other way. This pool is
roughly **twice** strategy 4's T=10 pool (2,126 designs against 1,110 on mOrange; 2,210 against
948 on EBFP), and with double the candidates the matched λ scale still yields only **20 designs
that clear ID and brightness, against 131** — and a worse best error among them.

So the confound was real but ran the *other* way: the mismatched grids, if anything, flattered the
ESM-based strategy rather than manufacturing the MSA guide's advantage.

### Why — the "badly scaled" λ was compensating for a bad proposal

The ID and brightness rates are what collapse: 63.7 % → 13.6 % ID and 15.9 % → 1.8 % bright on
mOrange. Raw peak accuracy does not — unfiltered best error is 4.2 nm (mOrange) and 0.0 nm (EBFP),
better than strategy 4's own 6.2 and 0.1. The ESM-2 proposal can still hit the peaks; it cannot do
it while staying in-distribution and bright.

This follows from the entropy numbers above. ESM-2's top-k on EGFP sits at 0.97 normalized entropy
— nearly uniform, carrying almost no information. At λ = 20/60/10 with T = 10 the effective
weights are 0.1 / 2 / 2 / 6 / 1, so the proposal is suppressed to a sixtieth of the brightness
term and the search is driven almost entirely by the surrogate and the classifier. Rescaling to
unit λ is principled *if the proposal is worth listening to* — which is exactly what
`../msa-guided/` established for the family profile and what ESM-2 fails on this fold. Strategy
4's apparently arbitrary λ scale was doing real work: drowning out its own proposal.

The heatmaps say the same thing directionally. Only λ_bright ≥ 2 (mOrange) or ≥ 1 (EBFP) yields
any design that clears both filters, and **the best cell sits at λ_bright = 4, the grid maximum**
— the optimum is on the boundary, pointing back toward the large λ_bright that strategy 4 used.

### The specific region the confound identified

Strategy 5's mOrange winners all came from λ_edit ∈ {0, 0.5}. That region was searched here and
is not where ESM-2 recovers: the entire λ_edit = 0 column is empty for mOrange, and the single
best matched-λ mOrange result (32.9 nm) sits at λ_edit = 0.5, λ_bright = 4 — the same corner, a
tenfold worse number than the family profile's 3.4 nm.

### Strategies 2 and 3 reproduce their original verdict

The λ_bright = 0 slices produced **zero** designs clearing ID and confident brightness — 0 of 42
(strategy 2) and 0 of 149 (strategy 3) on mOrange, 0 of 43 and 0 of 163 on EBFP — despite
including their best raw peak errors anywhere in the campaign (5.8, 4.7, 0.1 and 0.0 nm). The
original finding that these strategies hit the peaks and fail the filters was not an artifact of
their single λ setting; it survives a 25-cell sweep at a different λ scale.

## Equal-budget benchmark (scale-up)

The numbers above were measured at **uniform 3 trials/cell**, which gave the three slices very
different budgets (45 / 180 / 900 raw designs). `run_benchmark_scaleup.sh` then raised each slice
to a comparable **≥ 1,125 raw designs** — strategy 2 to 75 trials/cell, strategy 3 to 19, strategy
4 to 4 — matching strategies 1 and 6 at 375 target-free trials in
`../gibbs-sampling/designs_benchmark375/` and `../msa-gibbs/designs_benchmark375/`. Total 11,250 s,
zero failed cells. **`designs/` is therefore no longer uniform in depth**; anything pooling all 125
cells is depth-weighted toward the strategy-2 corner and must slice before aggregating.

Scored by `../benchmark_report.py` (same imported filters), mOrange:

| strategy | cells | trials/cell | raw | unique | best err | ID % | bright % | n ID&bright | best ID&bright |
|---|---|---|---|---|---|---|---|---|---|
| 1 gibbs (ESM-2) | 1 | 375 | 1,125 | 1,125 | 43.8 | 0.0 | 0.0 | 0 | — |
| 2 spectra guide | 5 | 75 | 1,125 | 1,121 | **0.7** | 0.6 | 0.0 | 0 | — |
| 3 constrained | 20 | 19 | 1,140 | 1,040 | 4.7 | 14.7 | 0.0 | 0 | — |
| 4 DMS guide | 100 | 4 | 1,200 | 1,008 | 4.2 | 28.1 | 3.8 | 20 | 31.8 |
| 5 MSA guide *(4× budget)* | 125 | 12 | 4,500 | 2,635 | 0.1 | **70.9** | **30.9** | **782** | **3.4** |
| 6 MSA gibbs | 1 | 375 | 1,125 | 1,125 | 26.5 | 8.9 | 0.0 | 0 | — |

EBFP behaves the same way: strategies 1, 2, 3 and 6 all return **0** designs clearing both filters,
strategy 4 reaches 29.3 nm from 55 passing designs, strategy 5 12.5 nm from 572.

**The equal-budget result reinforces the conclusion rather than softening it.** Giving strategy 2
a 25× deeper pool (45 → 1,125 raw) buys it a spectacular *unfiltered* best error — **0.7 nm on
mOrange, 0.0 nm on EBFP**, the best raw numbers anywhere in the campaign — and still **zero**
designs that are simultaneously in-distribution and confidently bright. Strategies 3 and 6 likewise
stay at 0 after their scale-ups. Depth is not what separates these strategies from the MSA guide;
the filters are, and no amount of extra sampling from a near-uniform ESM-2 proposal fixes that.

Strategy 4 is the only matched-λ slice that clears the bar at all, and it improves only slightly
with the extra trials (13 → 20 passing designs, 32.9 → 31.8 nm on mOrange), still an order of
magnitude behind the family profile.

**Caveat that remains.** Strategy 5 was deliberately left at its full 12 trials/cell — 4× every
other row — so its 3.4 nm is a best-achievable reference, not an equal-budget number. What the
scale-up does establish is the converse and more useful direction: the ESM-2-proposal strategies
were **not** budget-starved, since quadrupling to 25×-ing their pools moved none of them off zero.

## Depth caveat (original 3-trial sweep)

3 trials/cell = **9 designs per cell**, against strategy 5's 36. This is a **coarse localization
sweep**. It can answer *"does the ESM-2 proposal have a good λ region the T=10 grid missed?"* It
cannot answer *"what is the ESM-2 proposal's best achievable peak error at equal budget"*, because
best-of-N improves with N and this pool is roughly a third the size overall.

That caveat bounds how the result may be stated. What is safe: *at strategy 5's λ scale and a
comparable total pool to strategy 4's own, the ESM-2 proposal is worse on both targets* — the
942-vs-1,110 comparison above needs no depth adjustment. What is **not** established: that no
amount of extra sampling would close a 32.9-vs-3.4 nm gap. A tenfold gap is far outside what
quadrupling a pool has ever bought here (strategy 4's own quadrupling moved 24.2 → 23.7 nm), but
"implausible" is not "measured". Deepen the surviving cells to strategy 5's depth before quoting
this as a like-for-like bound:

```bash
TRIALS=12 LAM_EDIT="0 0.5" bash run_sweep.sh      # resumable: only the missing trials are run
```

## Run

```bash
cd design-campaign-EGFP/esm2_guided
python check_scale.py                    # pre-flight: is any cell degenerate?
python check_lam_bright0.py              # pre-flight: is lam_bright=0 really strategies 2/3?
setsid bash run_sweep.sh < /dev/null > /dev/null 2>&1 &
tail -f "$(cat .last_log)"
```

Sequential so the single GPU is never contended, and resumable at trial granularity — the engine
skips any cell whose CSV already holds ≥ `--trials` trials, so re-running after an interruption
only fills gaps. Pseudo-perplexity is disabled in the driver, as in both `../lambda_sweep/` and
`../msa-guided/`, so there is deliberately no `--backfill-ppl` pass.

**Budget: ~47 s per cell**, so 125 cells is about **1 h 40 min**. Derived from `../lambda_sweep/`'s
measured 81–83 s per cell for 12 trials on one target plus ~6 s of per-process model loading, i.e.
6.8 s per trial per target on an RTX PRO 4500; the single-cell probe came in at 46.9 s. Cost is
uniform across cells — `peaks_and_brightness_batched` embeds each candidate once and runs both
heads regardless of λ, so λ_bright = 0 buys no speedup.

## Output

`designs/lam-ex{P}_lam-em{P}_lam-bright{B}_lam-edit{E}/design_EGFP-<target>.csv`, one row per
(trial, round) with round 0 = the untouched scaffold. Same schema as `../lambda_sweep/`:
`pred_ex`, `pred_em`, `peak_err`, `pred_bright`, `lam_ex`, `lam_em`, `lam_bright`, `temp`, `k`,
`designed_seq`, with `ppl` deliberately blank.

**`lam_edit` is not a CSV column** — the engine records `lam_ex`, `lam_em` and `lam_bright` but
never `lam_edit`, in this effort and in every existing sweep alike. It is recoverable only from
the cell folder name, which is what downstream tooling parses (it is the `source` value carried
into the shortlists). Any analysis that slices by strategy must therefore read λ_edit from the
path, and a completeness check can only cross-validate the other three against the folder name.
