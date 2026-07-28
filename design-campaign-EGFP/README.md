# EGFP design campaign — recolour EGFP toward mOrange & EBFP

Take **one** well-characterised scaffold, **EGFP** (idx 171, PDB `4EUL`, ex 488 / em 507), and
redesign only its chromophore edit-window to move its spectrum toward **two** commercially available,
patent-expired, **non**-large-Stokes-shift targets that bracket the problem — a near, sequence-similar
**blue** shift and a distant, sequence-far **orange** shift:

| target | ex / em (nm) | colour | identity to EGFP | lineage |
|---|---|---|---|---|
| EBFP     | 380 / 440 | blue   | 99.2 % | avGFP (Clontech, patent expired) |
| mOrange  | 548 / 562 | orange | 30.4 % | DsRed/mFruit (Tsien, expired 2021) |

> **Consolidated scope.** This campaign was narrowed to the two targets above and the strategies
> below (with the parameters here). Earlier exploratory targets (**mEmerald**, **mCherry**), other
> `λ_bright` / `λ_edit` settings, and the previous post-hoc brightness-filter effort now live under
> [`archive/`](archive/) — see [`archive/README.md`](archive/README.md). Strategies 1–4 were the
> original consolidated set; **strategy 5 (`msa-guided/`) was added afterwards**, once the family
> MSA showed how little the ESM-2 proposal was constraining the search, and **strategy 6
> (`msa-gibbs/`) after that** as its unguided control.

All six design strategies here share the **same** scaffold, target palette, edit window, and
underlying engine (`fpdesign.campaign`), and differ **only** in how residues are selected. This lets
us compare the strategies head-to-head on the identical design problem. Two axes account for every
difference between them — where candidate residues come from, and whether anything steers the choice
among them — so the set forms a 2 × 2:

| | unguided (gibbs) | surrogate-guided |
|---|---|---|
| **ESM-2 proposal** | 1. `gibbs-sampling/` | 2. `guided-design/`, 3. `guided-design-constraint/`, 4. `brightness-guided/` |
| **family profile** | 6. `msa-gibbs/` | 5. `msa-guided/` |

Reading down a column isolates the **proposal**; reading across a row isolates the **steering**.
Strategy 6 exists because without it you cannot tell whether the MSA guide's advantage comes from
the family profile or from the λ retuning that came with it.

> **No oracle / in-sample predictions.** The surrogate that *guides* the search is the same model
> that *predicts* the resulting `(ex, em)`, and it was trained on all FP data (these popular FPs
> included). Recorded `pred_ex/pred_em` (and `pred_bright`) are therefore the models' **own**,
> in-sample, optimistic scores — useful as the design objective, **not** validation. The real judge
> is experiment.

## Layout

```
design-campaign-EGFP/
├─ make_pairs.py                       # setup: build the 2 EGFP→target pairs CSV
├─ pairs/campaign_pairs_egfp.csv       # shared: the 2 scaffold→target pairs (EBFP, mOrange)
├─ design_windows_egfp_tierB.json      # shared: 5 Å Tier-B edit window (copied into each effort)
│
├─ gibbs-sampling/                     # STRATEGY 1 "gibbs": pure ESM-2 masked-LM (target-free)
│  ├─ design_campaign.py                #   driver (strategy="gibbs", target_free=True)
│  ├─ run_campaign.sh
│  └─ designs/design_EGFP.csv           #   single target-free trajectory CSV
│
├─ guided-design/                      # STRATEGY 2 "spectra guide": surrogate-guided (peaks only)
│  ├─ design_campaign.py                #   driver (strategy="guided", λ_ex=λ_em=20)
│  ├─ run_campaign.sh
│  └─ designs/design_EGFP-<target>.csv
│
├─ guided-design-constraint/           # STRATEGY 3 "constrained spectra guide": guided + λ_edit=10
│  ├─ design_campaign.py                #   driver (adds the edit penalty)
│  ├─ run_campaign.sh
│  └─ designs_lam-edit10/design_EGFP-<target>.csv
│
├─ brightness-guided/                  # STRATEGY 4 "DMS guide": surrogate + brightness + edit penalty
│  └─ guided_design/
│     ├─ design_campaign.py             #   driver (λ_bright=60, λ_edit=10)
│     ├─ run_campaign.sh
│     └─ designs_lam-bright60_lam-edit10/design_EGFP-<target>.csv
│
├─ lambda_sweep/                       # SIDE STUDY: strategy 4 across a 3×3×3 grid of its λ weights
│  ├─ design_campaign.py               #   driver (per-cell outdir, pseudo-perplexity skipped)
│  ├─ run_sweep.sh                     #   27 cells × 12 trials per target, ~35 min per target
│  ├─ visualize_sweep.ipynb            #   heatmaps: mutations, |Δex|/|Δem|, ID & brightness hits
│  └─ designs/lam-ex{P}_lam-em{P}_lam-bright{B}_lam-edit{E}/design_EGFP-<target>.csv
│
├─ msa-guided/                         # STRATEGY 5 "MSA guide": family profile replaces ESM-2
│  ├─ build_msa_pssm.py                #   setup: 763-seq alignment → per-position family alphabets
│  ├─ msa_pssm_egfp.json               #   the PSSM (zero-frequency residues dropped outright)
│  ├─ design_campaign.py               #   driver (unit λ scale at T=1, pseudo-perplexity skipped)
│  ├─ check_scale.py                   #   selection-entropy probe vs the archived λ≈1 control
│  ├─ run_sweep.sh                     #   125 cells × 12 trials, both targets, ~5 h
│  └─ designs/lam-ex{P}_lam-em{P}_lam-bright{B}_lam-edit{E}/design_EGFP-<target>.csv
│
├─ msa-gibbs/                          # STRATEGY 6 "MSA gibbs": strategy 5's proposal, no steering
│  ├─ design_campaign.py               #   driver (target-free; reads ../msa-guided/msa_pssm_egfp.json)
│  ├─ run_campaign.sh                  #   96 trials × 3 rounds, ~11 s (no ESM-2 forward, no candidate scoring)
│  └─ designs/design_EGFP.csv          #   target-free COLS_FREE schema, one effort per scaffold
│
├─ make_shortlist_case.py              # build ONE per-strategy shortlist xlsx (ID + bright + diverse)
├─ make_batch.py                       # pick the wet-lab batch out of those shortlists
├─ scale_to_96.sh                      # scale the consolidated efforts to 96 trials (resumable)
├─ gen_shortlists.sh                   # watcher: emit each shortlist as its run finishes
├─ visualize_campaign.ipynb            # cross-strategy analysis + all figures
├─ figures/                            # saved PNGs from the notebook
├─ shortlists/                         # final wet-lab shortlists: one xlsx per (strategy × target)
│  └─ FPdesign-batch1.xlsx             #   batch 1: 8 MSA-guide candidates + 2 MSA-gibbs controls
└─ archive/                            # dropped targets, param variants, old single-file shortlists
```

Shared models live in the repo's `fpdesign/models/`:

- **Surrogate (guide + `(ex, em)` predictor):** ESM-2 `cnn-max-d1` trained on all FP data
  (`fpdesign/models/surrogate_cnn-max-d1_alldata.pt`, train MAE ≈ 5.2 nm).
- **Brightness classifier (guide for strategy 4):** `cnn-max-d2` bright/dim classifier trained on the
  sub-40k GFP DMS (`fpdesign/models/brightness_cnn-max-d2_40k.pt`, val AUROC ≈ 0.98). It emits a raw
  logit (bright ⇔ logit > 0); the campaign z-scores it, so its affine scale is irrelevant to ranking.

## Edit window (Tier-B)

All strategies edit the identical positions from `design_windows_egfp_tierB.json`: the 5 Å
heavy-atom chromophore pocket plus Tier-B H-bond partners (EGFP: Q95, H149, T204). **pos2 is forced
to aromatics**; the chromophore Gly and the maturation-catalytic Arg/Glu are **fixed**; the Tier-B
partners are restricted to `{S,T,Y,N,Q,D,E,H,K,R,W}`. Only the EGFP scaffold is edited, so no
scaffold↔target identity band is imposed (identity is metadata only).

## The five strategies

Each trial starts from the EGFP sequence and does **any-order masking** (a fresh random permutation
of the window positions each iteration). At each visited position we mask it, read the ESM-2
conditional logits, and keep the top-k allowed residues; strategies 1–4 differ in what happens next.
Strategy 5 changes the first step instead, replacing ESM-2 with the family alignment.

**1. `gibbs` (`gibbs-sampling/`) — pure ESM-2 masked-LM (target-free).** A true Gibbs draw from the
masked-LM conditional `p(x_i | x_{−i})`: sample directly from `softmax(logp / T)` at `T=1` with **no
target** term. The surrogate is loaded only to record each design's own `(ex, em)` as a diagnostic.
Because it never uses a target, a run is a single design effort **per scaffold** → one target-free
`designs/design_EGFP.csv`.

**2. `spectra guide` (`guided-design/`) — surrogate-guided (peaks only).** Splice each candidate,
predict `(ex, em)` with the surrogate, and sample at temperature `T` from
`s = z(logp_ESM) − λ_ex·z(|Δex|) − λ_em·z(|Δem|)`. Defaults: `T=10, k=10, λ_ex=λ_em=20`.

**3. `constrained spectra guide` (`guided-design-constraint/`) — guided + edit penalty.** Same as (2)
plus `− λ_edit·z(is_edit)`, where `is_edit=1` if a candidate residue differs from the **scaffold**
residue at that position. This steers toward **fewer** mutations from the (bright, well-folded) parent;
forced positions (e.g. pos2→aromatic) are unaffected. Run at `λ_edit=10` → `designs_lam-edit10/`.

**4. `DMS guide` (`brightness-guided/`) — surrogate + brightness + edit penalty.** Adds the brightness
term as well:
`s = z(logp_ESM) − λ_ex·z(|Δex|) − λ_em·z(|Δem|) + λ_bright·z(pred_bright) − λ_edit·z(is_edit)`.
`λ_bright` steers toward designs the classifier calls bright. Run at `λ_bright=60, λ_edit=10` →
`designs_lam-bright60_lam-edit10/`; `pred_bright` is logged per design. In analysis this set is
restricted to designs that are both **in-distribution** and **predicted bright** (the "DMS guide -
bright" group).

**5. `MSA guide` (`msa-guided/`) — the family alignment proposes, not ESM-2.** Same score as (4),
but `z(logp_ESM)` becomes `z(logp_MSA)`: the Henikoff-weighted residue frequency of the 763-sequence
FP family at that aligned column. Residues the family **never** uses there are dropped from the
alphabet outright (211 of 500 position-residue combinations), so they can never be selected. The
motivation is calibration — ESM-2 650M scores 12.6 % masked top-1 on EGFP against 66–80 % on
ordinary proteins of similar length, and 22.6 % of the edits strategies 1–4 actually made place a
residue no aligned FP uses at that column. Because every term is z-scored, λ is the relative weight,
so this effort re-scales to `λ = 1, T = 1` (all terms equal) instead of the inherited `λ = 20–60,
T = 10`. See [`msa-guided/README.md`](msa-guided/README.md).

**6. `MSA gibbs` (`msa-gibbs/`) — strategy 5's proposal with the steering removed.** Target-free like
(1), but drawing each residue from `softmax(top-k family log-frequency / T)` at `T = 1` instead of
from ESM-2's masked-LM conditional. It reads **the same PSSM file** as (5) — referenced, not copied —
so the two provably sample the identical distribution, which is what makes the guided-vs-unguided
comparison exact. There is no brightness term (the engine only accepts one with `strategy="guided"`,
and steering is precisely what this removes), so `pred_bright` is scored afterwards by the same
classifier every other strategy is judged by. The run costs **11 s** for 96 trials, against roughly
an hour for (1): Gibbs never scores candidate residues with the surrogate, and a profile lookup
replaces the ESM-2 forward pass. See [`msa-gibbs/README.md`](msa-gibbs/README.md).

Against (1) on the identical 288-design budget, swapping the proposal improves every axis — 18.0
mutations from EGFP vs 22.7, 8 % in-distribution vs 0 %, best peak error 29.2 vs 44.7 nm (mOrange)
and 27.1 vs 38.6 nm (EBFP). But **both unguided strategies produce zero predicted-bright designs out
of 288**, which is the cleanest statement available of what each half of the MSA guide contributes:
the family profile is why its designs are chemically plausible and in-distribution, and the
brightness term is why any of them are usable.

## Run

Each effort has a driver + a detachable wrapper. From inside the effort's folder:

```bash
# gibbs (target-free)
cd design-campaign-EGFP/gibbs-sampling
setsid bash run_campaign.sh < /dev/null > /dev/null 2>&1 &

# spectra guide
cd design-campaign-EGFP/guided-design
python design_campaign.py --probe                       # time one pair, project total, exit (no writes)
setsid bash run_campaign.sh < /dev/null > /dev/null 2>&1 &

# constrained spectra guide (λ_edit=10)
cd design-campaign-EGFP/guided-design-constraint
setsid bash run_campaign.sh --lam-edit 10 < /dev/null > /dev/null 2>&1 &

# DMS guide (λ_bright=60, λ_edit=10)
cd design-campaign-EGFP/brightness-guided/guided_design
setsid bash run_campaign.sh --lam-bright 60 --lam-edit 10 < /dev/null > /dev/null 2>&1 &

# monitor any effort
tail -f "$(cat .last_log)"
pgrep -af design_campaign.py            # empty = finished/stopped
```

Useful flags: `--trials --iters --temp --k --lam-ex --lam-em --lam-bright --lam-edit --pairs-limit
--ppl {all,endpoints}`. `--rescore` refills `pred_ex/pred_em/peak_err` in existing CSVs without
re-running the design loop; `--backfill-ppl` fills the blank intermediate-round ppl cells after an
`--ppl endpoints` run. Runs are **resumable at trial granularity** (re-running with more `--trials`
appends the missing trials).

## Output

One CSV per pair (guided variants) or per scaffold (gibbs), with one row per **(trial, round)**;
round 0 = the untouched scaffold, rounds 1..N = after each iteration. Core columns:

`pair, scaffold_name, scaffold_idx, scaffold_pdb, target_name, target_idx, selection, scaffold_ex,
scaffold_em, target_ex, target_em, seq_id_scaf_target, trial, round, n_editable, temp, k, lam_ex,
lam_em, pred_ex, pred_em, peak_err, ppl, ident_to_scaffold, designed_seq, scaffold_seq, target_seq`

- `peak_err = ½(|pred_ex − target_ex| + |pred_em − target_em|)` (mean of the two peak errors), with
  `pred_ex/pred_em` from the surrogate.
- the brightness-guided ("DMS guide") CSVs add `pred_bright, lam_bright`.
- the two gibbs CSVs are **target-free** (no `target_*` / `peak_err` columns); reconstruct per-target
  error from their `pred_ex/pred_em` as needed (the notebook's `load_target_free` does this).
- the MSA strategies leave `ppl` blank on purpose — ESM-2 pseudo-perplexity is not their naturalness
  measure, and the family log-likelihood under the PSSM is recoverable from the CSV without a GPU.

## Analysis

`visualize_campaign.ipynb` is the cross-strategy view (all figures land in `figures/`):

- **Overview & per-pair error** — designs vs scaffold vs target in `(ex, em)` space; per-pair
  surrogate design error and convergence vs iteration round. The MSA-gibbs convergence panel is flat
  by construction: a PSSM is context-independent, so each round is an independent redraw rather than
  a refinement, and the chain has no memory to converge with.
- **In-distribution / OOD check** — ESM-2 max-pool embeddings of each strategy's designs projected
  onto the four-scaffold GFP-DMS clouds (PCA), shown as KDE density contours alongside the 758 curated
  true FPs, with a nearest-neighbour distance OOD metric. The reference is **40,000 DMS sequences
  (10k per scaffold: avGFP, amacGFP, cgreGFP, ppluGFP)**; a design is "in-distribution" if its NN
  distance ≤ the DMS 99th percentile (`p99 ≈ 30.2`). One panel per strategy × target (10) plus the
  OOD metric. The MSA-guide (29–31 % OOD) and DMS-guide (36–37 %) designs stay largely
  in-distribution; the pure-guided (85–98 %), MSA-gibbs (92 %) and ESM-gibbs (100 %) designs
  extrapolate much further out.
- **Top-10 per strategy (EBFP, mOrange)** — two bar plots per target (ex and em) that **load the
  finalized `shortlists/*.xlsx`** (rather than re-selecting designs), so the figures match the wet-lab
  hand-off exactly. Each bar = mean ± sd of that strategy's shortlisted top-10, with the individual
  designs overlaid, a green bar for the EGFP scaffold, and a dashed line at the target. Bars run
  **unguided first, then guided**, so each target reads left-to-right as "what the proposal alone
  gives you" → "what adding the surrogate gives you". mOrange shows all six strategies; EBFP shows
  the four run for it (both gibbs strategies, **DMS guide - bright** and **MSA guide - bright**).
- **Mutation load of the top-10** (`campaign_top10_mutations.png`) — the same shortlists counted as
  substitutions from EGFP rather than predicted peaks, against the 25-position window ceiling. The
  ordering tracks the objective exactly: unguided sits near the ceiling (gibbs 22–23, MSA gibbs
  18–19), spectra guide 21.1, +edit penalty 15.5, and the brightness-steered strategies 9–11.
  **MSA guide is the most parsimonious on both targets** (9.7 EBFP, 9.6 mOrange) as well as the most
  accurate in-distribution strategy on mOrange, so it is not buying peak accuracy with mutations.
  Worth staring at: **real EBFP is 2 substitutions from EGFP**, while our best confidently-bright
  shortlist entry needs ~10 and still misses by 12.5 nm — the winning λ cell runs `λ_edit = 0`, so
  parsimony was never rewarded, which makes an edit-penalty-weighted rerun worth doing for EBFP
  specifically.
- **Strategy comparison** — mean/best surrogate error per strategy on the two shared pairs, plus a
  uniform `% predicted-bright` scored by the same cnn-max-d2 classifier across every strategy.

Regenerate everything with:

```bash
conda run -n esm2-fp-design jupyter nbconvert --to notebook --execute --inplace visualize_campaign.ipynb
```

The design CSVs record `pred_ex` / `pred_em` / `pred_bright`, so the notebook never recomputes those,
but the **ESM-2 max-pool embedding is not recorded anywhere** — it exists only to place designs on the
DMS PCA and to run the ID test, and it is the expensive part of both the notebook and
`make_shortlist_case.py`. `embed_cache.py` keys those vectors by sequence in `.embed_cache/`
(gitignored, ~36 MB, safe to delete) so each sequence is embedded once ever. This matters now that the
MSA sweep contributes ~5.4k designs: a full notebook run does **14,227 embedding lookups against
7,316 distinct sequences**, so more than half are served from the cache even within a single run. It
turned the run measured when the cache was added from **5m27s cold to 2m44s warm**, and a shortlist
rebuild from ~80 s to 13 s; with `msa-gibbs/` included a warm run is **3m29s**.

## Wet-lab shortlist

The final hand-off lives in `shortlists/` as **one xlsx per (strategy × target)** (the consolidated
efforts were scaled to **96 trials** each via `scale_to_96.sh`):

- `shortlist_mOrange_gibbs.xlsx`, `shortlist_mOrange_MSA-gibbs.xlsx`,
  `shortlist_mOrange_spectra-guide.xlsx`, `shortlist_mOrange_constrained-spectra-guide.xlsx`,
  `shortlist_mOrange_DMS-guide.xlsx`, `shortlist_mOrange_MSA-guide.xlsx`
- `shortlist_EBFP_gibbs.xlsx`, `shortlist_EBFP_MSA-gibbs.xlsx`, `shortlist_EBFP_DMS-guide.xlsx`,
  `shortlist_EBFP_MSA-guide.xlsx`

Each file starts with the two references (**EGFP** scaffold + the target, with their true dataset
ex/em) followed by the **top-10 diverse** designs (greedy, ≥ 5 residues apart, ranked by surrogate
peak error). Every case pools **all 3 iteration rounds of every trial** before selecting. The
**DMS-guide** and **MSA-guide** files then restrict the pool to designs that are both
**in-distribution** (NN-distance to the 40k reference ≤ p99 ≈ 30.2) **and confidently predicted
bright**; the other strategies take the plain closest-10. "Confidently" means classifier
**logit > 0.5** (`BRIGHT_T` in `make_shortlist_case.py`), not the model's own `> 0` decision
boundary: designs were clearing 0 by hundredths of a logit, which is a 0.51-probability call and not
worth a wet-lab slot. `is_bright` in the output still reports the model's plain `> 0` verdict. Every design row is annotated with `is_id`, `is_bright`,
`bright_logit`, predicted ex/em, `n_mut_vs_EGFP` (substitutions from the scaffold — a plain Hamming
distance, since designs only ever substitute inside the Tier-B window), the `source` run it came
from, and an E. coli codon-optimized DNA sequence. The reference rows carry it too, so EGFP reads 0
and EBFP reads 2; mOrange is left blank because at 236 aa it is a different length from the 239-aa
scaffold and a Hamming distance would be meaningless. Build one file (or rebuild all):

```bash
conda run -n esm2-fp-design python make_shortlist_case.py mOrange_DMS   # -> shortlists/shortlist_mOrange_DMS-guide.xlsx
# cases: mOrange_gibbs     mOrange_spectra  mOrange_constr  mOrange_DMS  EBFP_DMS
#        mOrange_MSA       EBFP_MSA
#        mOrange_MSAgibbs  EBFP_MSAgibbs    EBFP_gibbs
```

The two **gibbs** strategies are target-free, so one 288-design run backs both of their per-target
files and only the ranking differs. Neither produces a single predicted-bright design, so both take
the plain closest-10 — the ID-and-bright filter would return an empty pool.

**Both "- bright" strategies pool their whole λ sweep**, so neither is judged on a single setting.
MSA guide pools its **125 cells** (12 trials × 3 rounds each) — 2,635 (mOrange) and 2,835 (EBFP)
unique designs, of which 782 and 572 are ID & confidently bright. DMS guide pools its **27 cells** (λ_ex=λ_em ∈
10/20/30 × λ_bright ∈ 40/50/60 × λ_edit ∈ 10/15/20) on top of the tuned 96-trial run — 1,110 and 948
unique, of which 131 and 394 are ID & confidently bright. `source` records which cell every pick came from. Note
the tuned setting also exists as a sweep cell and the two are byte-identical, so the tuned run is
listed first and dedupe attributes shared designs to it.

MSA guide still draws from ~2.5× more candidates, so this is not an *equal*-budget comparison — but
pooling was worth doing precisely because it tests whether pool depth is the explanation. **It is
not.** Quadrupling the DMS-guide pool (287 → 1,110 for mOrange, 284 → 948 for EBFP) moved its best
peak error from 24.2 to 23.7 nm and from 7.9 to 7.9 nm. The 3.4-vs-23.7 nm gap on mOrange survives
giving the ESM-based strategy four times as many candidates to choose from.

Selecting on the same criterion the shortlist uses, this is where the strategies land:

| target | strategy | top-10 mean ex/em (nm) | best peak err | ID | bright |
|---|---|---|---|---|---|
| mOrange (548/562) | gibbs | 495.4 / 515.2 | 44.7 nm | 0/10 | 0/10 |
| mOrange | MSA gibbs | 508.8 / 525.1 | 29.2 nm | 0/10 | 0/10 |
| mOrange | spectra guide | 544.5 / 562.4 | **1.7 nm** | 0/10 | 0/10 |
| mOrange | constrained spectra guide | 542.6 / 562.4 | **0.8 nm** | 1/10 | 0/10 |
| mOrange | DMS guide - bright | 511.9 / 536.3 | 23.7 nm | 10/10 | 10/10 |
| mOrange | **MSA guide - bright** | 541.3 / 564.3 | **3.4 nm** | **10/10** | **10/10** |
| EBFP (380/440) | gibbs | 426.2 / 475.6 | 38.6 nm | 0/10 | 0/10 |
| EBFP | MSA gibbs | 433.7 / 479.8 | 27.1 nm | 1/10 | 0/10 |
| EBFP | **DMS guide - bright** | 400.5 / 463.7 | **7.9 nm** | **10/10** | **10/10** |
| EBFP | MSA guide - bright | 395.3 / 460.8 | 12.5 nm | 10/10 | 10/10 |

On **mOrange** the MSA guide is the first strategy to be simultaneously on-target *and*
in-distribution *and* confidently bright. Those used to be mutually exclusive: the strategies that
hit the mOrange peaks (spectra guide, 1.7 nm) were 0/10 on both filters, and the only strategy that
passed both filters (DMS guide) missed by 24 nm — still 23.7 nm after its whole λ sweep is pooled
in. At 3.4 nm and 10/10 on both filters, the MSA guide closes that gap.

**On EBFP it does not, and raising the brightness bar is what exposed that.** At the old `logit > 0`
threshold the MSA guide's best EBFP design was 5.6 nm, comfortably ahead of the DMS guide's 7.9 nm.
But that design scored a brightness logit of **0.04** — a 0.51-probability call. Requiring `> 0.5`
drops it and the next MSA-guide design in is 12.5 nm, so the DMS guide's 7.9 nm (logit 2.21) now
wins the blue target outright. The MSA guide's EBFP advantage was real only if you were willing to
bet a well on a coin-flip brightness call; its mOrange advantage survives the stricter bar
untouched. This is worth remembering whenever a strategy wins by a small margin on a filtered pool:
check the margin on the filter, not just on the objective.

The two unguided rows say which half of that is the proposal's doing. **MSA gibbs beats ESM gibbs
without any steering at all** — 29.2 vs 44.7 nm on mOrange, 27.1 vs 38.6 nm on EBFP, and the first
non-zero ID count any unguided strategy has produced — so the family profile is genuinely a better
place to sample from. But both are **0/10 bright**, and so is every strategy without a brightness
term. Neither ingredient is sufficient alone.

`gen_shortlists.sh` watches a `scale_to_96.sh` run and emits each file automatically as its run
completes. The barplots in `visualize_campaign.ipynb` load these files directly.

## Batch 1 — first wet-lab order

`shortlists/FPdesign-batch1.xlsx` (built by `make_batch.py`) is the **10 constructs going out
first** — 8 candidates and 2 controls, all from the two MSA-proposal strategies, because those are
the only ones that put designs in the on-target *and* in-distribution *and* predicted-bright corner
at all.

The eight candidates are **stratified by mutation load** rather than taken as ranks 1–N. Ranking
purely on surrogate error clusters the batch wherever the surrogate happens to be most confident,
which wastes the chance to ask how many edits the scaffold actually tolerates. Within a tier the
pick is still the lowest peak error, so quality decides among equals. Tier boundaries are set per
target because the two top-10s are distributed very differently:

- **mOrange** — 6, 6 │ 9, 9, 9 │ 10, 10, 11, 12, 14. `high` takes the 14, the top of the range; a 10
  sits one edit above the medium cluster and would not test anything new.
- **EBFP** — 7, 7, 8, 8 │ 10, 11, 11, 11 │ 12, 12. The low cluster is tight and the upper half
  crowds into 11–12, so the ladder can only span 8 → 11 → 12.

Note that shortlist design names are **rank-derived** (`<target>_<code>_<NN>` is assigned in
peak-error order), so a name does not pin a sequence — rebuilding a shortlist under different
criteria silently repoints the same name at a different design. `make_batch.py` therefore records
the expected edit count next to each pick and asserts it, turning a shifted shortlist into a loud
failure rather than a quietly wrong order sheet.

| batch id | design | target | tier | pred ex/em | err | muts | ID | bright logit |
|---|---|---|---|---|---|---|---|---|
| B1_01 | mOrange_MSA_03 | mOrange | low | 540.5 / 562.6 | 4.1 nm | **6** | yes | 0.78 |
| B1_02 | mOrange_MSA_07 | mOrange | low | 540.2 / 563.8 | 4.8 nm | **6** | yes | 0.90 |
| B1_03 | mOrange_MSA_01 | mOrange | medium | 541.1 / 562.0 | **3.4 nm** | 9 | yes | 1.09 |
| B1_04 | mOrange_MSA_04 | mOrange | medium | 541.5 / 563.9 | 4.2 nm | 9 | yes | 1.21 |
| B1_05 | mOrange_MSA_10 | mOrange | high | 538.6 / 562.9 | 5.1 nm | 14 | yes | 1.55 |
| B1_06 | EBFP_MSA_01 | EBFP | low | 394.7 / 450.4 | **12.5 nm** | 8 | yes | 0.76 |
| B1_07 | EBFP_MSA_02 | EBFP | medium | 382.9 / 466.4 | 14.6 nm | 11 | yes | 3.11 |
| B1_08 | EBFP_MSA_06 | EBFP | high | 393.8 / 463.7 | 18.8 nm | 12 | yes | 2.30 |
| B1_09 | mOrange_MSAgib_01 | mOrange | — | 507.1 / 544.4 | 29.2 nm | 21 | no | −11.70 |
| B1_10 | mOrange_MSAgib_02 | mOrange | — | 516.1 / 529.8 | 32.1 nm | 20 | no | −9.22 |

**The ladder already pays off before anything is expressed.** Across mOrange, error is flat from 6
to 14 edits (4.1 / 4.8 / 3.4 / 4.2 / 5.1 nm) — the extra substitutions buy nothing the surrogate can
see, so if the 6-edit B1_01 works it is strictly the better construct, and the 14-edit B1_05 is
purely a probe of how much edit load the scaffold tolerates. EBFP behaves the same way here
(12.5 / 14.6 / 18.8 nm), and in fact slightly *worsens* with load, so its ladder is a pure
robustness probe rather than an accuracy trade.

All eight guided constructs clear the brightness classifier by a real margin — the minimum logit in
the batch is **0.76**, against **0.04** in the earlier `> 0` draft. That is the point of the
stricter bar, and it cost accuracy on the blue target: the 5.6 nm EBFP design that headlined the
previous batch scored 0.04 and is gone, leaving 12.5 nm as the best confidently-bright EBFP design
the MSA guide has.

The two controls are both taken from the **mOrange** MSA-gibbs ranking, i.e. the two unguided
designs that drift furthest toward orange. The brightness bar does not apply to them — MSA gibbs
produces no predicted-bright designs at all. They are *expected to underperform*: 20–21 substitutions,
both out-of-distribution, both predicted non-bright (logit −11.7 and −9.2). They are in the batch to
answer the one thing the surrogates cannot — whether the family profile alone yields folded,
fluorescent protein. If they light up, the proposal is doing more work than the brightness
classifier credits it for; if they are dark while the eight guided designs are not, the
ID-and-bright filter is earning its place.

Substitutions vs the EGFP scaffold (1-based on the 239-aa construct, so the chromophore is
T66-Y67-G68):

```
B1_01  F47M L61I T66M Y146F T204H S206M
B1_02  F47L L65F T66M V69S Y146F T204H
B1_03  F47L L65F T66M Y93F Q95T V113I Y146F T204H S206T
B1_04  F47I L61I L65M T66M V69S V113I Y146F H149W T204H
B1_05  L45V F47M V62G T63C L65M T66M V69M Q95T Y146F V151L F166I I168Q T204H S206T
B1_06  T63H T64A T66S Y93L Q95T N122S H149D S206A
B1_07  L43I F47I V62A T63H T64S T66S V69I N122S Y146F N147G S206A
B1_08  L43V L45I F47I T63H T66S Y93L Q95W N122A Y146M N147A V151I S206N
B1_09  L45I F47A L61I V62S T63P T64Q L65C T66M V69S Q70R Y93E V113T N122S Y146E H149T V151M
       F166S I168M T204H S206I L221F
B1_10  L43I L45S F47L V62S T64S L65F T66M V69S Q70K Y93F Q95W V113I Y146E N147I H149T V151R
       F166I T204R S206L L221Q
```

Two things to note before ordering. **Every one of the ten changes position 66** — T66M in all five
mOrange designs and both controls, T66S in all three EBFP designs — so the chromophore-forming
residue is the campaign's single most-used lever and a systematic failure there takes out the whole
batch at once. Each target also has a shared core beyond that: all five mOrange designs carry
**T66M + Y146F + T204H** (and all edit position 47), and all three EBFP designs carry **T63H +
T66S** while every one of them also edits 122 and 206, though not always to the same residue
(S206A/S206A/S206N, N122S/N122S/N122A). Those cores are the batch's real hypotheses; the remaining
substitutions are variations around them.

Second, **no construct in the batch uses Y67H**, the classic EBFP chromophore substitution. The
three guided EBFP designs blue-shift through the T63H/T66S route, reshaping the proton-relay network
instead of replacing the chromophore tyrosine. That is the most interesting claim in the batch and
also the least precedented — the surrogates predict it, but it is not how EBFP is normally built.
An earlier draft of batch 1 included an unguided EBFP control that had independently arrived at
Y67H; moving both controls to the mOrange ranking removed the batch's only Y67H example, so nothing
here tests the conventional route.

The xlsx carries every column of the source shortlists (`n_mut_vs_EGFP`, `is_id`, `is_bright`,
`bright_logit`, the λ cell in `source`, the amino-acid sequence and an E. coli codon-optimized DNA
sequence), plus `batch_id`, `mut_tier`, `pred_peak_err_nm` and the originating `shortlist_file`.
EGFP, mOrange and EBFP references sit at the top with their TRUE measured peaks. Rebuild with:

```bash
conda run -n esm2-fp-design python make_batch.py   # -> shortlists/FPdesign-batch1.xlsx
```

Picks are pinned by shortlist design name, not by rank, so re-running after a shortlist rebuild
either reproduces the same ten sequences or fails loudly.

## Reproduce the shared assets

```bash
cd design-campaign-EGFP
python make_pairs.py        # regenerate pairs/campaign_pairs_egfp.csv (EBFP, mOrange)
```

The edit window `design_windows_egfp_tierB.json` is derived one level up in the conventional campaign
tooling and copied into each effort so every strategy edits the identical positions.
