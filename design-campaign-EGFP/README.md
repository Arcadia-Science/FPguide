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
> below. Earlier exploratory targets (**mEmerald**, **mCherry**) and the previous post-hoc
> brightness-filter effort live under `archive/` — see
> `archive/README.md`. `archive/` is **gitignored and read by nothing**; if
> active code needs a file, that file does not belong there.

> **Two retirements.** The ESM-2 guided arm was originally run at **T = 10, λ = 20/60/10** in three
> separate folders (`guided-design/`, `guided-design-constraint/`, `brightness-guided/`) plus a
> 27-cell λ sweep. That whole arm was superseded by
> [`esm2_guided/`](esm2_guided/), which runs the MSA effort's exact λ grid on the
> real ESM-2 proposal, so the two arms finally differ **only** in the proposal distribution. The
> T=10 runs are in `archive/superseded-unmatched-runs/` and have no shortlists.
> Separately, the **constrained spectra guide** (peaks + edit penalty, no brightness term) was
> **retired outright**: its 20 matched-sweep cells were run and are still on disk, but they are
> excluded from every analysis. Strategy numbering keeps its gap so "strategy 4" still means what
> it means in the write-ups.

The five surviving strategies share the **same** scaffold, target palette, edit window, and
underlying engine (`fpdesign.campaign`), and differ **only** in how residues are selected. This lets
us compare them head-to-head on the identical design problem. Two axes account for every difference
between them — where candidate residues come from, and whether anything steers the choice among
them — so the set forms a 2 × 2:

| | unguided (gibbs) | surrogate-guided |
|---|---|---|
| **ESM-2 proposal** | 1. `gibbs-sampling/` | 2. + 4. `esm2_guided/` (λ-cell slices) |
| **family profile** | 6. `msa-gibbs/` | 5. `msa-guided/` |

Reading down a column isolates the **proposal**; reading across a row isolates the **steering**.
Strategy 6 exists because without it you cannot tell whether the MSA guide's advantage comes from
the family profile or from the λ retuning that came with it. Strategies 2 and 4 are **slices of one
sweep** rather than separate runs: `λ_bright = 0` contributes exactly nothing to the guided score,
so the peaks-only cells reproduce strategy 2 bit-for-bit while still logging `pred_bright` — which
is what lets one uniform ID-and-bright filter judge every strategy. Verified, not assumed — 12 of
12 designs came out identical with the classifier loaded and zero-weighted against absent
altogether. The check that measured it has been retired to `archive/check_lam_bright0.py`; its
result is recorded under "λ_bright = 0 equivalence" in
[`esm2_guided/README.md`](esm2_guided/README.md).

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
├─ references/                         # shared: the 2 rows every shortlist opens with
│  ├─ reference_EGFP-<target>.csv       #   seqs + TRUE peaks; tracked, run-tree-independent
│  └─ README.md                         #   provenance + why these are not read out of a run
│
├─ gibbs-sampling/                     # STRATEGY 1 "gibbs": pure ESM-2 masked-LM (target-free)
│  ├─ design_campaign.py                #   driver (strategy="gibbs", target_free=True)
│  ├─ design_campaign_benchmark.py      #   375-trial equal-budget variant
│  ├─ PROVENANCE.md                     #   ⚠ designs/ backs the gibbs shortlists — do not extend
│  ├─ designs/design_EGFP.csv           #   96 trials × 3 rounds; the shortlist source
│  └─ designs_benchmark375/design_EGFP.csv    #   375 × 3 = 1,125, for benchmark_report.py
│
├─ esm2_guided/                        # STRATEGIES 2 + 4: ESM-2 proposal at the MSA λ scale (T=1)
│  ├─ design_campaign.py               #   driver (per-cell outdir, pseudo-perplexity skipped)
│  ├─ run_sweep.sh                     #   the 5×5×5 grid, both targets
│  ├─ analyze.py                       #   per-slice table, λ heatmap, head-to-head vs strategy 5
│  ├─ metrics_<target>.csv             #   per-cell metrics, written by analyze.py
│  └─ designs/lam-ex{P}_lam-em{P}_lam-bright{B}_lam-edit{E}/design_EGFP-<target>.csv
│                                      #   125 cells = 5 (strategy 2) + 100 (strategy 4)
│                                      #   + 20 RETIRED constrained-guide cells, excluded
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
│  ├─ design_campaign.py                #   driver (target-free; reads ../msa-guided/msa_pssm_egfp.json)
│  ├─ design_campaign_benchmark.py      #   375-trial equal-budget variant
│  ├─ designs/design_EGFP.csv           #   target-free COLS_FREE schema, one effort per scaffold
│  └─ designs_benchmark375/design_EGFP.csv
│
├─ make_shortlist_case.py              # build ONE shortlist xlsx (ID + bright + diverse); --all
├─ make_batch.py                       # pick the wet-lab batch out of those shortlists
├─ benchmark_report.py                 # equal-budget comparison across all five strategies
├─ visualization.ipynb                 # the 5 five-strategy figures, both targets
├─ embed_cache.py, xlsx_io.py          # shared helpers (sequence-keyed embeddings, xlsx writer)
├─ figures_benchmark/                  # visualization.ipynb's own PNG+SVG output (the only figure dir)
├─ shortlists/                         # wet-lab shortlists: one xlsx per (strategy × target)
│  └─ FPdesign-batch1.xlsx             #   batch 1: 8 MSA-guide candidates + 2 MSA-gibbs controls
└─ archive/                            # GITIGNORED, read by nothing: dropped targets, the retired
                                       # T=10 arm, superseded shortlists, visualize_campaign.ipynb
                                       # and figures/, the frozen PNGs it produced, and the
                                       # check_lam_bright0.py pre-flight check
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
conditional logits, and keep the top-k allowed residues; strategies 1, 2 and 4 differ in what
happens next. Strategy 5 changes the first step instead, replacing ESM-2 with the family alignment.

All the guided strategies score candidates with the same expression, and differ only in which terms
carry non-zero weight:

```
s = z(logp_proposal) − λ_ex·z(|Δex|) − λ_em·z(|Δem|) + λ_bright·z(pred_bright) − λ_edit·z(is_edit)
```

`is_edit = 1` when a candidate residue differs from the **scaffold** residue at that position, so
`λ_edit` steers toward fewer mutations from the (bright, well-folded) parent; forced positions
(e.g. pos2→aromatic) are unaffected. Every term is z-scored across the k candidates, so **λ is a
relative weight** — which is why the two arms are only comparable once run at the same λ scale.

**1. `gibbs` (`gibbs-sampling/`) — pure ESM-2 masked-LM (target-free).** A true Gibbs draw from the
masked-LM conditional `p(x_i | x_{−i})`: sample directly from `softmax(logp / T)` at `T=1` with **no
target** term. The surrogate is loaded only to record each design's own `(ex, em)` as a diagnostic.
Because it never uses a target, a run is a single design effort **per scaffold** → one target-free
`designs/design_EGFP.csv`.

**2. `spectra guide` (`esm2_guided/`, `λ_bright = 0, λ_edit = 0`) — surrogate-guided
(peaks only).** Splice each candidate, predict `(ex, em)` with the surrogate, and sample at
temperature `T` from the peaks-only score. 5 cells × 75 trials at `T=1, k=10`.

**4. `DMS guide` (`esm2_guided/`, `λ_bright > 0`) — surrogate + brightness + edit
penalty.** The full score, with `λ_bright` steering toward designs the classifier calls bright and
`pred_bright` logged per design. 100 cells × 4 trials. In analysis this set is restricted to
designs that are both **in-distribution** and **predicted bright** (the "DMS guide - bright" group).

> **Retired: `constrained spectra guide`** (peaks + edit penalty, `λ_bright = 0, λ_edit > 0`) was
> strategy 3. Its 20 cells were run and are still on disk under `esm2_guided/designs/`,
> but nothing reads them — `analyze.py` drops them before pooling and `benchmark_report.py` has no
> row for them, so both report **105 cells, not 125**.

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

# strategies 2 + 4: the whole matched-λ grid, both targets
cd design-campaign-EGFP/esm2_guided
python design_campaign.py --probe                       # time one pair, project total, exit (no writes)
setsid bash run_sweep.sh < /dev/null > /dev/null 2>&1 &

# MSA guide (strategy 5) / MSA gibbs (strategy 6)
cd design-campaign-EGFP/msa-guided  && setsid bash run_sweep.sh    < /dev/null > /dev/null 2>&1 &
cd design-campaign-EGFP/msa-gibbs   && setsid bash run_campaign.sh < /dev/null > /dev/null 2>&1 &

# monitor any effort
tail -f "$(cat .last_log)"
pgrep -af design_campaign.py            # empty = finished/stopped
```

A single matched-λ cell is just the driver with its four weights, which is how a retired strategy
can still be reproduced without reviving its folder:

```bash
cd design-campaign-EGFP/esm2_guided
python design_campaign.py --lam-ex 1 --lam-em 1 --lam-bright 0 --lam-edit 1 --temp 1   # ex-strategy 3
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
- every guided CSV that had the classifier loaded adds `pred_bright, lam_bright` — which, in the
  matched sweep, is **all 125 cells**, including the `λ_bright = 0` ones where the term is
  zero-weighted. That is deliberate: it is what lets one uniform ID-and-bright filter apply to the
  peaks-only strategy too.
- the two gibbs CSVs are **target-free** (no `target_*` / `peak_err` columns); reconstruct per-target
  error from their `pred_ex/pred_em` as needed — `make_shortlist_case.build` does this for the
  `gibbs` cases, and `benchmark_report.py` for the target-free rows.
- the MSA strategies leave `ppl` blank on purpose — ESM-2 pseudo-perplexity is not their naturalness
  measure, and the family log-likelihood under the PSSM is recoverable from the CSV without a GPU.

## Analysis

Two scripts and one notebook, all read-only, all importing their filters from
`make_shortlist_case.py` so the numbers are identical by construction to what the shortlists select
on.

> **Prerequisite, not in git.** All three import `make_shortlist_case.py`, which loads the
> 40,000-sequence in-distribution reference cloud
> (`GFP_DMS/DMS_data/sub40k_maxpool.npz`, 197 MiB, gitignored) at import time, and the ESM-2 650M
> weights for embedding designs. Fetch the published cloud:
>
> ```bash
> gh release download reference-cloud-v1 -p sub40k_maxpool.npz -D GFP_DMS/DMS_data/
> ```
>
> Rebuilding it instead means re-downloading two published DMS studies and running ~85 GB of
> per-residue embedding — see the **Reproduce** block in
> [`GFP_DMS/README.md`](../GFP_DMS/README.md), ending in
> `python GFP_DMS/build_maxpool_cache.py`. Without the cloud all three exit with those two options
> rather than a traceback. The design CSVs, shortlists and `figures_benchmark/` in this folder are
> committed, so the *results* are readable without fetching or rebuilding anything.

```bash
python benchmark_report.py                      # equal-budget table, all five strategies
python esm2_guided/analyze.py          # per-slice table + λ heatmap for strategies 2 + 4
jupyter nbconvert --to notebook --execute --inplace visualization.ipynb   # the figures
```

**`benchmark_report.py` — equal budget.** Every strategy gets a comparable **raw-design budget of
≥ 1,125 per target** (raw = cells × trials × 3 iterations), so no row wins by having been allowed
to draw more. Strategy 5 is deliberately left at 4× budget and flagged as such — read it as a
best-achievable reference, not a like-for-like competitor. It reports, per strategy: unique
designs, best/mean surrogate error, `% in-distribution`, `% predicted-bright`, and the count and
best error among designs clearing **both** filters — the shortlist's own admission bar.

**`esm2_guided/analyze.py` — where inside the λ grid the ESM-2 arm works.** Per-slice
summary (strategy 2 vs strategy 4), a `λ_bright × λ_edit` heatmap of best ID-and-bright error, and
a head-to-head against strategy 5 pooled and filtered exactly as the shortlists do. It writes
`metrics_<target>.csv`.

**`visualization.ipynb` — the figures, both targets.** Five figures for the five strategies on the
**equal budget** `benchmark_report.py` defines:

1. the share of every pool that is in-distribution, and that is in-distribution **and** confidently
   bright — one grey pair of bars per strategy, both targets pooled into it, with the per-target
   rates printed below;
2. a light swarm of every mOrange design's predicted `(ex, em)` against the scaffold and target
   rules, with each strategy's best ten drawn over it in full colour on a mean ± 1 SD crossbar,
   where the two **brightness-guided** strategies (`DMS guide`, `MSA guide`) are first restricted to
   designs that are both in-distribution and confidently bright while the other three — never given
   a brightness term — are ranked unfiltered;
3. the same figure for EBFP;
4. the surrogate MAE of every strategy's best ten, **one target per panel on a shared y axis** — a
   strip and a mean ± 1 SD crossbar per strategy, orange mOrange left, blue EBFP right;
5. the two **peaks** of that same ten with both campaigns on one axis — excitation and emission,
   each strategy contributing an orange sub-column and a blue one, each a **raincloud** (half violin
   of that campaign's whole ~1,125-design pool, the best ten as rain beside it), against three rules
   (the two targets in their own colours, the EGFP scaffold in green). The target-free gibbs arms
   get one grey violin with both campaigns' tens on it.

Figures 2 – 4 bracket **Welch's t-tests** between the tens of the two columns each one compares —
every pair inside a proposal, plus the two brightness-guided strategies against each other, on each
peak in figures 2 and 3 and on MAE within each target in figure 4 — with Holm across each family of
five and Mann-Whitney U printed beside it.

A closing numbers-only section ranks the same pools on surrogate error alone: those designs land on
the target and pass neither filter (0 of 50 confidently bright on either target), because inside the
MSA guide's pool r(brightness logit, peak error) = +0.32 (mOrange) and +0.51 (EBFP) — the gate costs
3.95 nm and 14.57 nm respectively. **Colour names a protein** (orange mOrange, blue EBFP, green EGFP)
and appears only where one of them does: the two reference rules of figures 2 and 3, each labelled
once on the excitation panel, and figure 4's markers, the one panel where the two targets share an
axis. Every design elsewhere is neutral grey — and every categorical axis is labelled on two
levels, the steering rule
per column under a rule naming the proposal, so the campaign's 2 × 2 reads off the axis. It uses the
shortlists' own filters, imported from `make_shortlist_case.py`, and drops only their diversity
rule, whose cost it prints. It writes `figures_benchmark/`, this folder's only figure directory, and
costs ≈ 70 s of GPU for the 2,250 target-free sequences whose `pred_bright` the runs never recorded,
plus the ID embeddings if `.embed_cache/` is cold.

**Where the ID test comes from.** A design is "in-distribution" if the NN distance from its ESM-2
max-pool embedding to the **40,000-sequence GFP-DMS reference** (10k each of avGFP, amacGFP,
cgreGFP, ppluGFP) is ≤ that reference's 99th percentile (`p99 ≈ 30.15`). That reference is
`sub40k` itself — the very variants the brightness classifier was fitted and selected on — built by
`GFP_DMS/build_maxpool_cache.py`, so "in-distribution" and "inside the training distribution" are
the same statement. The design CSVs record
`pred_ex` / `pred_em` / `pred_bright`, so nothing recomputes those — but the **embedding is not
recorded anywhere**, and it is the expensive part of every consumer here. `embed_cache.py` keys
those vectors by sequence in `.embed_cache/` (gitignored, ~36 MB, safe to delete) so each sequence
is embedded once ever.

> **The frozen `figures/` was archived.** Its 13 PNGs were produced by `visualize_campaign.ipynb`,
> which was retired to `archive/` — it was written against the T=10 folder layout and six
> shortlists, four of which no longer exist. They are **not** regenerable from the current tree and
> `nbconvert` will not reproduce them, so they followed the notebook that made them into
> [`archive/figures/`](archive) — gitignored, kept on the authors' machines, and no longer published
> even though the write-up references them. Everything that can still be recomputed lives in the two
> scripts and the notebook above, whose own figures go to `figures_benchmark/`; that is now this
> folder's only figure directory.

## Wet-lab shortlist

The final hand-off lives in `shortlists/` as **one xlsx per (strategy × target)** — six files, one
per surviving case:

- `shortlist_mOrange_gibbs.xlsx`, `shortlist_mOrange_MSA-gibbs.xlsx`, `shortlist_mOrange_MSA-guide.xlsx`
- `shortlist_EBFP_gibbs.xlsx`, `shortlist_EBFP_MSA-gibbs.xlsx`, `shortlist_EBFP_MSA-guide.xlsx`

The retired ESM-2 arm has none. Its six superseded files are in `archive/superseded-shortlists/`
(gitignored, read by nothing); the strategy-2/4 comparison now lives in `benchmark_report.py`,
which scores those pools on the identical filters without minting a wet-lab hand-off for them.

Each file starts with the two references (**EGFP** scaffold + the target, with their true dataset
ex/em, read from `references/`) followed by the **top-10 diverse** designs (greedy, ≥ 5 residues
apart, ranked by surrogate peak error). Every case pools **all 3 iteration rounds of every trial**
before selecting. The **MSA-guide** files then restrict the pool to designs that are both
**in-distribution** (NN-distance to the 40k reference ≤ p99 ≈ 30.15) **and confidently predicted
bright**; the other strategies take the plain closest-10. "Confidently" means classifier
**logit > 0.5** (`BRIGHT_T` in `make_shortlist_case.py`), not the model's own `> 0` decision
boundary: designs were clearing 0 by hundredths of a logit, which is a 0.51-probability call and not
worth a wet-lab slot. `is_bright` in the output still reports the model's plain `> 0` verdict. Every
design row is annotated with `is_id`, `is_bright`, `bright_logit`, predicted ex/em, `n_mut_vs_EGFP`
(substitutions from the scaffold — a plain Hamming distance, since designs only ever substitute
inside the Tier-B window), the `source` run it came from, and an E. coli codon-optimized DNA
sequence. The reference rows carry it too, so EGFP reads 0 and EBFP reads 2; mOrange is left blank
because at 236 aa it is a different length from the 239-aa scaffold and a Hamming distance would be
meaningless. Build one file, or all six:

```bash
conda run -n spectrum-to-fp-design python make_shortlist_case.py mOrange_MSA
conda run -n spectrum-to-fp-design python make_shortlist_case.py --all
conda run -n spectrum-to-fp-design python make_shortlist_case.py --verify-refs   # references/ vs the runs
# cases: mOrange_gibbs  EBFP_gibbs  mOrange_MSAgibbs  EBFP_MSAgibbs  mOrange_MSA  EBFP_MSA
```

### The four MSA shortlists are FROZEN

Design names are **rank-derived** (`<target>_<code>_<NN>`, assigned in peak-error order), so a name
does not pin a sequence — rebuilding under a different threshold, model or input silently repoints
the same name at a different design. Wet-lab batch 1 was chosen off these files and pins those
names, so the four MSA cases are frozen:

> a frozen case still **rebuilds in full** — pool, dedupe, ID filter, brightness, greedy top-10 —
> and the result is compared against the xlsx on disk. Match ⇒ the file is **not rewritten** (xlsx
> bytes are not reproducible; rewriting would dirty `git status` on every passing run). Mismatch ⇒
> a hard failure naming which designs were dropped or added, which cells changed, and whether it is
> a pure re-rank. To accept a new selection deliberately: delete the file, rebuild to mint a new
> baseline, then re-verify every pick in `make_batch.py`.

That check is what makes the reproduction claim testable rather than aspirational — all four were
confirmed to rebuild identically after `REF_CSV` moved to `references/`.

The two **gibbs** strategies are target-free, so one 288-design run backs both of their per-target
files and only the ranking differs. Neither produces a single predicted-bright design, so both take
the plain closest-10 — the ID-and-bright filter would return an empty pool.

**MSA guide pools its whole λ sweep**, so it is not judged on a single setting: **125 cells**
(12 trials × 3 rounds each) → 2,635 (mOrange) and 2,835 (EBFP) unique designs, of which 782 and 572
are ID & confidently bright. `source` records which cell every pick came from.

Selecting on the same criterion the shortlist uses, this is where the strategies land. The two
guided ESM-2 rows are the matched-λ slices scored by `benchmark_report.py` on an equal raw-design
budget; the shortlisted rows are the frozen files:

| target | strategy | top-10 mean ex/em (nm) | best peak err | ID | bright |
|---|---|---|---|---|---|
| mOrange (548/562) | gibbs | 495.4 / 515.2 | 44.7 nm | 0/10 | 0/10 |
| mOrange | MSA gibbs | 508.8 / 525.1 | 29.2 nm | 0/10 | 0/10 |
| mOrange | spectra guide (matched λ) | — | 0.7 nm raw, **none** pass both filters | 0.6 % | 0 % |
| mOrange | DMS guide (matched λ) | 487.4 / 511.9 | 31.8 nm | 28.1 % | 3.8 % |
| mOrange | **MSA guide - bright** | 541.3 / 564.3 | **3.4 nm** | **10/10** | **10/10** |
| EBFP (380/440) | gibbs | 426.2 / 475.6 | 38.6 nm | 0/10 | 0/10 |
| EBFP | MSA gibbs | 433.7 / 479.8 | 27.1 nm | 1/10 | 0/10 |
| EBFP | spectra guide (matched λ) | — | 0.0 nm raw, **none** pass both filters | 0.1 % | 0 % |
| EBFP | DMS guide (matched λ) | 415.2 / 470.0 | 29.3 nm | 26.7 % | 8.3 % |
| EBFP | **MSA guide - bright** | 395.3 / 460.8 | **12.5 nm** | **10/10** | **10/10** |

The ESM-2 rows report pool-level ID/bright rates rather than a 10/10 count because they have no
shortlist to count out of; `best peak err` for them is the best error among designs clearing both
filters, the same bar the MSA rows are held to.

**The peaks-only strategy is the sharpest illustration of why both filters matter.** It gets
closest to the target of anything in the campaign — 0.7 nm on mOrange, 0.0 nm on EBFP — and **not a
single one of those designs is both in-distribution and predicted bright**. Optimizing the
surrogate alone produces sequences that hit the requested spectrum and are, by every other measure
available, not proteins worth expressing.

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
conda run -n spectrum-to-fp-design python make_batch.py   # -> shortlists/FPdesign-batch1.xlsx
```

Picks are pinned by shortlist design name, not by rank, so re-running after a shortlist rebuild
either reproduces the same ten sequences or fails loudly.

## Reproduce the shared assets

```bash
cd design-campaign-EGFP
python make_pairs.py                             # -> pairs/campaign_pairs_egfp.csv (EBFP, mOrange)
python make_shortlist_case.py --verify-refs      # references/ still agrees with every design run
```

The edit window `design_windows_egfp_tierB.json` is derived by
[`fpdesign/build_design_windows.py`](../fpdesign/build_design_windows.py) (which defaults to this
campaign's `pairs/campaign_pairs_egfp.csv`) and copied into each effort so every strategy edits the
identical positions. Regenerating it reproduces the `windows` payload exactly; only `meta.generated_at`
changes.

`references/reference_EGFP-<target>.csv` holds the two rows every shortlist opens with — the EGFP
and target sequences and their true measured peaks. They used to be read out of whichever design
CSV was convenient, which meant retiring a run broke **every** shortlist, including ones that had
nothing to do with it. They are checked in instead, so a shortlist rebuilds from a fresh clone with
no design run present; `--verify-refs` re-compares them against all 500 live design CSVs. See
[`references/README.md`](references/README.md).

> **Rule of thumb for this folder:** if active code reads a file, that file is tracked. `archive/`
> is gitignored, so nothing under it may be a dependency — that invariant is what this layout was
> reorganized to restore.
