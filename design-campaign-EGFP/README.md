# EGFP design campaign — recolour EGFP toward mOrange & EBFP

Take **one** well-characterised scaffold, **EGFP** (idx 171, PDB `4EUL`, ex 488 / em 507), and
redesign only its chromophore edit-window to move its spectrum toward **two** commercially available,
patent-expired, **non**-large-Stokes-shift targets that bracket the problem — a near, sequence-similar
**blue** shift and a distant, sequence-far **orange** shift:

| target | ex / em (nm) | colour | identity to EGFP | lineage |
|---|---|---|---|---|
| EBFP     | 380 / 440 | blue   | 99.2 % | avGFP (Clontech, patent expired) |
| mOrange  | 548 / 562 | orange | 30.4 % | DsRed/mFruit (Tsien, expired 2021) |

> **Consolidated scope.** This campaign was narrowed to the two targets above and the four strategies
> below (with the parameters here). Earlier exploratory targets (**mEmerald**, **mCherry**), other
> `λ_bright` / `λ_edit` settings, and the previous post-hoc brightness-filter effort now live under
> [`archive/`](archive/) — see [`archive/README.md`](archive/README.md).

All four design strategies here share the **same** scaffold, target palette, edit window, and
underlying engine (`fpdesign.campaign`), and differ **only** in how residues are selected. This lets
us compare the strategies head-to-head on the identical design problem.

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
├─ make_shortlist_case.py              # build ONE per-strategy shortlist xlsx (ID + bright + diverse)
├─ scale_to_96.sh                      # scale the consolidated efforts to 96 trials (resumable)
├─ gen_shortlists.sh                   # watcher: emit each shortlist as its run finishes
├─ visualize_campaign.ipynb            # cross-strategy analysis + all figures
├─ figures/                            # saved PNGs from the notebook
├─ shortlists/                         # final wet-lab shortlists: one xlsx per (strategy × target)
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

## The four strategies

Each trial starts from the EGFP sequence and does **any-order masking** (a fresh random permutation
of the window positions each iteration). At each visited position we mask it, read the ESM-2
conditional logits, and keep the top-k allowed residues; the strategies differ in what happens next.

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
- the gibbs CSV is **target-free** (no `target_*` / `peak_err` columns); reconstruct per-target error
  from its `pred_ex/pred_em` as needed.

## Analysis

`visualize_campaign.ipynb` is the cross-strategy view (all figures land in `figures/`):

- **Overview & per-pair error** — designs vs scaffold vs target in `(ex, em)` space; per-pair
  surrogate design error and convergence vs iteration round.
- **In-distribution / OOD check** — ESM-2 max-pool embeddings of each strategy's designs projected
  onto the four-scaffold GFP-DMS clouds (PCA), shown as KDE density contours alongside the 758 curated
  true FPs, with a nearest-neighbour distance OOD metric. The reference is **40,000 DMS sequences
  (10k per scaffold: avGFP, amacGFP, cgreGFP, ppluGFP)**; a design is "in-distribution" if its NN
  distance ≤ the DMS 99th percentile (`p99 ≈ 30.2`). The DMS-guide designs stay largely
  in-distribution; the pure-guided and gibbs designs extrapolate much further out.
- **Top-10 per strategy (EBFP, mOrange)** — two bar plots per target (ex and em) that **load the
  finalized `shortlists/*.xlsx`** (rather than re-selecting designs), so the figures match the wet-lab
  hand-off exactly. Each bar = mean ± sd of that strategy's shortlisted top-10, with the individual
  designs overlaid, a green bar for the EGFP scaffold, and a dashed line at the target. mOrange shows
  all four strategies; EBFP shows only **DMS guide - bright** (the one shortlist generated for it).
- **Strategy comparison** — mean/best surrogate error per strategy on the two shared pairs, plus a
  uniform `% predicted-bright` scored by the same cnn-max-d2 classifier across every strategy.

Regenerate everything with:

```bash
conda run -n esm2-fp-design jupyter nbconvert --to notebook --execute --inplace visualize_campaign.ipynb
```

## Wet-lab shortlist

The final hand-off lives in `shortlists/` as **one xlsx per (strategy × target)** (the consolidated
efforts were scaled to **96 trials** each via `scale_to_96.sh`):

- `shortlist_mOrange_gibbs.xlsx`, `shortlist_mOrange_spectra-guide.xlsx`,
  `shortlist_mOrange_constrained-spectra-guide.xlsx`, `shortlist_mOrange_DMS-guide.xlsx`
- `shortlist_EBFP_DMS-guide.xlsx`

Each file starts with the two references (**EGFP** scaffold + the target, with their true dataset
ex/em) followed by the **top-10 diverse** designs (greedy, ≥ 5 residues apart, ranked by surrogate
peak error). The **DMS-guide** files restrict the pool to designs that are both **in-distribution**
(NN-distance to the 40k reference ≤ p99 ≈ 30.2) **and predicted bright**; the other strategies take
the plain closest-10. Every design row is annotated with `is_id`, `is_bright`, `bright_logit`,
predicted ex/em, and an E. coli codon-optimized DNA sequence. Build one file (or rebuild all):

```bash
conda run -n esm2-fp-design python make_shortlist_case.py mOrange_DMS   # -> shortlists/shortlist_mOrange_DMS-guide.xlsx
# cases: mOrange_gibbs  mOrange_spectra  mOrange_constr  mOrange_DMS  EBFP_DMS
```

`gen_shortlists.sh` watches a `scale_to_96.sh` run and emits each file automatically as its run
completes. The barplots in `visualize_campaign.ipynb` load these files directly.

## Reproduce the shared assets

```bash
cd design-campaign-EGFP
python make_pairs.py        # regenerate pairs/campaign_pairs_egfp.csv (EBFP, mOrange)
```

The edit window `design_windows_egfp_tierB.json` is derived one level up in the conventional campaign
tooling and copied into each effort so every strategy edits the identical positions.
