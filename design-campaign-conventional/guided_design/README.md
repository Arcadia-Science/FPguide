# Guided design — ESM-2 guided generation

Guide 24 structure-backed fluorescent-protein **scaffolds** toward chosen **targets** (70–90 %
sequence identity, both Stokes shift < 80 nm) by editing only each scaffold's chromophore
edit-window. A single **surrogate** model both *guides* the search and *predicts* the resulting
`(ex, em)`.

- **Surrogate (guide + predictor):** ESM-2 `cnn-max-d1` trained on **all** FP data (no train/test
  split) → `../models/surrogate_cnn-max-d1_alldata.pt` (train MAE ≈ 5.2 nm on `(ex, em)`).

> **No oracle.** This campaign trains the surrogate on *all* data, so there is no held-out,
> independent evaluator to serve as a judge (and the popular FPs used here are in every model's
> training data). The recorded `pred_ex/pred_em` are therefore the **surrogate's own** predictions —
> useful as the design objective, but *in-sample and optimistic*, **not** validation. The real judge
> is experiment.

This folder is one *design algorithm* that consumes the campaign's shared assets. Curation,
surrogate training, and window derivation live one level up (`../`):

```
design-campaign-conventional/
├─ models/surrogate_cnn-max-d1_alldata.pt   # shared: all-data surrogate
├─ pairs/campaign_pairs_24.csv              # shared: the 24 scaffold→target pairs
├─ design_windows_24.json                   # shared: 5 Å structure-based edit windows
├─ build_design_windows.py                  # setup: derive the windows
├─ select_campaign_pairs.py                 # setup: choose the 24 pairs
├─ train_surrogate_alldata.py               # setup: train the all-data surrogate
└─ guided_design/                           # ← this effort
   ├─ design_campaign.py                     # the guided-generation driver
   ├─ run_campaign.sh                        # detachable job wrapper
   ├─ designs/                               # one CSV per pair (trajectories)
   └─ logs/                                  # run logs
```

## Method

For each pair we run **6 independent trials** starting from the scaffold sequence. All trials of a
pair share the **identical edit window** (from `design_windows_24.json`): chromophore pos1 & pos2 +
the 5 Å heavy-atom pocket read off the scaffold's experimental structure; **pos2 is restricted to
aromatics {Y,W,H,F}**; the chromophore Gly (pos3) and the maturation-catalytic Arg/Glu are **fixed**.

Each trial does **any-order masking**: a fresh random permutation of the window positions is drawn
every iteration, so the 6 trials explore different edit orders. At each visited position:

1. mask it and read the **ESM-2** conditional logits → keep the **top-k** allowed residues;
2. splice each candidate and predict `(ex, em)` with the **surrogate**;
3. score `s = z(logp_ESM) − λ_ex·z(|Δex|) − λ_em·z(|Δem|)` and **sample** at temperature `T`;
4. accept the sampled residue and move on.

**Settings:** `trials=6`, `iters=3`, `T=10`, `k=10`, `λ_ex=λ_em=20`.

**Acceleration:** one pair at a time, but its 6 trials advance the *same* window slot together in a
single batched GPU forward (fp16 autocast, sub-batched ESM-2 forwards). The surrogate predicts
`(ex, em)` for the sequence each round (recorded as `pred_ex/pred_em`); ESM-2 pseudo-perplexity is
logged as a naturalness diagnostic.

## Run

```bash
cd design-campaign-conventional/guided_design

# time one pair and project the full 24-pair run, then exit (no writes)
python design_campaign.py --probe

# full run (ppl every round by default); ~85 min on one GPU; resumable (skips finished pairs)
setsid bash run_campaign.sh < /dev/null > /dev/null 2>&1 &

# faster variant: ppl only at scaffold + final round (~60–65 min)
PPL=endpoints bash run_campaign.sh

# monitor
tail -f "$(cat .last_log)"
pgrep -af design_campaign.py         # empty = finished/stopped
```

Useful flags: `--trials --iters --temp --k --lam-ex --lam-em --pairs-limit --ppl {all,endpoints}`.

`python design_campaign.py --rescore` re-fills `pred_ex/pred_em/peak_err` of the existing CSVs with
the surrogate (no re-running the design loop) — used after switching off the oracle.

## Output

One CSV per pair, `designs/design_<scaffold>-<target>.csv`, with one row per **(trial, round)**
(round 0 = scaffold; rounds 1..3 = after each iteration → 6 × 4 = 24 rows + header). Columns:

`pair, scaffold_name, scaffold_idx, scaffold_pdb, target_name, target_idx, selection,
scaffold_ex, scaffold_em, target_ex, target_em, seq_id_scaf_target, trial, round, n_editable,
temp, k, lam_ex, lam_em, pred_ex, pred_em, peak_err, ppl, ident_to_scaffold, designed_seq,
scaffold_seq, target_seq`

`peak_err = ½(|pred_ex − target_ex| + |pred_em − target_em|)` with `pred_ex/pred_em` from the
**surrogate**. Round 0 is the surrogate's prediction for the unmodified scaffold.

## Results (24 pairs / 144 trajectories; surrogate-predicted `peak_err`)

Best-of-6 (final-round surrogate `peak_err`) improved on the scaffold for **24 / 24** pairs;
**20 / 24** reached a best design predicted within **10 nm** of the target. These are the surrogate's
*own* predictions of designs it optimized, so they are **in-sample/optimistic**, not validation.
Selected outcomes (scaffold err → best of 6, nm):

| pair | scaffold err | best of 6 |
|---|---|---|
| mCerulean2.D3 → ShG24 | 43.8 | **0.0** |
| mCherry → DsRed | 34.3 | **0.1** |
| EGFP → TagYFP | 17.7 | **0.2** |
| GFPxm162 → Aquamarine | 56.0 | **0.2** |
| mKate2 → Crimson | 4.2 | **0.2** |
| GFPxm191uv → Aquamarine | 45.1 | **0.5** |
| W1C → GFPxm191uv | 37.9 | **1.1** |
| td-RFP639 → mRubyFT | 174.7 | **3.0** |
| mTagBFP2 → Crimson | 186.5 | **6.5** |
| mEosFP → mc1 (hardest) | 35.0 | 24.3 |

The four hardest pairs (best 17–24 nm) are the `mEos*/rsFusionRed3/E2-Red` cases where the target sits
far from what the window can reach. Regenerate this summary any time from the CSVs; see
`../visualize_campaign.ipynb` for the full interactive view.

## Next step — ESM-2 Gibbs-sampling design (optionally surrogate-scored)

The current driver is a *greedy, one-pass* guided decoder: each window position is visited once per
iteration and resampled from the top-k masked-LM logits. A natural, more principled successor is
**Gibbs sampling** over the editable positions, treating ESM-2 as the sequence prior.

- **Pure ESM-2 Gibbs (masked-LM Gibbs, à la Wang & Cho, "BERT has a Mouth", 2019).** Random-scan
  Gibbs: repeatedly pick a random window position `i`, mask it, and **sample** from the full
  conditional `p(x_i | x_{−i})` given by ESM-2 (not top-k, not argmax). Run many sweeps with
  burn-in + thinning to draw well-mixed samples from the joint MLM-implied distribution. This
  yields more **diverse** and more **native-like** sequences than greedy top-k, and makes the
  "any-order masking" idea a proper stationary-distribution sampler rather than a single pass.

- **Surrogate-scored Gibbs (energy-based / Metropolis-adjusted).** Fold the property objective into
  the sampler so it targets `π(x) ∝ p_ESM(x) · exp(−E(x)/T)` with the surrogate energy
  `E(x) = λ_ex|Δex| + λ_em|Δem|`. Two clean options:
  1. **Locally-biased conditional:** at each Gibbs step, reweight the ESM-2 conditional over
     candidate residues by `exp(−E/T)` (evaluate the surrogate on the ~20 single-residue variants —
     the same batched call we already do) and sample from the tempered product. Cheap; reuses all
     current batching.
  2. **Metropolis–Hastings:** propose a residue from the ESM-2 conditional, **accept** with
     `min(1, exp(−ΔE/T))`. Gives an asymptotically correct sampler for `π`, with a tunable
     prior/objective trade-off via `T` and `λ`.

- **Why it's worth trying:** better exploration of the fitness landscape (escapes the greedy
  collapse), controllable diversity vs. objective trade-off, and a distribution we can actually
  sample many candidates from per pair. It reuses everything here unchanged — the models, the 5 Å
  windows, the aromatic/fixed constraints, the batched forwards, and the trajectory CSV schema — so
  it can drop in as `guided_design/design_campaign_gibbs.py` alongside the current greedy driver for
  a head-to-head comparison (surrogate `peak_err`, ppl, diversity, and identity-to-scaffold).
