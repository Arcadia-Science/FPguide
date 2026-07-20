# Gibbs sampling — pure ESM-2 masked-LM design

Guide 24 structure-backed fluorescent-protein **scaffolds** toward chosen **targets** (70–90 %
sequence identity, both Stokes shift < 80 nm) by editing only each scaffold's chromophore
edit-window. This is the ESM-2-only sibling of [`../guided_design`](../guided_design): the design
procedure is **identical**, but the per-position choice is scored by **ESM-2's masked-LM likelihood
alone** — there is **no surrogate peak-error (MAE) term** biasing the search toward the target.

- **Generator:** ESM-2 masked-LM. At each visited window position we mask it, read ESM-2's
  conditional `p(x_i | x_{−i})`, and sample from the top-k allowed residues. Sweeping random-scan
  positions and resampling each from its ESM-2 conditional is **masked-LM Gibbs sampling** over the
  editable window (à la Wang & Cho, "BERT has a Mouth", 2019).
- **Surrogate (diagnostic only):** ESM-2 `cnn-max-d1` trained on **all** FP data →
  `../models/surrogate_cnn-max-d1_alldata.pt` (train MAE ≈ 5.2 nm on `(ex, em)`). It is loaded only
  to record where each ESM-2-driven sequence lands in `(ex, em)` space; it **never** steers the
  search. There is **no oracle** for this task (the real judge is experiment).

This folder is one *design algorithm* that consumes the campaign's shared assets. Curation,
surrogate training, and window derivation live one level up (`../`):

```
design-campaign-conventional/
├─ models/surrogate_cnn-max-d1_alldata.pt   # shared: all-data surrogate (diagnostic here)
├─ pairs/campaign_pairs_24.csv              # shared: the 24 scaffold→target pairs
├─ design_windows_24.json                   # shared: 5 Å structure-based edit windows
├─ guided_design/                           # sibling: surrogate-guided decoder
└─ gibbs-sampling/                          # ← this effort (ESM-2 only)
   ├─ design_campaign.py                     # the ESM-2 masked-LM Gibbs driver
   ├─ run_campaign.sh                        # detachable job wrapper
   ├─ designs/                               # one CSV per pair (trajectories)
   └─ logs/                                  # run logs
```

## Method

For each pair we run **6 independent trials** starting from the scaffold sequence. All trials of a
pair share the **identical edit window** (from `../design_windows_24.json`): chromophore pos1 & pos2 +
the 5 Å heavy-atom pocket read off the scaffold's experimental structure; **pos2 is restricted to
aromatics {Y,W,H,F}**; the chromophore Gly (pos3) and the maturation-catalytic Arg/Glu are **fixed**.

Each trial does **any-order masking**: a fresh random permutation of the window positions is drawn
every iteration, so the 6 trials explore different edit orders. At each visited position:

1. mask it and read the **ESM-2** conditional logits → keep the **top-k** allowed residues;
2. **sample** the residue **directly from that conditional** — `softmax(logp_ESM / T)` with the
   **raw** log-probs (not z-scored) at `T=1`, i.e. a true Gibbs draw from `p(x_i | x_{−i})`
   restricted to the top-k;
3. accept the sampled residue and move on.

Compared with `../guided_design`, step 2 drops the surrogate term
`− λ_ex·z(|Δex|) − λ_em·z(|Δem|)`: no candidate is scored by its predicted distance to the target.

**Settings:** `trials=6`, `iters=3`, `T=1`, `k=10`.

### Why `T=1` and raw log-probs (difference from `../guided_design`)

The guided driver scores `s = z(logp_ESM) − λ_ex·z(|Δex|) − λ_em·z(|Δem|)` with **all terms
z-scored** (std ≈ 1 across the k candidates), `λ=20`, `T=10`, then samples `softmax(s / T)`. The
`λ=20` error terms give a spread of ~±3–4 after `/T`, so they **dominate** and make the choice
sharply peaked toward the surrogate target; the z-scored ESM term contributes only ~±0.15 after
`/T`, i.e. it is **near-uniform** and acts as a weak tie-break.

Here the surrogate terms are gone. If we had kept guided's **z-scored** ESM term at `T=10`, the only
surviving term would be that near-uniform one, so we would be sampling ≈ uniformly over the top-k —
not actually following ESM-2. To fix this we use the **raw** log-probs at `T=1`, so `softmax(logp)`
reconstructs ESM-2's own conditional `p(x_i | x_{−i})` (over the top-k). That is the intended
masked-LM Gibbs behavior.

**Acceleration:** one pair at a time, but its 6 trials advance the *same* window slot together in a
single batched GPU forward (fp16 autocast, sub-batched ESM-2 / surrogate / pseudo-perplexity). The
surrogate predicts `(ex, em)` once per round as a diagnostic; ESM-2 pseudo-perplexity is logged as a
naturalness diagnostic. Because the design loop makes no in-loop surrogate calls, iterations are
faster than the guided driver's.

## Run

```bash
cd design-campaign-conventional/gibbs-sampling

# time one pair and project the full 24-pair run, then exit (no writes)
python design_campaign.py --probe

# full run (ppl every round by default); resumable at TRIAL granularity
setsid bash run_campaign.sh < /dev/null > /dev/null 2>&1 &

# faster variant: ppl only at scaffold + final round
PPL=endpoints bash run_campaign.sh

# EXPAND every pair from 6 to 24 designs later WITHOUT recomputing the first 6
bash run_campaign.sh --trials 24

# fast: design with ppl only at endpoints, then backfill the intermediate-round ppl later
PPL=endpoints bash run_campaign.sh --trials 24
python design_campaign.py --backfill-ppl        # fills empty ppl cells (deduped + batched)

# monitor
tail -f "$(cat .last_log)"
pgrep -af design_campaign.py         # empty = finished/stopped
```

Useful flags: `--trials --iters --temp --k --pairs-limit --ppl {all,endpoints}`.

## Output

One CSV per pair, `designs/design_<scaffold>-<target>.csv`, with one row per **(trial, round)**
(round 0 = scaffold; rounds 1..3 = after each iteration → 6 × 4 = 24 rows + header). The schema is
kept **identical to `../guided_design`** for head-to-head comparison. Columns:

`pair, scaffold_name, scaffold_idx, scaffold_pdb, target_name, target_idx, selection,
scaffold_ex, scaffold_em, target_ex, target_em, seq_id_scaf_target, trial, round, n_editable,
temp, k, lam_ex, lam_em, pred_ex, pred_em, peak_err, ppl, ident_to_scaffold, designed_seq,
scaffold_seq, target_seq`

- `lam_ex` / `lam_em` are **left blank** here — no surrogate MAE term is used.
- `pred_ex, pred_em` are the **surrogate's** `(ex, em)` for the current sequence, and
  `peak_err = ½(|pred_ex − target_ex| + |pred_em − target_em|)` — a **diagnostic** measuring where
  the ESM-2-driven sequence lands, **not** an objective the search optimizes.

### Resumable / expandable at trial granularity

On each run the driver counts how many trials a pair CSV already holds and computes **only the
missing trials** `[have, --trials)`, **appending** them. So:

- Interrupting and re-running never recomputes finished trials.
- Re-running with a larger `--trials` **expands** a pair (e.g. `6 → 24`) and appends trials `6..23`
  without touching the first 6.

Each trial's RNGs (both the visiting order and the residue sampling) are seeded per trial from
`(SEED, scaffold_idx, trial)`, so **trial k is bit-for-bit identical** regardless of when it is
drawn or how many other trials are drawn alongside it.
