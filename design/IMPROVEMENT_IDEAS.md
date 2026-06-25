# Surrogate prediction — improvement ideas (backlog, not yet implemented)

Recommendations to improve FP spectrum prediction, ranked by impact-per-effort. Grounded in observations:
mean-pool is worst (max/min/std/CNN better); models overfit fast (train ≪ val); color is set by a few
chromophore residues; ~45% below baseline on a *random* (local) split but barely above baseline on a
*cluster* (OOD) split — i.e. the ceiling is data/representation, especially out-of-distribution.

## 1. Scalar targets + larger dataset  *(highest leverage, low effort)*
- Predict **`ex_max` / `em_max`** (± FWHM) instead of the 1002-dim curve — far fewer outputs ⇒ less
  overfitting, and it's the quantity that matters. Curves are recoverable from peak+width (or PCA 8–16 coeffs).
- This unlocks **~2× the data**: FPbase has **~990 proteins with sequences + peak maxima** vs only **453 with
  full ex/em curves**. The current pipeline discards half the data by requiring curves. More families + easier
  target directly attacks the data bottleneck that capped the OOD result.

## 2. Expose the chromophore directly  *(attacks the root cause)*
- Color comes from a few chromophore-environment residues; pooling dilutes them. **Align** sequences (existing
  `PairwiseAligner`, or an MSA) to a common coordinate, locate the **chromophore tripeptide (X-Y-G, ~pos 65–67
  in avGFP)**, and feed embeddings of the chromophore + a surrounding window — hand the model the decisive
  positions instead of hoping max-pool finds them.

## 3. Richer / multi-layer representation  *(cheap)*
- Concatenate **`[mean, max, min, std]`** as the readout (each helped individually).
- Use a **learned scalar-mix over several ESM-2 layers**, not just layer 33 — mid layers carry more local
  structural signal relevant to the chromophore pocket.

## 4. Fine-tune the representation  *(higher effort, lifts the ceiling)*
- Frozen ESM features are generic — that's the OOD ceiling. **LoRA / last-1–2-layer fine-tuning** on the
  spectrum task makes the representation task-aware ("Approach 2"). Use LoRA + strong regularization + early
  stopping given the small dataset.

## 5. Regularization & ensembling  *(incremental robustness)*
- Push **dropout / weight-decay** harder; keep early stopping (models overfit by ~epoch 10).
- **Ensemble** across poolings/seeds — averages variance, gives uncertainty (useful for trusting designs).
- **Multi-task** (ex_max, em_max, brightness, QY jointly) regularizes the shared trunk.

## By goal
- **Local re-design (random split):** already useful (em-MAE ~22 nm). Gains from #1 (data/scalars) + #5 (ensembling).
- **OOD design (cluster split):** ceiling is data/representation — only #1 (more families), #2 (chromophore
  features), #4 (fine-tuning) will move it.
