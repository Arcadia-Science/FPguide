# Attempt — high-similarity, spectrally-distinct pairs (surrogate-train only, λ-ramp)

**Date:** 2026-06-29
**Artifacts:** `design_CFP-mEGFP.csv`, `design_EBFP-avGFP.csv`, `design_P4-EGFP.csv` (per-round traces, round 0 = scaffold)

## Design process
- **Selection pool:** `surrogate_role==train` only (oracle role **not** constrained).
- **Pairs (`find_pairs`):** top 4-mer-similar in-pool pairs with emission gap 20–80 nm, sequence identity ≤ 0.98, no sequence reused; scaffold = bluer, target = redder. → very high identity (~98%) with distinct spectra.
- **Generator:** ESM-2 masked-LM windowed refinement (40-residue central window), one sweep/round, continuing from prev round; score = naturalness + λ·cosine(surrogate spectrum, target). Surrogate & oracle = `cnn-max`.
- **Schedule:** start **λ=20, +3/round, 10 rounds** (λ = 20, 23, …, 47).

## Result — best oracle cosine across rounds
| pair | seq id | scaffold→best cos | best round (λ) | scaffold/target oracle role |
|---|---|---|---|---|
| CFP → mEGFP | 98% | 0.792 → **0.921** | 10 (λ47) | test / test |
| P4 → EGFP | 98% | 0.738 → 0.813 | 5 (λ32) | val / train |
| EBFP → avGFP | 98% | 0.827 → 0.749 (**no gain**) | 2 (λ23) | test / train |

## Why superseded
Endpoints were filtered only on the **surrogate** split, so several were **oracle val/test** (e.g. EBFP, CFP, mEGFP). The surrogate-guided edits maximized surrogate cosine (→0.99) while the oracle disagreed on the novel designs, and the EBFP scaffold's high oracle baseline (0.83) is itself suspect because the oracle never trained on EBFP. Replaced by a **train/train-only** selection (both surrogate AND oracle trained on scaffold and target) so the evaluation baseline and guidance are both in-distribution.
