# Attempt — ~80% identity, distinct ex_max & em_max (surrogate-train only, λ-ramp)

**Date:** 2026-06-29
**Artifacts:** `design_CyOFP1-mCardinal.csv`, `design_mTangerine-E2-Crimson.csv` (per-round traces, round 0 = scaffold)

## Design process
- **Selection pool:** `surrogate_role==train` only (oracle role **not** constrained).
- **Pairs (`find_peakshift_pairs`):** sequence identity in [0.75, 0.85] (closest to 0.80) with **both** |Δex_max| ≥ 15 nm and |Δem_max| ≥ 15 nm; scaffold = bluer, target = redder.
- **Generator / schedule:** identical to the high-sim attempt — 40-residue window, naturalness + λ·cosine(surrogate, target), `cnn-max` surrogate & oracle, start **λ=20, +3/round, 10 rounds**.

## Result — best oracle cosine across rounds
| pair | seq id | Δex / Δem | scaffold→best cos | best round (λ) | scaffold/target oracle role |
|---|---|---|---|---|---|
| CyOFP1 → mCardinal | 80% | 107 / 70 nm | 0.372 → 0.453 | 4 (λ29) | train / train |
| mTangerine → E2-Crimson | 80% | 43 / 61 nm | 0.683 → 0.557 (**no gain**) | 5 (λ32) | test / test |

## Why superseded
mTangerine→E2-Crimson are both **oracle-test**; the surrogate saturated near 0.84 while the oracle fell below the scaffold — surrogate/oracle disagreement on out-of-distribution endpoints + novel designs. Replaced by a **train/train-only** selection so both models trained on the scaffold and target.
