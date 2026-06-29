# Attempt — moderate sequence & spectrum similarity (ID, λ-ramp)

**Date:** 2026-06-29
**Artifacts:** `design_vsfGFP-0-SHardonnay.csv`, `design_asulCP-Katushka.csv` (per-round traces, round 0 = scaffold)

## Design process
- **Group:** in-sample only (`Srole==train`, surrogate-seen targets).
- **Pair selection (`find_moderate_pairs`):** pairs **moderate in BOTH** dimensions — sequence identity in **[0.40, 0.70]** and scaffold↔target **per-half spectral cosine** in **[0.40, 0.85]** (4-mer + vectorized spectral-cosine prefilter, exact alignment identity to confirm); scaffold = bluer, target = redder.
- **Generator:** ESM-2 masked-LM windowed refinement (40-residue window), one sweep per round, continuing from the previous round; score = `naturalness + λ·cosine(surrogate spectrum, target)`. Surrogate & oracle = `cnn-max`.
- **Schedule:** start **λ=20, +3/round, 10 rounds** (λ = 20, 23, …, 47).

## Pairs & result (oracle cosine vs target)
| example | seq id | spectral cos | em (scaf→tgt) | scaffold cos (r0) | best (round) | final (r10) |
|---|---|---|---|---|---|---|
| vsfGFP-0 → SHardonnay | 63% | 0.62 | 510→530 nm | 0.66 | 0.70 (r4) | 0.68 |
| asulCP → Katushka | 64% | 0.63 | 595→635 nm | 0.78 | **0.89 (r6, λ35)** | 0.83 |

## Takeaway
Far-red asulCP→Katushka improves to cos ≈ 0.89; green vsfGFP-0→SHardonnay stays ~0.68. As elsewhere, gains saturate by round ~5–6 and the higher-λ tail does not help.

**Superseded by:** the `~80%-identity, distinct ex_max & em_max` selection (see newer attempt / current Section 7).
