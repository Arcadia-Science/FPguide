# Attempt — sequence-similar pairs, iterative windowed design with λ ramp-up

**Date:** 2026-06-26 (latest run)
**Artifacts:** `designs.csv` (final design per group × start-λ), `designs_history.csv` (per-round trace, 44 rows)

## Design process
- **Pair selection:** within each group, auto-pick the **most 4-mer-similar** scaffold→target pair with a real emission gap (`EM_GAP_MIN=20` nm) but **not a near-duplicate** (`IDENT_MAX=0.98`); scaffold = bluer member, target = redder. This guarantees a homolog pair where the spectral change is achievable.
- **Groups (ID vs OOD by whether the surrogate saw the target spectrum):**
  - **in-sample** (`Srole==train`): scaffold **P4** (em **448 nm, blue**) → target **EGFP** (em **507 nm, green**), 98% sequence identity.
  - **OOD** (`Srole==test`): scaffold **SCFP3A** (em **474 nm, cyan**) → target **DarkVenus** (em **525 nm, yellow-green**), 96% sequence identity.
- **Generator:** ESM-2 masked-LM windowed refinement, 40-residue window centered on the scaffold; one sweep per round, continuing from the previous round's design.
- **Guidance:** `naturalness + λ·cosine(surrogate spectrum, target)`; surrogate = `cnn-max`, oracle = `cnn-max` (independent).
- **Iteration schedule (10 rounds):**
  - rounds **1–5**: constant λ at the start value (**20** or **30**).
  - rounds **6–10**: **λ ramp-up, +2 per round** (start-λ20 → 22,24,26,28,30; start-λ30 → 32,34,36,38,40).
  - round 0 = the scaffold itself.

## Result (oracle cosine vs target)
| group | start-λ | scaffold cos (r0) | best (round) | final (r10) | end ppl |
|---|---|---|---|---|---|
| in-sample | 20 | 0.738 | 0.784 (r3) | 0.750 | 14.8 |
| in-sample | 30 | 0.738 | **0.813 (r5)** | 0.792 | 15.9 |
| OOD | 20 | 0.227 | 0.346 (r5) | 0.341 | 15.2 |
| OOD | 30 | 0.227 | **0.388 (r9, λ38)** | 0.349 | 15.5 |

## Takeaway
- **in-sample** reaches cos ≈ 0.78–0.81 and tracks EGFP's green emission peak; gains plateau by round ~5.
- **OOD** stays stuck at cos ≈ 0.33–0.39 — the surrogate never saw the DarkVenus spectrum, so its guidance is the bottleneck.
- The **λ ramp (rounds 6–10) did not help**: cosine oscillates/declines slightly and pseudo-perplexity drifts; pushing λ higher trades naturalness without breaking the OOD plateau.

## Reproduce
`design/guided_design_approach1.ipynb` → Section 4 (params: `N_ROUNDS=5`, `EXTRA_ROUNDS=5`, `LAM_STEP=2`, `LAMBDAS=[20,30]`, `WINDOW=40`, `EM_GAP_MIN=20`, `IDENT_MAX=0.98`); set `REGENERATE=True`.
