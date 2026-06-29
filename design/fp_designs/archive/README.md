# Design attempts — archive

Snapshots of past property-guided design runs from `design/guided_design_approach1.ipynb`.
Each folder is one attempt with a `README.md` describing its design process (scaffold/target
fluorescence, guidance objective, λ schedule, iteration scheme) and results.

The **active** working files live one level up in `fp_designs/` (`designs.csv`,
`designs_history.csv`) and are overwritten every time the notebook regenerates — copy a finished
run into a new dated folder here to keep it.

| folder | date | scaffold → target (em) | groups | λ schedule | iterations | headline result |
|---|---|---|---|---|---|---|
| `2026-06-26_multitarget_green-scaffold_lambda5-15` | 2026-06-26 | fixed green scaffold (~510) → many targets (440–580) | ID & OOD | {5, 15} | 1 sweep | green targets cos ≈ 0.82–0.97; blue/red fail (≈0.16–0.26) |
| `2026-06-26_seqsim-pairs_lambda-ramp` | 2026-06-26 | P4 (448)→EGFP (507); SCFP3A (474)→DarkVenus (525) | ID & OOD | start {20,30}, then +2/round | 10 rounds (5 const + 5 ramp) | ID cos ≈ 0.79–0.81; OOD stuck ≈ 0.33–0.39; ramp didn't help |
| `2026-06-29_moderate-seqspec_lambda-ramp` | 2026-06-29 | vsfGFP-0 (510)→SHardonnay (530); asulCP (595)→Katushka (635) | ID | start 20, +3/round | 10 rounds | moderate seq (63–64%) + spectrum (cos ~0.62); asulCP→Katushka cos→0.89 |
| `2026-06-29_high-sim_surrogate-train_lambda-ramp` | 2026-06-29 | CFP→mEGFP; P4→EGFP; EBFP→avGFP (~98% id) | surrogate-train (mixed oracle role) | start 20, +3/round | 10 rounds | CFP→mEGFP 0.79→0.92; EBFP→avGFP no gain (oracle never saw EBFP) |
| `2026-06-29_peakshift-80pct_surrogate-train_lambda-ramp` | 2026-06-29 | CyOFP1→mCardinal; mTangerine→E2-Crimson (~80% id) | surrogate-train (mixed oracle role) | start 20, +3/round | 10 rounds | mixed oracle roles → surrogate/oracle disagreement; superseded by train/train-only |

## Naming convention
`YYYY-MM-DD_<pair-strategy>_<lambda-strategy>` — e.g. `2026-06-26_seqsim-pairs_lambda-ramp`.
