# in-silico-test — peak-conditioned guided design

The project's in-silico design experiment. A surrogate and an **independent** oracle are trained on
the curated peak dataset; the search edits a structure-defined chromophore pocket toward a target
`(ex_max, em_max)` under the surrogate's guidance; every design is then scored by the oracle, which
the search never consults. An unguided pocket-resampling arm runs the same search with the target
switched off, so the surrogate's contribution can be separated from what resampling alone achieves.

## The train/val/test split

Surrogate and oracle are split **nested**, oracle first, so the surrogate can never have seen
anything the oracle is evaluated on:

| | split | sizes (N=758) |
|---|---|---|
| **oracle** | 80/10/10 of the *entire* dataset | train 606 · val 76 · test 76 |
| **surrogate** | 70/15/15 carved *only from oracle-train* | train 424 · val 91 · test 91 · excluded 152 |

Nesting buys strict isolation: the oracle's held-out val and test rows are disjoint from everything
the surrogate trained on, so an oracle score on a design is not a verdict from a model that shared
the surrogate's training signal. It costs ~20% of the surrogate's training data — the 152 rows that
land in oracle val/test are unusable for the surrogate, which is the price of the guarantee.

> An earlier scheme split the two models *coordinated* rather than nested, with reciprocal
> invariants (`S_test ⊆ O_train`, `O_test ⊆ S_train`) so each could be evaluated on data the other
> trained on. `dataset_pipeline/make_dual_split.py` still writes it, but nothing reads it: every
> number in this folder comes from the nested split above, built by `0_data_split/make_dual_split.py`.

## Layout

Code is grouped by pipeline stage; shared inputs, artifacts and the notebooks stay at the root,
since they're consumed across stages.

```
design_common.py               every path + the dataset/split/hits loaders — START HERE
0_data_split/                  make_dual_split.py                    -> data/dual_splits.csv
1_surrogate_oracle_training/   sweep, 3-fold CV, final refit         -> trained_models/
2_design_task_specification/   validate_structures, curate_pairs_task2,
                               build_windows              -> pairs_task2/, design_windows.json
3.1_design_run_guided/         design_knownstruct_guided.py          -> peak_designs/  (guided)
                               run_task2_guided.py                   — the stage entry point
3.2_design_run_gibbs/          design_knownstruct_gibbs.py           -> peak_designs/  (unguided null)
                               run_task2_gibbs.py                    — the stage entry point
archive/                       the ORIGINAL task set, its three arms and the cross-set
                               comparison — see archive/README.md
lib/                           vendored modules — copies, don't edit here
data/  structures/             inputs and the RCSB cache
figures/                       this folder's exported PNG/SVG/HTML figures
sweep_results.ipynb  figures.ipynb
```

**That chain runs task set 2**, in which each scaffold is paired with a *random* qualifying
target. The earlier **task set 1** — each scaffold paired with its *most spectrally distant*
qualifying target — and its three design arms are archived. Four scripts were still needed by the
live chain and so moved up into it rather than into `archive/`: `validate_structures.py` and
`build_windows.py` into stage 2, and the two design engines into 3.1 and 3.2, whose
`run_task2_*.py` are thin runners over them. Task 1's *outputs* were not archived — only code was —
so the archived comparison script and notebooks still run against them; nothing in the live chain
reads them. Full account, and task 1's own results, in `archive/README.md`.

> **Stage numbers are per era.** `archive/` keeps task 1's own numbering (2, 3.1, 3.2, 3.3), which
> does not line up with the live stages: live `3.1` is the ESM-2 guided arm, archived `3.1` is task
> 1's MSA arm. Anything archived is cited with its `archive/` prefix throughout.

Stage scripts live one level down, so each starts with a short bootstrap putting the root, `lib/`
on `sys.path`. Run them from the root (`python 2_design_task_specification/curate_pairs_task2.py`).
Each stage folder also keeps its own run logs.

## Standalone by design

Every piece of *code* the pipeline runs lives here; nothing is imported from `../fpdesign` or
`../msa_conservation` at runtime. Vendored as real local files:

| file | what it is |
|---|---|
| `lib/pockets.py` | window geometry (5 Å pocket + H-bond partners) from an RCSB structure |
| `lib/peak_models.py` | model architectures + checkpoint save/load |
| `lib/prostt5_embed.py` | ProstT5 residue embedding, for oracle scoring |
| `lib/sweep_peak_oracle_base.py` | the shared architecture-sweep implementation |
| `structure_hits.csv` | which dataset entries have a ≥97%-identity PDB entry |

**Two deliberate exceptions**, both large read-only caches that are split-independent inputs to
every experiment in the repo — duplicating them per folder would be waste, so both are symlinks:

- `data/` → `dataset_pipeline/data/peak/curated/`: the shared curated dataset plus ~2GB of ESM-2 /
  ProstT5 residue-embedding caches. `data/dual_splits.csv` (the nested split above) *is* a real
  local file, written here rather than symlinked.
- `structures/` → the repo-level [`../structures/`](../structures): the RCSB PDBx cache, ~175MB.
  Self-populating — `pockets.py` fetches a miss from RCSB, which now lands in the shared cache.
  Every path inside this folder is unchanged, so `structures/experimental/` still resolves.

Both are content-addressed downloads of immutable public data, so sharing them across experiments
cannot leak anything between splits.

Model weights come from the HF cache (ProstT5, ESM-2) — a package-level dependency, not a
repo-folder one.

## Pipeline

Run in order. Each step's output is committed to disk, and every long step is resumable.

```bash
S0=0_data_split; S1=1_surrogate_oracle_training
S2=2_design_task_specification; S31=3.1_design_run_guided; S32=3.2_design_run_gibbs

python $S0/make_dual_split.py                     # -> data/dual_splits.csv
python $S1/sweep_peak_oracle.py --role both --seeds 0   # 48 configs x 2 roles -> trained_models/
python $S1/cv_all_surrogate.py                    # 3-fold CV, all 48 surrogate configs
python $S1/train_final_surrogate.py               # refit the CV winner on train+val

# the task set -- a RANDOM qualifying target per scaffold (36 S-pool + 36 S-test)
python $S2/validate_structures.py                 # -> structure_validation.json  (~12 min, one-time)
python $S2/curate_pairs_task2.py                  # -> pairs_task2/*.csv  (~10 min scan, then cached)
python $S2/build_windows.py                       # -> design_windows.json
python $S31/run_task2_guided.py                     # guided       (72 x 3,  ~16 min)
python $S32/run_task2_gibbs.py                    # unguided null (72 x 12, ~7 min)

for a in esm2_t2_rand3 gibbs_t2_r12; do python score_traj_surrogate.py --arm $a; done
```

Then `sweep_results.ipynb` (the sweep + CV leaderboards) and `figures.ipynb` — the write-up
figures end to end: the landscape (§1), emission by lineage (§2), the nested split (§3), both
architecture sweeps (§4-5), both models' held-out predictions (§6-7), and **the design results**
(§8) — the guided arm against its unguided control, per cycle and per condition, with the paired
tests. `figures.ipynb` exports to [`figures/`](figures) next to it (`FIGDIR` in its setup cell);
every folder in this repo keeps its own figures, there is no shared figure directory.

The cross-task-set table in [Design results](#design-results-task-set-2--the-live-task-set) came
from `archive/compare_task_sets.py`, archived with task set 1 since half of what it prints *is*
task 1. It still runs, reading both task sets' design CSVs where they sit.

### Model selection: the CV mattered

The single-split sweep and 3-fold CV disagree on the best surrogate architecture:

| | winner | metric |
|---|---|---|
| single split (48 configs) | `cnn-concat-d2` | 11.42 nm val MAE |
| 3-fold CV (all 48 configs) | **`cnn-max-d1`** | 15.34 ± 1.37 nm mean fold val MAE |

`cnn-concat-d2` drops to 3rd under CV (15.65 nm) — the single-split lead was not robust. CV is
used for **selection only**; the deployed model is a fresh refit on train+val (515 rows) with a
fixed epoch budget of 71 (the mean of the per-fold bests, 56/69/88) since there's no held-out val
left to early-stop on.

**The surrogate's 70/15/15 does not survive this.** Once selection moved to CV over train ∪ val and
the deployed model was refit on that same union, nothing downstream depends on where the 424/91 line
fell: the effective protocol is **85/15 train/test within oracle-train** (515/91), with architecture
and epoch budget chosen by 3-fold CV inside the 515. The one place the boundary still shows up is a
nuisance — `KFold(shuffle=True)` permutes `concat(train, val)` by position, so fold membership does
depend on it. The consequence for the design campaign is that S-train and S-val are one condition;
see [Task selection](#task-selection-spectral-distance-not-identity).

Note also that `fold_va_mae` is the *minimum over epochs* on the fold's own held-out slice (it is
both the early-stopping set and the reported metric), so it is a selection criterion, not a
generalization estimate. Every config gets the same advantage, which is what makes the ranking fair;
comparisons against anything that does no epoch selection belong on the fixed test set instead.

Deployed models: surrogate `cnn-max-d1` **17.55 nm** held-out test MAE · oracle `cnn-max-d1`
**8.71 nm** val / 12.27 nm test (ProstT5, trained on the 80/10/10 oracle split; CV here was scoped
to the surrogate only).

### Task selection: spectral distance, not identity

Design tasks are scaffold→target pairs. The original pipeline matched each scaffold to the target
closest to 80% identity within [70%, 90%] — which guarantees "same-family homolog" but says nothing
about whether the target's spectrum actually *differs*. Two 80%-identity relatives can have nearly
identical ex/em, in which case guided design barely has to move to "succeed". Since the question
here is whether the algorithm can steer designs **toward genuinely different places** in ex/em
space, `curate_pairs_task2.py` gates on distance instead:

- identity is a **floor + cap** (50%–98%), not a band centered on 80% — a near-identical pair with
  a large spectral shift is exactly the informative case (few edits available, far to travel)
- scaffold↔target ex/em Euclidean distance must exceed **40 nm**, >2× the surrogate's 17.55 nm test
  MAE, so success can't be noise
- length tolerance ±30, targets drawn from the oracle's own training pool with a moderate Stokes
  shift and a chromophore motif, scaffolds restricted to the structure-validated set
- among the targets that pass all of that — ~74 per scaffold at the median — **one is drawn
  uniformly at random**, from a stream seeded by `(SEED, scaffold_idx)` so the draw is reproducible
  from identity alone

That last rule is the one thing that changed when task set 1 was archived. It kept the *most
spectrally distant* qualifying target per scaffold, which is the hardest legitimate task available
and not a typical one: the argmax collapses onto a handful of extreme proteins, so its 108 tasks
used only 17 distinct targets at a median 190 nm. Drawing uniformly holds every eligibility rule
fixed and moves only the difficulty — 72 tasks over **60 distinct targets** at a median **79 nm** —
which is what makes the two task sets a controlled pair (see
`archive/README.md`).

Result — **36 pairs per cohort (72 tasks)**:

| cohort | condition | identity | distance (nm) | available pool |
|---|---|---|---|---|
| `knownstruct_Spool` | `seen` | 50–98% (med 77%) | 40–289 (med 79) | 36 / 212 |
| `knownstruct_Stest` | `held-out` | 50–96% (med 70%) | 43–274 (med 76) | 36 / 38 |

(The same scaffolds' *furthest* targets would sit at a median 173 / 202 nm — the argmax rule task 1
used.)

**Two cohorts, two conditions.** Task 1 selected per surrogate *role* (train/val/test) and merged
train+val only at analysis time, because its val pool was the binding constraint at 36 and merging
before selection would let train scaffolds crowd val ones out. Here the pools are merged up front,
because the deployed surrogate does not distinguish train from val: it was refit on train ∪ val, so
an S-val scaffold sits in its training data exactly as an S-train scaffold does. The cohorts are
therefore the two conditions themselves — **`S-pool` (36, in the training pool)** and **`S-test`
(36, unseen)** — and `design_common.COHORT_CONDITION` still maps both task sets onto them.

### How many tasks can this dataset support?

36/cohort is essentially the **maximum balanced design**. The funnel:

| | count |
|---|---|
| dataset entries | 758 |
| have an experimental structure (≥97% id PDB hit) | 445 |
| …and surrogate role ∈ train/val/test | 358 |
| …and pass structure/alignment validation | 266 |
| …and have ≥1 valid target at ≥40 nm | 250 |

So **250 tasks** one-per-scaffold, or **~27,900** if a scaffold may be paired with more than one
target — that is exactly the candidate pool `curate_pairs_task2.py` caches (27,866 pairs over 335
scaffolds, median 74 per scaffold), of which one target per scaffold is drawn.

What binds the balanced case is neither the distance floor nor the structure gate but the **test
cohort, which caps at 38** — only that many of the 758 entries are both structure-known and in the
surrogate's test role (task 1's per-role split hit the same wall at 37 on val). Relaxing
`--min-dist` buys nothing there. Going meaningfully past ~72 balanced tasks needs either multiple
targets per scaffold, or predicted structures to lift the 445-entry experimental-structure
requirement.

### Why `validate_structures.py` exists

`structure_hits.csv` answers a **sequence-level** question: does a PDB entry exist whose declared
entity sequence matches this protein at ≥97%? It never looks at coordinates. But a usable design
window needs a **coordinate-level** property — the atoms actually modelled in one chain must map
cleanly onto this exact sequence, because the window is a set of 0-based indices *into* that
sequence derived from a structure→sequence residue mapping. `pockets.experimental_window` therefore
gates on ≥90% local identity over ≥70% coverage and raises rather than emit a silently mis-numbered
pocket.

Those two questions diverge routinely, and **92 of 358** candidate scaffolds fail:

- SEQRES describes the crystallization construct; the model describes what resolved from the
  density. Disordered termini/loops give high identity but low coverage — `vsfGFP-0` matches 5MFC
  at *100% identity but 66% coverage*.
- the entry may be a complex or fusion whose best-matching chain isn't the FP (22MM's best chain
  matches mCherry at 28% identity).
- **PDB entries are shared across near-identical dataset entries**, so one bad structure eliminates
  a whole family at once: 2G2S kills 23 scaffolds (all `htFuncLib_*`), 7SUN 11, 22MM 9 (all
  `mCherry*`), 9NA8 6 (`deGFP*`).

None of that is knowable without fetching, parsing, picking the best chain and aligning — i.e.
without attempting the build. So validation attempts it once for all 358 candidates (154 unique PDB
entries) and caches the verdict; curation then filters on it up front. **266/358 usable.**

This is what makes selection a single clean pass. Without it, failures surface during
`build_windows.py`, cohorts have to be patched round by round, and the even distance spread degrades
as each patch inherits the previous ordering instead of re-spreading over what's actually still
available. (An earlier iterate-and-patch loop converged to the same 30 pairs in 3 rounds — the
validation-first path reaches them in one, and is far more legible.)

### The design window: what may be edited, and why

Two constraints collapse the ~20²³⁰ sequences of a 230-residue FP into something searchable.
**Physics** — ex/em are set by the polarity, charge and packing of the residues touching the
chromophore, so the causal levers concentrate in the first contact shell, while distant mutations
move folding, brightness and stability rather than peak position. **Model validity** — the surrogate
saw a few hundred FPs and is only credible near that manifold; a search free to mutate anywhere finds
sequences that score well *because* they are far from anything it was trained on. Restricting edits
to the pocket serves both at once.

The window is recomputed per scaffold, not a fixed position list, because the scaffolds are homologs
with different lengths and numbering. `lib/pockets.py` (driven by `build_windows.py`) picks the RCSB
chain whose modelled sequence best locally aligns to the dataset sequence, builds a
structure-residue → sequence-index map from that alignment, finds chromophore position 1 as the X of
the X-[YWHF]-G motif nearest canonical 65, takes the tripeptide's heavy atoms — or, in mature
structures where the tripeptide is deposited as a fused hetero-residue (CRQ, NRQ and relatives),
that residue's atoms — and calls every standard residue with a heavy atom within **5 Å** the pocket.
Window positions are *indices into the sequence*, so a bad structure→sequence map does not produce a
worse window but a wrong one: hence the ≥90% identity / ≥70% coverage gate that raises rather than
emits, and the validation pass [above](#why-validate_structurespy-exists).

| | |
|---|---|
| **editable** | chromophore positions 1–2 + the 5 Å pocket |
| **fixed** | chromophore position 3 + the catalytic pair (GFP's R96/E222, generalized as the Arg and Glu nearest the chromophore) |
| alphabet at chromophore 2 | aromatics {Y, W, H, F} |
| alphabet at H-bond partners | {S, T, Y, N, Q, D, E, H, K, R, W}, for residues whose side-chain N/O lies within 3.5 Å of a chromophore polar atom |

The fixed set is not conservatism: position 3 is the glycine required for backbone cyclization and
the catalytic pair drives maturation, so mutating either yields a protein that may fold but never
fluoresces — a failure mode the surrogate has essentially no training data for and would score as if
nothing had happened. Chromophore position 2 is held aromatic because its ring *is* the conjugated
system whose π→π* gap the design is trying to move. H-bond partners set the chromophore's protonation
state and stabilize the excited state, so design may exchange one partner for another but not
silently delete the interaction; that test is a heavy-atom distance only — no explicit hydrogens, no
angular criterion, no water-mediated bridges — and therefore a capability proxy that over- and
under-calls at the margins by construction.

Across the 108 scaffolds: pockets 12–32 residues (median 24), editable sets 14–34 positions (26),
0–6 H-bond partners (3). Every single-position move is structurally plausible and chemically
appropriate to its role, which is what lets a greedy search travel 40+ nm while keeping ~92%
identity to its scaffold.

> **The family term is gone.** Each editable alphabet used to be intersected with what a
> 763-sequence family alignment supported at that column (Henikoff-weighted, so over-represented
> clades were downweighted), and the survivors' renormalized frequencies became a per-position PSSM
> — the proposal distribution the archived MSA arm sampled from, which left per-position alphabets
> of 2–20 residues (median 13). Empty intersections fell back to the structural constraint or the
> wild-type, a path never taken (0 of 2702 editable positions). The live arms take their candidate
> ranking from ESM-2's masked-LM logits instead and never consulted the PSSM, so it and the
> alignment were removed with that arm — see `archive/msa/` and `archive/README.md`. Windows are now
> structural only.

## Design results (task set 2 — the live task set)

Two arms, run by [`3.1`](3.1_design_run_guided) and its null [`3.2`](3.2_design_run_gibbs) on the same
72 tasks:

- **guided** — at each editable position the candidates are ESM-2 650M masked-LM proposals inside
  the scaffold's Tier-B window, ranked by `z(logp_esm) − z(|ex_err|) − z(|em_err|)` at 1/1/1, i.e.
  kept or rejected by the **surrogate's** distance to the target. k = 10, T = 1.0, 2 cycles,
  3 trials, random visit order.
- **unguided null** — the identical search with λ_ex = λ_em = 0, so the target never enters it:
  Gibbs sampling from the same masked LM in the same window, with the surrogate out of the loop.
  12 trials, because this arm's output is a distribution rather than a point.

Everything is scored by the **oracle**, on the ProstT5 side of the pipeline and never consulted by
either search: `peak_err` = 0.5(|Δex| + |Δem|), the same nm as the model MAEs above. The null is
what the guided numbers must be read against — resampling 26 pocket positions moves ex/em whether
or not anything is steering, and on the archived task set that free movement accounted for about
two thirds of the apparent gain.

The archived task set 1 (each scaffold paired with its *furthest* qualifying target) is the row
this one is read against; both of its arms are still on disk, so
`archive/compare_task_sets.py` still prints both rows:

| task set | n | start | guided (mean of trials) | unguided null | guidance | error closed: guided / null |
|---|---|---|---|---|---|---|
| 1 — furthest target | 72\* | 132.9 | 91.4 | 105.8 | **+14.4 nm** (*p* = 2e−8) | 0.27 / 0.16 |
| 2 — random target | 72 | 70.2 | **43.9** | 53.4 | **+9.6 nm** (*p* = 5e−10) | 0.31 / 0.14 |

\*task 1's row is its ESM-2 guided arm and its Gibbs null on the 72 tasks they share, not the
108-task means in `archive/README.md`.
"Error closed" is the per-task share of the scaffold's own initial error — the axis that survives
two task sets starting at different distances. `archive/compare_task_sets.py` regenerates this;
`archive/compare_task_sets.log` is the run it was written from. Task 2's own arm-vs-null
comparison — the part that does not depend on task 1 — is `figures.ipynb` §8.

**The guidance survives the change, and on the fraction axis it reads slightly stronger.** Guided
beats its own null on **62/72** random-target tasks (*p* = 5e−10), and closes 0.31 of the scaffold
error where the null closes 0.14 — so the null reproduces **45%** of what guidance achieves here
versus **59%** on task 1. In absolute nm the picture is less flattering and essentially unchanged
(26.3 nm of guided movement, 16.8 of it unguided, i.e. 64% free vs task 1's 65%): the two axes
disagree because averaging nm weights the few far tasks, while the fraction weights every task
equally. Either way, the "13–14 nm beyond pocket resampling" that task 1 reported
(`archive/README.md`) becomes **~10 nm** here, on tasks
that are half as far away to begin with.

**Half of task set 2 starts inside the models' noise, and that half barely moves.** 36 of the 72
tasks begin within 50 nm of their target (oracle-scored), which is under 3× the surrogate's 17.55 nm
test MAE, and the design closes only a quarter of it:

| scaffold→target distance | n | start | design (sel.) | mean gain | frac. of error closed |
|---|---|---|---|---|---|
| ≤50 nm | 36 | 39.2 | 29.1 | −10.1 | 0.25 |
| 50–75 | 12 | 58.7 | 40.7 | −18.0 | 0.30 |
| 75–100 | 9 | 87.1 | 41.6 | −45.6 | 0.51 |
| 100–150 | 10 | 123.0 | 57.0 | −66.0 | 0.55 |
| >150 | 5 | 184.7 | 96.1 | −88.6 | 0.48 |

**Above 100 nm, an ordinary target is *easier* than an extreme one at the same distance.** Task 2
closes 0.55 / 0.48 in the 100–150 and >150 nm bands where task 1 closes 0.36 / 0.34 at the same
distances. Task 1's targets are the extreme tail of the dataset's spectral range, and getting to
them may simply not be reachable from a pocket edit; a randomly drawn target 120 nm away is an
ordinary protein, and the search gets over half of the way there. The caveat is sample size — 15
tasks across those two bands, against a 23 nm trial spread.

**16 of 72 tasks end worse than their scaffold** (task 1: 7 of 72). That is the same effect from the
other side: when a task starts 40 nm out, a search whose scorer has a 17.55 nm MAE is as likely to
walk away as toward, and the ≥40 nm curation floor no longer keeps every task clear of the noise.
Under a random-target regime the floor probably has to rise with the surrogate's error, not sit at a
fixed 40 nm.

**Unguided sampling stops working on the held-out cohort, and that is about the tasks, not the
surrogate.** Per condition:

| task 2 | n | start | guided | null | guidance | closed: guided / null |
|---|---|---|---|---|---|---|
| `S-pool` | 36 | 72.9 | 38.4 | 47.0 | +8.5 nm (*p* = 3e−5) | 0.45 / 0.33 |
| `S-test` | 36 | 67.4 | 49.3 | 59.9 | +10.6 nm (*p* = 5e−8) | 0.17 / **−0.06** |

The null *increases* mean error on the held-out cohort — with targets this close, resampling the
pocket drifts a design away from its target as readily as toward it, which is exactly the failure
mode task 1's 100–300 nm distances hid. But note that the **guidance gap is the same in both
conditions** (+8.5 vs +10.6 nm), and the cohort difference is just as large in the arm that has no
surrogate at all (0.33 vs −0.06). So this is a difference between the two cohorts' *tasks*, not
evidence that the surrogate does better on scaffolds it was trained on — the same conclusion task 1
reached, arrived at from the null instead of from the conditions.

Runtime: 16 min for 3.1 (216 searches), 7 min for 3.2 (864). Outputs land in
`knownstruct_task2_esm2_rand3/` and `knownstruct_task2_gibbs_r12/`, and
`figures.ipynb` §8 asks the per-cycle question — the distance distributions with paired tests, as
the archived `archive/visualization_task1_design.ipynb` asks of task set 1.

Two further analyses were run and are **not** in the live notebook; both are in
`archive/visualization_task2.ipynb`, with their figures in `archive/retired_figures/`. Their
findings stand and are reported here because nothing else supersedes them. The first draws the
absolute offset from the target for the three deployable methods (the surrogate's top pick, its
top-3 mean, the unguided control) as one block per method, each ranked from its best pair to its
worst, so the three distributions are compared as shapes: guidance is worth 11.6 nm a pair overall
but only 5.2 nm on the hardest third.

The second changes how everything above should be read. **The unguided control does not improve on its scaffold by moving
toward the target — it collapses onto the middle of the FP distribution** (spread 23/25 nm about its
mean against the scaffolds' 58/59; 37.3 nm from the pool centre against 70.4), which is a better
guess than a peripheral scaffold for a target drawn at random. Scored against a predictor that
ignores the scaffold, the target and the sequence and always outputs the average FP (58.2 nm), the
control manages 53.4 nm and wins on only 40 of 72 tasks, tracking that predictor at *r* = +0.69; the
guided arm manages 43.9 and wins on 48. The control's apparent gain over the scaffold is therefore
not a property of the search but the gap between two things fixed before any design ran — how far
the scaffold started and how peripheral the target is (*r* = +0.83 with that difference). That is
also what the S-pool/S-test asymmetry was: from one cohort to the other the average-FP error rises
12.1 nm and the control's rises 12.9, while the scaffolds start 5.5 nm *closer*. **Read design runs
in this folder against the average-FP predictor, not against the unedited scaffold.**

## Files

**Pipeline** — one folder per stage, see [Layout](#layout).

**Shared config** — `design_common.py` (root) holds every path and the dataset/split/hits loaders.
Start here when tracing what reads what.

**Artifacts** — `data/dual_splits.csv` · `trained_models/{surrogate,oracle}_sweep/` ·
`trained_models/surrogate_cv3.csv` (all 48 configs × 3 folds) ·
`trained_models/surrogate_final/cnn-max-d1_trainval.pt` · `structure_validation.json` ·
`pairs_task2/` (+ `_candidate_pool_cache.json`, **every** qualifying pair — 27,866 of them — which
is what makes a random re-draw possible) ·
`design_windows.json` (a union over both task sets: 137 scaffolds) ·
`peak_designs/structure/` (seven design runs: the live `knownstruct_task2_{esm2_rand3,gibbs_r12}`,
plus task 1's five — `knownstruct_cv_surrogate{,_esm2}` first passes,
`knownstruct_{msa,esm2}_rand3` 3-trial reruns, `knownstruct_gibbs_r12` unguided null — each with a
`surrogate_traj.csv` post-hoc cache)

**Archived, but still on disk** — `pairs/` (task 1, + `_full_pool_cache.json`, one row per
scaffold: its argmax target) and task 1's five design runs above. Only code was archived, never
results, so `archive/compare_task_sets.py` and the two notebooks in `archive/` still run against
them; no live stage reads them. See `archive/README.md`.

**Notebooks** — `sweep_results.ipynb` (sweep + CV) · `figures.ipynb` (every write-up figure:
landscape and lineages §1-2, the split §3, both sweeps §4-5, both models' test predictions §6-7,
the design results §8) · in `archive/`: `visualization.ipynb` and `visualization_task2.ipynb` (the
two notebooks `figures.ipynb` was merged from), `visualize_knownstruct.ipynb` and
`visualization_task1_design.ipynb` (task set 1)

## Notes and gotchas

- **Both CV scripts write the same `trained_models/surrogate_cv3.csv`.** `cv_all_surrogate.py`
  resumes from it and only re-runs configs missing folds, so the top-5 run's results are reused
  rather than recomputed. `cv_surrogate_top5.py` must stay — it defines `FOLD_SEED`, `N_FOLDS` and
  `train_eval_idx`, which `cv_all_surrogate.py` imports.
- **The design engines are resumable per task**, skipping any task whose output CSV already covers
  the requested `--trials`/`--iters`. Changing either therefore requires clearing the affected CSVs
  under `peak_designs/` first, or the old runs are silently kept.
- **`3.1`/`3.2` are runners, and the engines beside them default to task set 2.** Running an engine
  bare writes to the task-2 output root; reproducing an archived task-1 arm means passing
  `--pairs-dir pairs`, its cohorts and its `--outdir` explicitly (each engine's docstring spells
  the invocation out, as does `archive/README.md`).
- **The two task sets are not paired and must not be pooled.** They share only 43 of 72 scaffolds
  and 3 of 72 pairs, and they start at very different distances (132.9 vs 70.2 nm), so any
  task-1-vs-task-2 statement belongs on the fraction-of-error-closed axis or on each arm's gap to
  its own null. `archive/compare_task_sets.py` prints the cross-set rows without a significance
  test for this reason.
- **`design_windows.json` is a union over task sets, not a snapshot of one.** A window is a
  property of the scaffold, so building task 2 added its 29 new scaffolds and left task 1's 108 in
  place. Consequently the file's `n_scaffolds` (137) is larger than any single task set, and
  `--rebuild` drops everything not in the cohorts being built — which on the default (task 2) would
  discard task 1's windows.
- **Task set 2's cohorts are `Spool`/`Stest`, task set 1's are `Strain`/`Sval`/`Stest`.** Both map
  to the same two conditions through `design_common.COHORT_CONDITION`; the difference is only
  whether train+val were merged when selecting or when reporting.
- **`curate_pairs_task2.py --from-cache`** re-selects from `pairs_task2/_candidate_pool_cache.json`
  instead of redoing the ~10-minute all-pairs identity scan. The cache records the criteria that
  produced it and a mismatch is a hard error, so it can't silently yield pairs for the wrong
  criteria. The pool is cached *before* the validation filter, so it stays reusable when structure
  verdicts change — and unlike task 1's one-row-per-scaffold cache it holds every qualifying pair,
  which is what a random re-draw needs.
- **`sweep_peak_oracle.py` loads its base module via an explicit spec** under the distinct
  `sys.modules` key `sweep_peak_oracle_base`. The wrapper is itself importable as
  `sweep_peak_oracle`, so a plain import could otherwise resolve to the half-initialized wrapper.
  It also patches `load_data.__defaults__` — the `cur=CUR` default was bound at def time, so
  reassigning `base.CUR` alone would not take effect.
- **`structures/experimental/` is a cache, not a fixture.** It holds only the entries this
  experiment's scaffolds need; anything missing is fetched from RCSB on first use. Re-curating with
  different scaffolds will trigger new downloads.
