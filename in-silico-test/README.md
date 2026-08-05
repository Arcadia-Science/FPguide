# in-silico-test — does a differently-structured split change what we conclude?

A self-contained replication of the `esm2_design` guided-design experiment under a **different
train/val/test structure**, to test whether model selection and design outcomes are artifacts of
how the data was split.

The main pipeline uses a *coordinated* dual split with reciprocal invariants
(`S_test ⊆ O_train`, `O_test ⊆ S_train`) so surrogate and oracle can each be evaluated on data the
other trained on. This folder inverts the priority and **nests** instead:

| | split | sizes (N=758) |
|---|---|---|
| **oracle** | 80/10/10 of the *entire* dataset | train 606 · val 76 · test 76 |
| **surrogate** | 70/15/15 carved *only from oracle-train* | train 424 · val 91 · test 91 · excluded 152 |

The nesting guarantees the surrogate never sees anything in the oracle's held-out val/test — a
stricter isolation than the main pipeline — at the cost of ~20% of the surrogate's training data
(the 152 rows sitting in oracle val/test are unusable). Whether that trade changes any conclusion
is the point of the experiment.

## Layout

Code is grouped by pipeline stage; shared inputs, artifacts and the notebooks stay at the root,
since they're consumed across stages.

```
design_common.py               every path + the dataset/split/hits loaders — START HERE
0_data_split/                  make_dual_split.py                    -> data/dual_splits.csv
1_surrogate_oracle_training/   sweep, 3-fold CV, final refit         -> trained_models/
2_design_task_specification/   validate_structures, curate_pairs,
                               build_windows                         -> pairs/, design_windows.json
3.1_design_run_MSA/            design_knownstruct.py                 -> peak_designs/  (MSA arm)
3.2_design_run_ESM2/           design_knownstruct_esm2.py            -> peak_designs/  (ESM-2 arm)
lib/                           vendored modules — copies, don't edit here
msa/                           vendored MSA code + the family alignment (self-contained unit)
data/  structures/             inputs and the RCSB cache
sweep_results.ipynb  visualize_knownstruct.ipynb  visualization.ipynb
```

Stage scripts live one level down, so each starts with a short bootstrap putting the root, `lib/`
and `msa/` on `sys.path`. Run them from the root (`python 2_design_task_specification/curate_pairs.py`).
Each stage folder also keeps its own run logs.

## Standalone by design

Everything the pipeline needs lives here; nothing is read from `../esm2_design` or
`../msa_conservation` at runtime. Vendored as real local files:

| file | what it is |
|---|---|
| `lib/pockets.py` | window geometry (5 Å pocket + H-bond partners) from an RCSB structure |
| `lib/peak_models.py` | model architectures + checkpoint save/load |
| `lib/prostt5_embed.py` | ProstT5 residue embedding, for oracle scoring |
| `lib/sweep_peak_oracle_base.py` | the shared architecture-sweep implementation |
| `msa/conservation.py` | alignment loading + Henikoff sequence weighting |
| `msa/data/` | the MSA **result**: 763-sequence family alignment + metadata |
| `structures/experimental/` | RCSB PDBx cache — self-populating (a miss is fetched) |
| `structure_hits.csv` | which dataset entries have a ≥97%-identity PDB entry |

**One deliberate exception:** `data/` holds symlinks to `dataset_pipeline/data/peak/curated/` —
the shared curated dataset plus ~2GB of ESM-2 / ProstT5 residue-embedding caches. Those are
split-independent inputs to every experiment in the repo, so duplicating 2GB per folder would be
waste. `data/dual_splits.csv` (this experiment's own split) *is* a real local file.

Model weights come from the HF cache (ProstT5, ESM-2) — a package-level dependency, not a
repo-folder one.

## Pipeline

Run in order. Each step's output is committed to disk, and every long step is resumable.

```bash
S0=0_data_split; S1=1_surrogate_oracle_training
S2=2_design_task_specification; S3=3.1_design_run_MSA; S4=3.2_design_run_ESM2

python $S0/make_dual_split.py                     # -> data/dual_splits.csv
python $S1/sweep_peak_oracle.py --role both --seeds 0   # 48 configs x 2 roles -> trained_models/
python $S1/cv_all_surrogate.py                    # 3-fold CV, all 48 surrogate configs
python $S1/train_final_surrogate.py               # refit the CV winner on train+val

python $S2/validate_structures.py                 # -> structure_validation.json  (~12 min, one-time)
python $S2/curate_pairs.py --n 36                 # -> pairs/*.csv  (--from-cache skips the ~9min scan)
python $S2/build_windows.py                       # -> design_windows.json
python $S3/design_knownstruct.py                  # -> peak_designs/...  (108 tasks x 3 trials)
python $S4/design_knownstruct_esm2.py --no-ppl    # same tasks, ESM-2 proposal
```

Then `sweep_results.ipynb` (sweep + CV figures) and `visualize_knownstruct.ipynb` (per-task design
trajectories, regression check).

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
space, `curate_pairs.py` selects on distance instead:

- identity is a **floor + cap** (50%–98%), not a band centered on 80% — a near-identical pair with
  a large spectral shift is exactly the informative case (few edits available, far to travel)
- scaffold↔target ex/em Euclidean distance must exceed **40 nm**, >2× the surrogate's 17.55 nm test
  MAE, so success can't be noise
- per scaffold, the **most spectrally distant** valid target is kept
- each cohort's `--n` pairs are spread evenly across **distance**, giving a deliberate near→far
  difficulty range

Result — **36 pairs per role cohort (108 tasks)**, spanning the full range in all three:

| cohort | condition | identity | distance (nm) | available pool |
|---|---|---|---|---|
| `knownstruct_Strain` | `seen` | 50–95% | 68–301 (med 184) | 36 / 176 |
| `knownstruct_Sval` | `seen` | 50–95% | 68–298 (med 194) | 36 / **36** (entire pool) |
| `knownstruct_Stest` | `held-out` | 50–95% | 83–300 (med 192) | 36 / 38 |

**Three files, two conditions.** Selection runs per surrogate role — the val pool is the binding
constraint at 36, so merging pools before selecting would let train scaffolds crowd val ones out —
but the deployed surrogate does not distinguish train from val: it was refit on train ∪ val, so an
S-val scaffold sits in its training data exactly as an S-train scaffold does. Analysis therefore
groups the same 108 tasks into the two conditions that exist, **`S-pool` (72 = S-train + S-val)**
and **`S-test` (36)**, via `design_common.COHORT_CONDITION`. The manifests, the output directories
and the design runs are untouched by this — it is a regrouping, not a re-run.

### How many tasks can this dataset support?

36/cohort is essentially the **maximum balanced design**. The funnel:

| | count |
|---|---|
| dataset entries | 758 |
| have an experimental structure (≥97% id PDB hit) | 445 |
| …and surrogate role ∈ train/val/test | 358 |
| …and pass structure/alignment validation | 266 |
| …and have ≥1 valid target at ≥40 nm | 250 |

So **250 tasks** one-per-scaffold, or **~21,000** if a scaffold may be paired with more than one
target (each scaffold has ~85 qualifying targets on average; we keep only the most distant).

What binds the balanced case is neither the distance floor nor the structure gate: the **val cohort
caps at 37 even with no distance requirement at all**, because only 54 of the 758 entries are both
structure-known *and* in the surrogate's val role. Relaxing `--min-dist` buys nothing there. Going
meaningfully past ~110 balanced tasks needs either multiple targets per scaffold, or predicted
structures to lift the 445-entry experimental-structure requirement.

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

Each editable alphabet is finally intersected with what the 763-sequence family alignment supports at
that column, using Henikoff-weighted frequencies to downweight over-represented clades; the survivors
keep their renormalized frequencies and become the position's proposal distribution (the PSSM the MSA
arm samples). Empty intersections fall back to the structural constraint, or the wild-type residue —
a path **never taken here (0 of 2702 editable positions)**. Across the 108 scaffolds: pockets 12–32
residues (median 24), editable sets 14–34 positions (26), 0–6 H-bond partners (3), per-position
alphabets 2–20 residues (13). The space is still enormous (~13²⁶ per scaffold), but every
single-position move is simultaneously structurally plausible, chemically appropriate to its role,
and observed in the natural family — which is what lets a greedy search travel 40+ nm while keeping
~92% identity to its scaffold. The [ESM-2 arm](#esm-2-proposal-arm-32_design_run_esm2) holds every
structural component of this window fixed and replaces only the family term.

### Design results (2 cycles × 3 trials, random visit order, λ_ex = λ_em = 1.0, k = 10, T = 1.0)

Oracle-scored. Two reporting rules matter here and both are about *not* consulting the held-out judge:

- **The design for a task is the trial whose SURROGATE error is lowest at the final cycle.** Each task
  is searched 3 times and the trials land in genuinely different places, so some selection is
  unavoidable; selecting by surrogate is a rule the search may apply, selecting by oracle is not.
- **S-train and S-val are one condition, not two tiers.** The deployed surrogate was refit on
  train ∪ val (`n_train=515`, test never trained on), so an S-val scaffold sits *inside* its training
  pool exactly as an S-train scaffold does. All 108 tasks are reported, grouped as `S-pool` (72, in
  the training pool) and `S-test` (36, unseen).

Across all **108 tasks**: **103/108 improved** over their scaffold, mean error **133.2 → 87.2 nm**.
Runtime ~21 min on one GPU for 324 searches.

| condition | n | scaffold | design (surrogate-selected) | improved | every trial improved |
|---|---|---|---|---|---|
| `S-pool` (S-train + S-val) | 72 | 130.7 | 85.6 | 68/72 | 60/72 |
| `S-test` | 36 | 138.1 | 90.5 | 35/36 | 31/36 |
| all | 108 | 133.2 | **87.2** | 103/108 | 91/108 |

**Being inside the surrogate's training pool buys nothing measurable.** `S-pool` beats `S-test` by
4.9 nm on the final design, but it also starts 7.4 nm closer (130.7 vs 138.1) — as a fraction of
initial error the two are 0.35 vs 0.34, and the held-out cohort improves on a *larger* share of its
tasks (35/36 vs 68/72). Whatever the search is exploiting, it is not memorization of the scaffold.

**Trial variance is the largest effect in this experiment.** The 3 trials of a task differ by
**28.1 nm of oracle error on average** (median 25.4, max 82.8) — larger than any difference between
arms, conditions, or visit orders measured anywhere in this folder. Two consequences: single-trial
results here are not reproducible to better than ~25 nm, and "improved over scaffold" drops from
103/108 to **91/108 when every trial has to improve**. Selection across trials is worth something
real: the mean of the trials is 94.0 nm, surrogate-selection gets 87.2, oracle-selection
(unobtainable) would get 80.0 — so on the trial axis the surrogate recovers about half of what
peeking would. (Against the wider trial × cycle bound of 72.6 nm that §8 prices, it recovers 32%.)

Designs stay close to their scaffold (mean **92% identity**), so the movement comes from a handful
of pocket edits rather than drifting toward the target sequence — which is the point: the search is
steering within the scaffold's own structural context, not interpolating toward the target.

Absolute gain grows with task difficulty, and the *fraction* of initial error closed does not
collapse on the hardest tasks:

| scaffold→target distance | n | start | design | mean gain | frac. of error closed |
|---|---|---|---|---|---|
| ≤100 nm | 21 | 70.4 | 48.2 | −22.2 | 0.28 |
| 100–150 | 48 | 126.0 | 81.3 | −44.7 | 0.36 |
| 150–200 | 34 | 170.9 | 112.1 | −58.8 | 0.34 |
| 200–250 | 5 | 208.3 | 137.9 | −70.5 | 0.34 |

The fraction closed is flat at 0.34–0.36 for every band above 100 nm and dips only on the 21 easiest
tasks (0.28) — a shape the single-trial pass (0.18–0.35, non-monotonic in distance) did not show, so
that structure was trial noise. Bands are on the *oracle's* scaffold error, the same axis the gain is
measured against.

**The second cycle buys almost nothing, and the cycle axis is not systematically non-monotonic.**
Mean error by cycle is 133.2 → 95.6 → 94.0 nm. Over all 324 trajectories, cycle 2 is worse than
cycle 1 for 145 and better for 179 — mean **−1.6 nm** (sd 24.9), i.e. no drift in either direction.
This corrects an earlier reading of these runs: the statistic *"N tasks ended worse than a cycle they
passed through"* is `max(0, err₂ − err₁)`, the positive part of a zero-mean difference, so it is
biased positive by construction — at E|Δ| = 18.9 nm it averages 8.6 nm even for a search that only
diffuses. There is no evidence of systematic regression to report, only of a sampler with no memory:
nothing rejects a worse sequence, so a cycle is closer to an independent redraw than to a refinement.

That still argues for an **accept-only-if-better guard on the surrogate**, which the search does have
legitimate access to. `visualize_knownstruct.ipynb` §8 prices every selection rule over both axes
(trial and cycle) against the unobtainable oracle-picked bound.

### ESM-2 proposal arm (`3.2_design_run_ESM2/`)

A controlled swap of the one component the Tier-B design inherited from the family alignment. Same
108 tasks, same windows, same surrogate/oracle, same 1/1/1 z-scored weights, same seed — the only
change is where the per-position proposal comes from:

| | proposal at the edited position |
|---|---|
| `3.1_design_run_MSA/` | family-MSA PSSM — static Henikoff-weighted log-frequencies from `design_windows.json` |
| `3.2_design_run_ESM2/` | ESM-2 650M masked-LM logits, conditioned on the design's **current** sequence |

The window's *structural* parts are kept verbatim (editable set = chromophore 1–2 + 5 Å pocket;
aromatics at chromophore position 2; H-bond-capable residues at H-bond partners). What ESM-2
replaces is the *family* part: the intersection with alignment support, and the frequencies. So a
position's candidates here are its structural constraint, or all 20 AAs where it has none.

**Does the ESM-2 arm use family support?** Not in the search. Candidates come from
`position_constraints` alone (all 20 AAs where a position has none), the ranking over them is ESM-2's
masked-LM log-probability, and the score is `z(logp_esm) − z(|ex_err|) − z(|em_err|)` — the PSSM
enters neither. Family information survives in exactly two places, both outside the search: as a
*task-eligibility* precondition (`validate_structures.py` requires the scaffold be present in the
alignment at its own length, since the window build reads a PSSM through that row — so the same 108
scaffolds are eligible in both arms), and as the `fam_logp` *diagnostic* written every round so the
two arms compare on the same naturalness axis. That is what makes "27% of ESM-2's edits fall outside
the family-supported alphabet" measurable rather than impossible.

Both arms, all 108 tasks, 3 trials each, identical windows/models/weights/seed:

| | scaffold | design (sel.) | mean of trials | improved | trial spread | identity | fam_logp/pos |
|---|---|---|---|---|---|---|---|
| MSA PSSM | 133.2 | **87.2** | 94.0 | 103/108 | 28.1 nm | 0.922 | −2.11 |
| ESM-2 | 133.2 | **89.2** | 92.8 | 97/108 | 24.4 nm | 0.903 | −3.80 |

Per condition, final design (nm):

| | `S-pool` (72) | `S-test` (36) |
|---|---|---|
| MSA PSSM | **85.6** | 90.5 |
| ESM-2 | 90.5 | **86.6** |

The arms trade places between conditions — MSA wins on the seen tasks, ESM-2 on the held-out ones,
by ~5 nm each way. With a 24–28 nm within-task trial spread, that is a reading on how little
separates them, not a generalization story.

**The two arms are indistinguishable in aim, and the earlier claim that ESM-2 was 3.5 nm better was
trial noise.** The sign of the difference depends on which statistic you pick — MSA is 2.0 nm better on
the surrogate-selected design (ESM-2 wins 48/108, Wilcoxon *p* = 0.30), ESM-2 is 1.2 nm better on the
mean over trials (ESM-2 wins 55/108, *p* = 0.69) — and both gaps are ~10× smaller than the 24–28 nm
within-task spread that the 3-trial runs exposed. Any single-trial comparison of these arms was
measuring its own noise.

What *does* separate them is the trade, which the trials do not wash out:

- ESM-2 edits **less conservatively** — 90.3% vs 92.2% identity to scaffold, and a family
  log-likelihood **1.8× worse per position** (−3.80 vs −2.11), i.e. it routinely picks residues the
  family alignment barely supports at that column. The PSSM arm cannot leave family support by
  construction (verified: 0 of 11,627 edits outside it); ESM-2 puts **3,863 of its 14,463 edits
  (27%) outside** the family-supported alphabet.
- **6 fewer tasks improve at all** (97 vs 103), so the wider proposal costs some reliability.
- ESM-2's trials are **less variable** (24.4 vs 28.1 nm spread), the one axis on which it is cleanly
  ahead — a sequence-conditioned proposal is more repeatable than a static profile sampled at T = 1.

Runtime 25 min vs 21 min for 324 searches, with `--no-ppl`. Pseudo-perplexity costs about a whole
design cycle per round (it masks every residue of every design in turn) and dominates at 3 trials, so
it is off by default in these runs and the `ppl` column is blank; `fam_logp` is free and still written
every round, so the naturalness axis is intact.

Outputs land in `knownstruct_msa_rand3/` (`C.PIPE_OUT_R3`) and `knownstruct_esm2_rand3/`
(`C.PIPE_OUT_ESM2_R3`). The single-trial, fixed-order first pass of each arm is left in place in
`knownstruct_cv_surrogate/` and `knownstruct_cv_surrogate_esm2/` for comparison.

## Files

**Pipeline** — one folder per stage, see [Layout](#layout).

**Shared config** — `design_common.py` (root) holds every path and the dataset/split/hits loaders.
Start here when tracing what reads what.

**Artifacts** — `data/dual_splits.csv` · `trained_models/{surrogate,oracle}_sweep/` ·
`trained_models/surrogate_cv3.csv` (all 48 configs × 3 folds) ·
`trained_models/surrogate_final/cnn-max-d1_trainval.pt` · `structure_validation.json` ·
`pairs/` (+ `_full_pool_cache.json`, the full pre-filter pool with the criteria that built it) ·
`design_windows.json` · `peak_designs/structure/knownstruct_cv_surrogate/`

**Notebooks** — `sweep_results.ipynb` · `visualize_knownstruct.ipynb` · `visualization.ipynb`

## Notes and gotchas

- **Both CV scripts write the same `trained_models/surrogate_cv3.csv`.** `cv_all_surrogate.py`
  resumes from it and only re-runs configs missing folds, so the top-5 run's results are reused
  rather than recomputed. `cv_surrogate_top5.py` must stay — it defines `FOLD_SEED`, `N_FOLDS` and
  `train_eval_idx`, which `cv_all_surrogate.py` imports.
- **`design_knownstruct.py` is resumable per task**, skipping any task whose output CSV exists.
  Changing `--iters` therefore requires clearing the affected CSVs under `peak_designs/` first, or
  the old runs are silently kept.
- **`curate_pairs.py --from-cache`** re-selects from `pairs/_full_pool_cache.json` instead of
  redoing the ~9-minute all-pairs identity scan. The cache records the criteria that produced it and
  a mismatch is a hard error, so it can't silently yield pairs for the wrong criteria. The pool is
  cached *before* the validation filter, so it stays reusable when structure verdicts change.
- **`sweep_peak_oracle.py` loads its base module via an explicit spec** under the distinct
  `sys.modules` key `sweep_peak_oracle_base`. The wrapper is itself importable as
  `sweep_peak_oracle`, so a plain import could otherwise resolve to the half-initialized wrapper.
  It also patches `load_data.__defaults__` — the `cur=CUR` default was bound at def time, so
  reassigning `base.CUR` alone would not take effect.
- **`structures/experimental/` is a cache, not a fixture.** It holds only the entries this
  experiment's scaffolds need; anything missing is fetched from RCSB on first use. Re-curating with
  different scaffolds will trigger new downloads.
