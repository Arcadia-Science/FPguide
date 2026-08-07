# in-silico-test — does a differently-structured split change what we conclude?

A self-contained replication of the original peak-conditioned guided-design experiment
(`archive/esm2_design/`) under a **different
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
3.3_design_run_gibbs/          design_knownstruct_gibbs.py           -> peak_designs/  (unguided null)
                               compare_arms.py                       -> the three-arm table
4_design_task2/                curate_pairs_task2.py                 -> pairs_task2/   (random target)
5.1_design_run_ESM2/           run_task2_esm2.py                     -> peak_designs/  (task 2, guided)
5.2_design_run_gibbs/          run_task2_gibbs.py                    -> peak_designs/  (task 2, null)
                               compare_task_sets.py                  -> task 1 vs task 2
lib/                           vendored modules — copies, don't edit here
msa/                           vendored MSA code + the family alignment (self-contained unit)
data/  structures/             inputs and the RCSB cache
sweep_results.ipynb  visualize_knownstruct.ipynb  visualization.ipynb  visualization_task2.ipynb
```

Stage scripts live one level down, so each starts with a short bootstrap putting the root, `lib/`
and `msa/` on `sys.path`. Run them from the root (`python 2_design_task_specification/curate_pairs.py`).
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
| `msa/conservation.py` | alignment loading + Henikoff sequence weighting |
| `msa/data/` | the MSA **result**: 763-sequence family alignment + metadata |
| `structure_hits.csv` | which dataset entries have a ≥97%-identity PDB entry |

**Two deliberate exceptions**, both large read-only caches that are split-independent inputs to
every experiment in the repo — duplicating them per folder would be waste, so both are symlinks:

- `data/` → `dataset_pipeline/data/peak/curated/`: the shared curated dataset plus ~2GB of ESM-2 /
  ProstT5 residue-embedding caches. `data/dual_splits.csv` (this experiment's own nested split)
  *is* a real local file.
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
S2=2_design_task_specification; S3=3.1_design_run_MSA; S4=3.2_design_run_ESM2
S5=3.3_design_run_gibbs; S6=4_design_task2; S7=5.1_design_run_ESM2; S8=5.2_design_run_gibbs

python $S0/make_dual_split.py                     # -> data/dual_splits.csv
python $S1/sweep_peak_oracle.py --role both --seeds 0   # 48 configs x 2 roles -> trained_models/
python $S1/cv_all_surrogate.py                    # 3-fold CV, all 48 surrogate configs
python $S1/train_final_surrogate.py               # refit the CV winner on train+val

# task set 1 -- each scaffold paired with its most spectrally distant qualifying target
python $S2/validate_structures.py                 # -> structure_validation.json  (~12 min, one-time)
python $S2/curate_pairs.py --n 36                 # -> pairs/*.csv  (--from-cache skips the ~9min scan)
python $S2/build_windows.py                       # -> design_windows.json
python $S3/design_knownstruct.py                  # -> peak_designs/...  (108 tasks x 3 trials)
python $S4/design_knownstruct_esm2.py --no-ppl    # same tasks, ESM-2 proposal
python $S5/design_knownstruct_gibbs.py --no-ppl   # unguided null: 3.2 with lam_ex = lam_em = 0

# task set 2 -- same rules, but a RANDOM qualifying target (36 S-pool + 36 S-test)
python $S6/curate_pairs_task2.py                  # -> pairs_task2/*.csv  (~10 min scan, then cached)
python $S2/build_windows.py --pairs-dir pairs_task2 \
       --cohorts knownstruct_Spool knownstruct_Stest   # adds its scaffolds to design_windows.json
python $S7/run_task2_esm2.py                      # 5.1 guided  (72 x 3,  ~16 min)
python $S8/run_task2_gibbs.py                     # 5.2 null    (72 x 12, ~7 min)

for a in msa_rand3 esm2_rand3 gibbs_r12 esm2_t2_rand3 gibbs_t2_r12; do
    python score_traj_surrogate.py --arm $a
done
python $S5/compare_arms.py --by-cohort            # three arms, on the 72 tasks they share
python $S8/compare_task_sets.py --by-condition    # task 1 vs task 2, each against its own null
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

Read the 133.2 → 87.2 against the [unguided null](#unguided-control-33_design_run_gibbs-what-the-guidance-actually-buys),
not against zero: resampling the pocket with the surrogate switched off already reaches 105.8 nm, so
~65% of that gain is not the guidance.

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

### Unguided control (`3.3_design_run_gibbs/`): what the guidance actually buys

Every number above is an improvement **over the scaffold**, and a scaffold is not a null: resampling
26 pocket positions moves ex/em whether or not anything is steering. This arm supplies the missing
null. It is `3.2` with **λ_ex = λ_em = 0**, which collapses the score to

```
3.1 / 3.2    score = z(logp_proposal) − 1.0·z(|ex_err|) − 1.0·z(|em_err|)
3.3          score = z(logp_esm)
```

so the target spectrum never enters the search: each editable position is resampled from ESM-2's
conditional given the design's current sequence — a Gibbs sweep over the pocket under the same
Tier-B structural constraints — and the oracle just watches where it wanders. Same windows, same
proposal, same k = 10 / T = 1.0 / 2 cycles, same per-trial random visit order, same seeding.

**Cohorts and trials: S-train (36) + S-test (36), 12 trials = 864 searches.** S-val is dropped —
it is the same reporting condition as S-train (`seen`), so 36 seen / 36 held-out is balanced and
costs a third less. 12 trials rather than 3 because this arm's output *is* a distribution. **All
comparisons below are therefore recomputed on these same 72 tasks**, not read off the 108-task
means above.

| | scaffold | mean of trials | surrogate-selected | oracle-best (unobtainable) | improved | identity | fam_logp |
|---|---|---|---|---|---|---|---|
| 3.1 MSA PSSM (3 trials) | 132.9 | 92.7 | **86.9** | 79.0 | 69/72 | 0.921 | −51.4 |
| 3.2 ESM-2 guided (3 trials) | 132.9 | **91.4** | 88.0 | 80.4 | 65/72 | 0.904 | −95.2 |
| 3.3 Gibbs unguided (12 trials) | 132.9 | 105.8 | 101.2 | 84.9 | 59/72 | 0.902 | −100.9 |
| 3.3 Gibbs unguided (first 3) | 132.9 | 105.7 | 101.3 | 93.5 | 55/72 | 0.902 | −100.4 |

(`3.3_design_run_gibbs/compare_arms.py --by-cohort` regenerates this and the per-cohort breakdown;
`compare_arms.log` is the run it was written from. "mean of trials" is the primary axis because it
is the only one here that does not depend on how many trials an arm drew; "first 3" matches the
null's trial count to the guided arms for the two statistics that do.)

**The guidance is the one manipulation in this comparison that clearly does something.** Guided beats
unguided by **13.0 nm (MSA) / 14.4 nm (ESM-2)** on the mean-of-trials axis, Wilcoxon *p* = 1e−5 and
1.7e−8, and the gap survives on the surrogate-selected axis (+14.3 / +13.2 nm, *p* < 5e−4) and in both
conditions separately. Set that against the arm swap measured the same way on the same 72 tasks —
ESM-2 1.3 nm better on mean-of-trials (*p* = 0.60), MSA 1.1 nm better on the surrogate-selected design
(*p* = 0.61). Changing where the proposal comes from is not measurable at this sample size; switching
the surrogate off is, by roughly an order of magnitude in effect size.

**But two thirds of the headline gain is not the guidance.** Unguided sampling alone takes 132.9 →
105.8 nm, closing **20% of the scaffold error** where the guided arms close 30–31%. So of the 41.5 nm
`S-pool`+`S-test` gain the Design results section reports, **27.1 nm (65%) is reproduced by a search that
never sees the target** — mutating the pocket at all pulls a design toward the dataset's centre of
mass, and most tasks start far from it. The right statement of what the surrogate contributes is
"13–14 nm beyond pocket resampling", not "46 nm below the scaffold".

**Trial variance is intrinsic to the sampler, not created by the guidance.** Matched at 3 trials the
within-task spread is 23.1 nm unguided vs 22.4 (ESM-2) and 27.6 (MSA) guided — indistinguishable.
The 24–28 nm spread flagged above as "the largest effect in this experiment" is therefore a
property of sampling k = 10 candidates at T = 1.0, and would be there with the surrogate switched off.
(The unguided arm's 40.2 nm spread at 12 trials is just max−min over 4× as many draws.)

**The same holds for the two things separating the arms in the ESM-2 comparison above.** Unguided ESM-2
already sits at 0.902 identity and −100.9 fam_logp, essentially where guided ESM-2 lands (0.904,
−95.2) and far from MSA (0.921, −51.4). ESM-2's less conservative editing and worse family
log-likelihood are properties of the ESM-2 proposal itself, not consequences of steering it.

**One uncomfortable number.** Oracle-picking the best of the null's 12 trials gives 84.9 nm — better
than either guided arm's surrogate-selected design (86.9 / 88.0). That is not a usable result (it
peeks at the held-out judge, and needs 4× the searches), but it prices the selection rule: with a
17.55 nm surrogate against a 24 nm trial spread, *choosing* among unguided samples with a perfect
judge is worth about as much as *steering* with an imperfect one. The null's own surrogate-selection
recovers only 4.6 of its 20.9 nm oracle-best gap.

Runtime **7 min** for 864 searches, vs 25 min for 3.2's 324: with both λ at 0 the k = 10 candidates'
surrogate predictions are multiplied by zero, so the arm skips the surrogate forward pass entirely
(10 sequences per position per search — the bulk of 3.2's cost). It is an exact short-circuit, not an
approximation; `--lam-ex 1 --lam-em 1` restores the guided path and reproduces 3.2 byte-for-byte
(verified). Outputs land in `knownstruct_gibbs_r12/` (`C.PIPE_OUT_GIBBS_R12`); the surrogate can
still be applied post-hoc with `python score_traj_surrogate.py --arm gibbs_r12`.

**Is it exactly Gibbs?** No, deliberately: the step keeps 3.2's machinery so the arms differ in one
place only. Candidates are the top k = 10 of the allowed alphabet rather than all of it, and their
log-probs are z-scored before the softmax exactly as 3.2 z-scores them against the error terms —
both rescale the conditional. `--proposal raw` drops the z-scoring and samples the truncated
conditional itself; `zscore` is the default because that is what makes the difference between this
arm and 3.2 attributable to the guidance alone.

### Task set 2 (`4_design_task2/` → `5.1`, `5.2`): the same algorithm on an *ordinary* target

Everything above is measured on tasks built by pairing each scaffold with the **most spectrally
distant** of its ~85 qualifying targets. That is the hardest legitimate task per scaffold, and it is
not a typical one — the argmax rule collapses onto a handful of extreme proteins, so task 1's 108
tasks use only **17 distinct targets** at a median distance of 190 nm. Nothing so far says the
algorithm behaves the same way on a target someone would actually ask for.

Task set 2 changes exactly one thing: the target is drawn **uniformly at random** from the identical
qualifying set — same identity floor/cap, same ≥40 nm distance floor, same length tolerance, same
oracle-train target pool, same structure-validated scaffolds. The result is 72 tasks over **60
distinct targets** at a median distance of **79 nm**, with the chosen target sitting at rank ~30 of
~85 by distance instead of rank 1. Cohorts are merged up front into the two that the analysis
already used — `knownstruct_Spool` (36, S-train + S-val, inside the refit surrogate's training pool)
and `knownstruct_Stest` (36, never trained on) — since task 1 only kept three files to balance a
per-role distance spread.

The guided and unguided arms are the *same scripts* (`5.1`/`5.2` are runners that call
`3.2`/`3.3` with the task-2 manifests and a separate output root), so nothing but the pairs differs.

| task set | n | start | guided (mean of trials) | unguided null | guidance | error closed: guided / null |
|---|---|---|---|---|---|---|
| 1 — furthest target | 72\* | 132.9 | 91.4 | 105.8 | **+14.4 nm** (*p* = 2e−8) | 0.27 / 0.16 |
| 2 — random target | 72 | 70.2 | **43.9** | 53.4 | **+9.6 nm** (*p* = 5e−10) | 0.31 / 0.14 |

\*task 1's row is 3.2/3.3 on the 72 tasks they share, not the 108-task means published above.
"Error closed" is the per-task share of the scaffold's own initial error — the axis that survives
two task sets starting at different distances. `5.2_design_run_gibbs/compare_task_sets.py` regenerates
this; `compare_task_sets.log` is the run it was written from.

**The guidance survives the change, and on the fraction axis it reads slightly stronger.** Guided
beats its own null on **62/72** random-target tasks (*p* = 5e−10), and closes 0.31 of the scaffold
error where the null closes 0.14 — so the null reproduces **45%** of what guidance achieves here
versus **59%** on task 1. In absolute nm the picture is less flattering and essentially unchanged
(26.3 nm of guided movement, 16.8 of it unguided, i.e. 64% free vs task 1's 65%): the two axes
disagree because averaging nm weights the few far tasks, while the fraction weights every task
equally. Either way, "13–14 nm beyond pocket resampling" from the task-1 section becomes **~10 nm**
here, on tasks that are half as far away to begin with.

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

Runtime: 16 min for 5.1 (216 searches), 7 min for 5.2 (864). Outputs land in
`knownstruct_task2_esm2_rand3/` and `knownstruct_task2_gibbs_r12/`, and
`visualization_task2.ipynb` asks the same two design questions `visualization.ipynb` asks of task
set 1 — the per-cycle distance distributions with paired tests, and how far every design landed from
its target. Its second figure is built differently from Section 9's: rather than plotting the peaks
themselves, it draws the absolute offset from the target for the three deployable methods (the
surrogate's top pick, its top-3 mean, the unguided control) as one block per method, each ranked
from its best pair to its worst, so the three distributions are compared as shapes. That is what
shows guidance to be worth 11.6 nm a pair overall but only 5.2 nm on the hardest third.

It also adds a third figure with no counterpart in `visualization.ipynb`, and this one changes how
the other two should be read. **The unguided control does not improve on its scaffold by moving
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
`pairs/` (task 1, + `_full_pool_cache.json`, one row per scaffold: its argmax target) ·
`pairs_task2/` (task 2, + `_candidate_pool_cache.json`, **every** qualifying pair — 27,866 of them —
which is what makes a random re-draw possible) ·
`design_windows.json` (a union over both task sets: 137 scaffolds) ·
`peak_designs/structure/` (seven design runs: `knownstruct_cv_surrogate{,_esm2}` first passes,
`knownstruct_{msa,esm2}_rand3` 3-trial reruns, `knownstruct_gibbs_r12` unguided null, and task 2's
`knownstruct_task2_{esm2_rand3,gibbs_r12}` — each with a `surrogate_traj.csv` post-hoc cache)

**Notebooks** — `sweep_results.ipynb` · `visualize_knownstruct.ipynb` · `visualization.ipynb`
(Sections 8-9 are task set 1's design figures) · `visualization_task2.ipynb` (the same two questions
on task set 2; Sections 1-7 of `visualization.ipynb` are task-independent and are not repeated there)

## Notes and gotchas

- **Both CV scripts write the same `trained_models/surrogate_cv3.csv`.** `cv_all_surrogate.py`
  resumes from it and only re-runs configs missing folds, so the top-5 run's results are reused
  rather than recomputed. `cv_surrogate_top5.py` must stay — it defines `FOLD_SEED`, `N_FOLDS` and
  `train_eval_idx`, which `cv_all_surrogate.py` imports.
- **`design_knownstruct.py` is resumable per task**, skipping any task whose output CSV exists.
  Changing `--iters` therefore requires clearing the affected CSVs under `peak_designs/` first, or
  the old runs are silently kept.
- **`3.3`'s cohorts are a strict subset of `3.1`/`3.2`'s** (72 of 108 — no S-val). Any three-arm
  table has to be recomputed on the shared 72 tasks; the 108-task means published for 3.1/3.2 are
  not comparable to it directly. On the 72, 3.1/3.2 come out at 92.7/91.4 nm mean-of-trials rather
  than the 94.0/92.8 they report over 108.
- **The two task sets are not paired and must not be pooled.** They share only 43 of 72 scaffolds
  and 3 of 72 pairs, and they start at very different distances (132.9 vs 70.2 nm), so any
  task-1-vs-task-2 statement belongs on the fraction-of-error-closed axis or on each arm's gap to
  its own null. `compare_task_sets.py` prints the cross-set rows without a significance test for
  this reason.
- **`design_windows.json` is a union over task sets, not a snapshot of one.** A window is a
  property of the scaffold, so `build_windows.py --pairs-dir pairs_task2 …` adds task 2's 29 new
  scaffolds and leaves task 1's 108 in place. Consequently the file's `n_scaffolds` (137) is larger
  than any single task set, and `--rebuild` drops everything not in the cohorts being built.
- **Task set 2's cohorts are `Spool`/`Stest`, task set 1's are `Strain`/`Sval`/`Stest`.** Both map
  to the same two conditions through `design_common.COHORT_CONDITION`; the difference is only
  whether train+val were merged when selecting or when reporting.
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
