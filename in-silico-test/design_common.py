"""Config + dataset loader for this experiment's known-structure guided design task.

This module lives at the experiment ROOT and is the single place every path is defined; each
pipeline stage imports it. Stages are separate folders, run in order:

  0_data_split/                  make_dual_split.py         -> data/dual_splits.csv
  1_surrogate_oracle_training/   sweep + 3-fold CV + refit  -> trained_models/
  2_design_task_specification/   validate + curate + build  -> pairs_task2/, design_windows.json
  3.1_design_run_guided/           design_knownstruct_guided.py -> peak_designs/  (guided, ESM-2)
  3.2_design_run_gibbs/          design_knownstruct_gibbs.py-> peak_designs/  (unguided null)

That chain runs TASK SET 2: each scaffold paired with a RANDOM qualifying target, over two merged
cohorts (S-pool = train+val, S-test). The ORIGINAL task set -- each scaffold paired with its most
spectrally distant qualifying target -- and its three design arms are archived under ``archive/``
with the notebooks and the cross-set comparison that read them. Four scripts stayed live and moved
INTO the stages above rather than into the archive:

  * ``validate_structures.py`` and ``build_windows.py``, from task 1's task-specification stage;
  * the two design engines, from task 1's two ESM-2 arms. ``run_task2_*.py`` beside each is the
    named entry point; the engine's own CLI defaults are the live task-2 configuration.

WATCH THE STAGE NUMBERS. ``archive/`` keeps task 1's own numbering (2, 3.1, 3.2, 3.3), which does
NOT line up with the live stages above -- e.g. live 3.1 is the ESM-2 guided arm, while archived
3.1 is task 1's MSA arm. Archived paths are always written with the ``archive/`` prefix.

Task 1's OUTPUTS stay where they were written (``pairs/``, the PIPE_OUT_* paths below,
``design_windows.json``'s task-1 scaffolds): only code was archived, never results, so
``archive/compare_task_sets.py`` and the two notebooks in ``archive/`` still run against them.
Nothing in the live chain reads them. See ``archive/README.md``.

Stage scripts sit one level down, so each begins with a small bootstrap putting this root plus
``lib/`` on ``sys.path``. Artifacts and shared inputs stay at the root, since they
are consumed across stages and by the notebooks here.

The folder is SELF-CONTAINED IN CODE: every module it runs lives here, so nothing is imported from
``../fpdesign`` or ``../msa_conservation`` at runtime. Copied in (not symlinks, not imports across
folders) -- ``lib/`` holds the vendored modules, which should not be edited here since they are
copies:

  lib/pockets.py        window geometry (5 A pocket + H-bond partners) from an RCSB structure
  lib/peak_models.py    model architectures + checkpoint save/load
  lib/prostt5_embed.py  ProstT5 embedding for oracle scoring
  lib/sweep_peak_oracle_base.py   the shared architecture-sweep implementation
  structure_hits.csv    which dataset entries have a >=97%-identity experimental structure --
                        a property of the protein itself, independent of any train/val/test split

TWO external references remain, both SYMLINKS to large read-only caches that are split-independent
inputs to every experiment in the repo (duplicating them per folder would be waste):

  data/         -> dataset_pipeline/data/peak/curated/ : the shared curated dataset and its ~2GB of
                   ESM-2 / ProstT5 residue embedding caches. ``data/dual_splits.csv`` (this
                   experiment's own nested split) IS a real local file, not a symlink.
  structures/   -> ../structures : the repo-level RCSB PDBx cache (~175MB), self-populating -- a
                   miss fetched by pockets.py now lands in the shared cache. Paths inside this
                   folder are unchanged, so ``structures/experimental/`` still resolves.

Both hold immutable public downloads, so sharing them across experiments cannot leak anything
between splits.

Pairs (``pairs_task2/pairs_knownstruct_{Spool,Stest}.csv``, from ``curate_pairs_task2.py``) and
their Tier-B windows + MSA PSSMs (``design_windows.json``, from ``build_windows.py``) are
generated here from scratch, so cohort membership follows THIS experiment's nested split
(surrogate train/val/test) rather than the original esm2_design dual split.

THREE COHORT FILES, TWO REPORTING CONDITIONS. The pair manifests are split by the scaffold's
surrogate ROLE because that is what makes the per-cohort distance spread balanced, but the model
those tasks are actually run against does not distinguish train from val: the deployed surrogate
is refit on the train UNION val pool (``train_final_surrogate.py``, n_train=515), so an S-val
scaffold sits inside its training data exactly as an S-train scaffold does. The train/val
boundary is an artifact of the sweep's single-split protocol and does not survive the 3-fold CV
+ refit that produced the deployed model. Analysis therefore groups the same 108 tasks into the
two conditions that exist -- ``seen`` (72) and ``held-out`` (36) -- via ``COHORT_CONDITION``
below. Nothing about the design runs changes; this is a relabeling of what was always there.

Pair selection targets SPECTRAL DISTANCE, not identity-closeness: each scaffold is matched to
the most spectrally distant target passing an identity floor/cap + minimum ex/em distance, and
each cohort is spread evenly across that distance. See ``curate_pairs.py`` for the rationale --
the goal is to demonstrate guided design can move designs toward genuinely different places in
ex/em space, which a "moderate homology" band alone does not guarantee.

The models scoring the design: the surrogate is ``train_final_surrogate.py``'s cnn-max-d1,
refit on this experiment's own nested-split train+val (515 rows) after 3-fold CV confirmed that
architecture; the oracle is this experiment's own oracle sweep winner (also cnn-max-d1, trained
on the 80/10/10 oracle split).

Unlike the source ``common.py`` this was adapted from, this one uses absolute paths throughout
and does NOT ``os.chdir`` -- avoids mutating the process's working directory as a side effect of
import, which would be surprising for a script imported from other tooling in the same session.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                # experiment root -- everything below is local

LIB_DIR = HERE / "lib"                                # vendored modules (copies; don't edit here)
CUR = HERE / "data"                                   # curated dataset (symlinks) + our dual_splits.csv
SPLIT_CSV = HERE / "data" / "dual_splits.csv"         # OURS: make_dual_split.py
STRUCT_DIR = HERE / "structures" / "experimental"     # RCSB PDBx cache, self-populating
HITS_CSV = HERE / "structure_hits.csv"                # structure-known scaffolds (split-independent)
WINDOWS_JSON = HERE / "design_windows.json"           # OURS: build_windows.py (built from scratch)
PAIRS_DIR_T2 = HERE / "pairs_task2"                   # OURS: 2_design_task_specification (random target per scaffold)
PAIRS_DIR = HERE / "pairs"                            # ARCHIVED task 1 (furthest target); outputs kept

# TASK SET 2 (stages 2 / 3.1 / 3.2) -- THE LIVE ARMS. Each scaffold paired with a RANDOM
# qualifying target instead of its furthest one, over two merged cohorts (S-pool, S-test).
PIPE_OUT_ESM2_T2_R3 = HERE / "peak_designs" / "structure" / "knownstruct_task2_esm2_rand3"
PIPE_OUT_GIBBS_T2_R12 = HERE / "peak_designs" / "structure" / "knownstruct_task2_gibbs_r12"

# ---- ARCHIVED task set 1 (each scaffold paired with its most spectrally distant target). The
# code that produced these runs is under archive/; the runs themselves stay here, and archive/'s
# compare_task_sets.py and notebooks read them as task 2's baseline.
PIPE_OUT = HERE / "peak_designs" / "structure" / "knownstruct_cv_surrogate"
# same tasks + same Tier-B windows, ESM-2 masked-LM proposal instead of the family PSSM
PIPE_OUT_ESM2 = HERE / "peak_designs" / "structure" / "knownstruct_cv_surrogate_esm2"

# The two dirs above hold the FIRST pass of each arm: one trial per task, editable positions
# visited in fixed N->C sequence order. Both arms were then rerun with a random visiting order
# (a fresh permutation per trial per cycle, as design-campaign-EGFP does) and 3 independent
# trials per task, which is the current default of 3.1/3.2 -- those land here. Separate dirs so
# the first pass stays reproducible and comparable; `--outdir` overrides either.
PIPE_OUT_R3 = HERE / "peak_designs" / "structure" / "knownstruct_msa_rand3"
PIPE_OUT_ESM2_R3 = HERE / "peak_designs" / "structure" / "knownstruct_esm2_rand3"

# UNGUIDED CONTROL (archived 3.3): the ESM-2 arm with lam_ex = lam_em = 0, i.e. Gibbs sampling from the
# masked-LM inside the same Tier-B window, with the surrogate removed from the loop entirely.
# It answers "how much of 3.1/3.2's movement is the guidance, and how much is resampling the
# pocket at all?". Run on S-train + S-test only (36 + 36), 12 trials, so the null distribution
# of a task's outcome is estimated well enough to compare against a 3-trial guided run.
PIPE_OUT_GIBBS_R12 = HERE / "peak_designs" / "structure" / "knownstruct_gibbs_r12"

# Same scripts, same models, same windows as task 2 -- only the pair manifests differ, which is
# what lets the two task sets be read against each other directly.

SURR_CKPT = HERE / "trained_models" / "surrogate_final" / "cnn-max-d1_trainval.pt"  # final refit
ORAC_CKPT = HERE / "trained_models" / "oracle_sweep" / "cnn-max-d1_s0.pt"           # oracle winner

SEED = 42
# Task 1 kept one pair manifest per scaffold surrogate role -- the design outputs are keyed by
# cohort directory, and selecting per role is what kept its distance spread balanced. Task 2
# merges train+val up front instead (TASK2_COHORTS below), since only the condition matters.
DEFAULT_COHORTS = ["knownstruct_Strain", "knownstruct_Sval", "knownstruct_Stest"]   # archived

# ...but only TWO conditions are meaningful for the DEPLOYED surrogate (see the module docstring):
#   seen      -- scaffold is inside the refit surrogate's train+val pool   (S-train + S-val, 72)
#   held-out  -- scaffold the surrogate has never been trained on          (S-test, 36)
COHORT_CONDITION = {"knownstruct_Strain": "seen", "knownstruct_Sval": "seen",
                    "knownstruct_Stest": "held-out",
                    # task set 2 pools train+val into ONE cohort up front, so its manifests are
                    # the conditions themselves rather than a grouping over three role files
                    "knownstruct_Spool": "seen"}
CONDITIONS = ["seen", "held-out"]
CONDITION_COHORTS = {c: [k for k in DEFAULT_COHORTS if COHORT_CONDITION[k] == c] for c in CONDITIONS}
CONDITION_LABEL = {"seen": "S-pool", "held-out": "S-test"}   # display names for figures/tables

TASK2_COHORTS = ["knownstruct_Spool", "knownstruct_Stest"]   # 36 + 36, THE LIVE PAIR -- stage 2


def load_dataset(cur=CUR):
    """Load the curated peak dataset. Returns a dict with rows (assignment rows),
    N, peaks (N,2), EXM, EMM, seqs (list[str] by index), and name2idx."""
    cur = str(cur)
    rows = list(csv.DictReader(open(os.path.join(cur, "peaks_assignments.csv"))))
    n = len(rows)
    peaks = np.load(os.path.join(cur, "peaks.npy")).astype(np.float32)
    seqs = [None] * n
    h = None
    for line in open(os.path.join(cur, "sequences.fasta")):
        line = line.strip()
        if line.startswith(">"):
            h = int(line[1:].split("|")[0])
        elif line:
            seqs[h] = line
    name2idx = {r["name"]: i for i, r in enumerate(rows)}
    return dict(rows=rows, N=n, peaks=peaks, EXM=peaks[:, 0], EMM=peaks[:, 1],
                seqs=seqs, name2idx=name2idx)


def load_split():
    """index -> (surrogate_role, oracle_role) from this experiment's own dual split."""
    return {int(r["index"]): (r["surrogate_role"], r["oracle_role"])
            for r in csv.DictReader(open(SPLIT_CSV))}


def load_hits():
    """index -> pdb_id for dataset entries with a >=97%-identity experimental structure."""
    return {int(r["idx"]): r["pdb_id"] for r in csv.DictReader(open(HITS_CSV)) if r["pdb_id"]}


def pairs_csv_path(cohort, pairs_dir=PAIRS_DIR):
    """Manifest path for a cohort. `pairs_dir` selects the task set: PAIRS_DIR (task 1, furthest
    target) or PAIRS_DIR_T2 (task 2, random target)."""
    return os.path.join(str(pairs_dir), f"pairs_{cohort}.csv")


def read_pairs(cohort, pairs_dir=PAIRS_DIR):
    fn = pairs_csv_path(cohort, pairs_dir)
    if not os.path.exists(fn):
        raise FileNotFoundError(
            f"missing pair manifest for cohort {cohort!r}: {fn}\n"
            f"(run curate_pairs.py in this folder first)")
    return list(csv.DictReader(open(fn)))


def condition(cohort):
    """'seen' or 'held-out' -- whether the deployed surrogate was trained on this cohort's
    scaffolds. S-train and S-val both map to 'seen' (the refit used train+val)."""
    try:
        return COHORT_CONDITION[cohort]
    except KeyError:
        raise KeyError(f"unknown cohort {cohort!r}; expected one of {DEFAULT_COHORTS}") from None


def read_condition_pairs(cond):
    """Every pair in a reporting condition, each row tagged with its source cohort + condition."""
    if cond not in CONDITIONS:
        raise KeyError(f"unknown condition {cond!r}; expected one of {CONDITIONS}")
    out = []
    for coh in CONDITION_COHORTS[cond]:
        for r in read_pairs(coh):
            out.append({**r, "cohort": coh, "condition": cond})
    return out
