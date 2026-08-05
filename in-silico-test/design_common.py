"""Config + dataset loader for this experiment's known-structure guided design task.

This module lives at the experiment ROOT and is the single place every path is defined; each
pipeline stage imports it. Stages are separate folders, run in order:

  0_data_split/                  make_dual_split.py        -> data/dual_splits.csv
  1_surrogate_oracle_training/   sweep + 3-fold CV + refit -> trained_models/
  2_design_task_specification/   validate + curate + build -> pairs/, design_windows.json
  3.1_design_run_MSA/            design_knownstruct.py     -> peak_designs/  (family-PSSM arm)
  3.2_design_run_ESM2/           design_knownstruct_esm2.py-> peak_designs/  (ESM-2 proposal arm)

Stage scripts sit one level down, so each begins with a small bootstrap putting this root plus
``lib/`` and ``msa/`` on ``sys.path``. Artifacts and shared inputs stay at the root, since they
are consumed across stages and by the notebooks here.

The folder is SELF-CONTAINED: every code and data dependency lives here, so nothing is read from
``../esm2_design`` or ``../msa_conservation`` at runtime. Copied in as real files (not symlinks,
not imports across folders) -- ``lib/`` holds the vendored modules, which should not be edited
here since they are copies:

  lib/pockets.py        window geometry (5 A pocket + H-bond partners) from an RCSB structure
  lib/peak_models.py    model architectures + checkpoint save/load
  lib/prostt5_embed.py  ProstT5 embedding for oracle scoring
  lib/sweep_peak_oracle_base.py   the shared architecture-sweep implementation
  msa/conservation.py   family-MSA alignment loading + Henikoff weighting
  msa/data/             the MSA RESULT itself (fp_all.aln.fasta + fp_all_meta.csv), i.e. the
                        763-sequence whole-family alignment the PSSMs are computed from
  structures/experimental/  RCSB PDBx cache; self-populating (pockets.py fetches on a miss)
  structure_hits.csv    which dataset entries have a >=97%-identity experimental structure --
                        a property of the protein itself, independent of any train/val/test split

The ONE remaining external reference is ``data/``, whose entries are symlinks to
``dataset_pipeline/data/peak/curated/`` -- the shared curated dataset and its ~2GB of ESM-2 /
ProstT5 residue embedding caches. Those are split-independent inputs to every experiment in the
repo and duplicating 2GB per experiment folder would be wasteful, so they stay symlinked.
``data/dual_splits.csv`` (this experiment's own nested split) IS a real local file.

Pairs (``pairs/pairs_knownstruct_{Strain,Sval,Stest}.csv``, from ``curate_pairs.py``) and their
Tier-B windows + MSA PSSMs (``design_windows.json``, from ``build_windows.py``) are generated
here from scratch, so cohort membership follows THIS experiment's nested split (surrogate
train/val/test) rather than the original esm2_design dual split.

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
MSA_DIR = HERE / "msa"                                # conservation.py + data/fp_all.aln.fasta
WINDOWS_JSON = HERE / "design_windows.json"           # OURS: build_windows.py (built from scratch)
PAIRS_DIR = HERE / "pairs"                            # OURS: curate_pairs.py (our split, distance-spread)

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
SURR_CKPT = HERE / "trained_models" / "surrogate_final" / "cnn-max-d1_trainval.pt"  # final refit
ORAC_CKPT = HERE / "trained_models" / "oracle_sweep" / "cnn-max-d1_s0.pt"           # oracle winner

SEED = 42
# One pair manifest per scaffold surrogate role. These stay three files: the design outputs are
# keyed by cohort directory, and selecting per role is what keeps the distance spread balanced.
DEFAULT_COHORTS = ["knownstruct_Strain", "knownstruct_Sval", "knownstruct_Stest"]

# ...but only TWO conditions are meaningful for the DEPLOYED surrogate (see the module docstring):
#   seen      -- scaffold is inside the refit surrogate's train+val pool   (S-train + S-val, 72)
#   held-out  -- scaffold the surrogate has never been trained on          (S-test, 36)
COHORT_CONDITION = {"knownstruct_Strain": "seen", "knownstruct_Sval": "seen",
                    "knownstruct_Stest": "held-out"}
CONDITIONS = ["seen", "held-out"]
CONDITION_COHORTS = {c: [k for k in DEFAULT_COHORTS if COHORT_CONDITION[k] == c] for c in CONDITIONS}
CONDITION_LABEL = {"seen": "S-pool", "held-out": "S-test"}   # display names for figures/tables


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


def pairs_csv_path(cohort):
    return os.path.join(str(PAIRS_DIR), f"pairs_{cohort}.csv")


def read_pairs(cohort):
    fn = pairs_csv_path(cohort)
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
