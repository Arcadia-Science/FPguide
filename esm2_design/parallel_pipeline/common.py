"""Shared config + dataset loader for the known-structure FP-design pipeline.

The pipeline uses EXPERIMENTAL RCSB structures (no ESMFold) and runs as standalone
scripts:

  curate_knownstruct.py   -> find scaffolds with an experimental structure + pair them
  select_knownstruct.py   -> pick per-split subsets (S-train / S-val / S-test)
  design_knownstruct.py    -> batched (CUDA fp16) guided design, scaffold -> target
  summarize_knownstruct.py -> per-cohort result summaries

They import this module. Importing it anchors the working directory to
``peak_design/`` so the repo-relative paths used by ``peak_models`` /
``prostt5_embed`` / ``pockets`` / the curated dataset all resolve exactly as in
the notebooks, no matter where the scripts are launched from.

The generic (~90% identity) cohorts below are the legacy defaults; the active
known-structure campaign targets ~80% identity (see curate_knownstruct.py).
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

# Anchor the working directory to peak_design/ regardless of launch location.
PEAK_DIR = Path(__file__).resolve().parent.parent
os.chdir(PEAK_DIR)

# --- repo-relative paths (resolved from peak_design/) --------------------------
CUR = os.path.join("..", "dataset_pipeline", "data", "peak", "curated")
STRUCT_EXPERIMENTAL_DIR = os.path.join("structures", "experimental")
PIPE_OUT = os.path.join("peak_designs", "structure", "parallel_pipeline")
PAIRS_DIR = os.path.join(PIPE_OUT, "pairs")
SURR_CKPT = os.path.join("trained_models", "surrogate_sweep", "cnn-max-d1_s0.pt")
ORAC_CKPT = os.path.join("trained_models", "oracle_sweep", "cnn-max-d2_s0.pt")

# --- curation constants --------------------------------------------------------
SS_MAX = 80.0            # Stokes-shift ceiling for scaffold and target
ID_LO = 0.85             # target identity band (Strain/Stest cohorts, ~90%)
ID_HI = 0.93
ID_TARGET = 0.90
SEED = 42

# chromophore tripeptide: X-[YWHF]-G, expected in the barrel core (pos ~50-85)
CHROMO_AA2 = set("YWHF")

# cohort -> (surrogate_role, oracle_role) required for scaffold / target
COHORTS = {
    "Strain_Otrain": ("train", "train"),
    "Stest_Otrain": ("test", "train"),
}

PAIRS_COLS = [
    "scaffold_idx", "scaffold_name", "scaffold_SS", "scaffold_surr_role",
    "target_idx", "target_name", "target_SS", "target_orac_role", "identity",
]


def has_chromo(seq, lo=50, hi=85):
    """True if the chromophore tripeptide motif X-[YWHF]-G occurs with its first
    residue at sequence index in [lo, hi] (the barrel core for FP-sized proteins)."""
    return any(
        seq[i + 1] in CHROMO_AA2 and seq[i + 2] == "G" and lo <= i <= hi
        for i in range(len(seq) - 2)
    )


def load_dataset(cur=CUR):
    """Load the curated peak dataset. Returns a dict with rows (assignment rows),
    N, peaks (N,2), EXM, EMM, SS (=EMM-EXM), seqs (list[str] by index), Srole/Orole
    (np arrays of split roles), and name2idx."""
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
    dual = list(csv.DictReader(open(os.path.join(cur, "dual_splits.csv"))))
    srole = np.array([d["surrogate_role"] for d in dual])
    orole = np.array([d["oracle_role"] for d in dual])
    name2idx = {r["name"]: i for i, r in enumerate(rows)}
    return dict(rows=rows, N=n, peaks=peaks, EXM=peaks[:, 0], EMM=peaks[:, 1],
                SS=peaks[:, 1] - peaks[:, 0], seqs=seqs, Srole=srole, Orole=orole,
                name2idx=name2idx)


def pairs_csv_path(cohort):
    return os.path.join(PAIRS_DIR, f"pairs_{cohort}.csv")


def read_pairs(cohort):
    fn = pairs_csv_path(cohort)
    if not os.path.exists(fn):
        raise FileNotFoundError(
            f"missing pair manifest for cohort {cohort!r}: {fn}\nrun curate_knownstruct.py first")
    return list(csv.DictReader(open(fn)))
