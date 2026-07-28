#!/usr/bin/env bash
# Multiple sequence alignment of the whole curated FP dataset (763 unique sequences).
#
# FFT-NS-i (progressive + iterative refinement, --maxiterate 1000) rather than the
# --auto default: at N=763 --auto falls back to the single-pass FFT-NS-2, and the
# refinement measurably tightens the barrel columns.
#
# --thread 1 is deliberate. MAFFT's parallel refinement visits subalignments in
# nondeterministic order, so multithreaded runs of the same input differ by a few
# columns (1959 vs 1965 observed) and shift downstream counts by +-1. Single-threaded
# with a fixed seed is bit-identical run to run (verified by md5) at a cost of ~4 min
# instead of ~30 s -- worth it for a result that gets quoted.
#
# Input order is preserved (no --reorder) so alignment rows stay aligned to
# data/fp_all_meta.csv by msa_id.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

mafft --maxiterate 1000 --thread 1 --randomseed 0 \
      data/fp_all.fasta > data/fp_all.aln.fasta 2> logs/mafft.log

grep -c '^>' data/fp_all.aln.fasta
