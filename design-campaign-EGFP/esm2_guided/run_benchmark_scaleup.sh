#!/usr/bin/env bash
# Scale the three matched-lambda strategy slices to an equal RAW-DESIGN budget (>= 1,125 each).
#
# The 125 cells of this grid partition onto campaign strategies 2, 3 and 4 by which lambdas are
# zero (see README.md, "One sweep covers strategies 2, 3 and 4"). Those slices have very different
# cell counts, so a uniform trials-per-cell gives them very different budgets. This script gives
# each slice the depth that brings it to ~1,125 raw designs = cells x trials x 3 iterations:
#
#   strategy 2  (lam_bright=0, lam_edit=0)         5 cells x 75 trials x 3 = 1,125
#   strategy 3  (lam_bright=0, lam_edit>0)        20 cells x 19 trials x 3 = 1,140
#   strategy 4  (lam_bright>0, lam_edit any)     100 cells x  4 trials x 3 = 1,200
#
# matching strategy 1 and 6's 375 trials x 3 = 1,125 (target-free, one run serves both targets)
# in ../gibbs-sampling/designs_benchmark375/ and ../msa-gibbs/designs_benchmark375/.
#
# CONSEQUENCE WORTH KNOWING: after this runs, `designs/` is NO LONGER UNIFORM in depth -- cells
# carry 75, 19 or 4 trials depending on their slice. Any analysis that pools all 125 cells is
# therefore depth-weighted toward the strategy-2 corner and must slice before it aggregates.
# The 32.9 nm (mOrange) / 29.9 nm (EBFP) results recorded in README.md were measured at UNIFORM
# 3 trials/cell, before this scale-up; they are frozen as that, not superseded by it.
#
# Resumable and additive: the engine skips any cell already holding >= --trials trials, so each
# phase only runs the missing trials (72, 16 and 1 per cell respectively) and re-running this
# script after an interruption costs only the gap.
#
# Sequential so the single GPU is never contended. Budget at 6.8 s/trial/target + ~6 s per cell of
# process startup, both targets per cell:
#   strategy 2   5 cells x 72 new trials  ~82 min
#   strategy 3  20 cells x 16 new trials  ~75 min
#   strategy 4 100 cells x  1 new trial   ~33 min
#                                total    ~3 h 10 min
#
# Launch detached so it survives disconnect:
#   setsid bash run_benchmark_scaleup.sh < /dev/null > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/scaleup_${TS}.log"
echo "$LOG" > .last_scaleup_log

exec > "$LOG" 2>&1
t0=$SECONDS
echo "===== matched-lambda benchmark scale-up start $(date) ====="

# phase <label> <trials> <lam_bright list> <lam_edit list>
phase() {
  local label="$1" trials="$2" lb="$3" le="$4"
  echo ""
  echo "######## $label :: TRIALS=$trials lam_bright='$lb' lam_edit='$le' | $(date)"
  local t=$SECONDS
  TRIALS="$trials" LAM_BRIGHT="$lb" LAM_EDIT="$le" bash run_sweep.sh
  echo "######## $label done rc=$? in $((SECONDS - t))s | cumulative $((SECONDS - t0))s"
}

phase "strategy 2 (peaks only)"        75 "0"           "0"
phase "strategy 3 (peaks + edit)"      19 "0"           "0.5 1 2 4"
phase "strategy 4 (peaks + bright)"     4 "0.5 1 2 4"   "0 0.5 1 2 4"

echo ""
echo "===== matched-lambda benchmark scale-up done $(date) | $((SECONDS - t0))s ====="
