#!/usr/bin/env bash
# EGFP MSA-guided lambda sweep: lam_ex=lam_em x lam_bright x lam_edit = 5 x 5 x 5 = 125 cells,
# 12 trials x 3 iterations each, both targets (EBFP and mOrange) in one process per cell.
#
# The lambda scale here is NOT the one used by lambda_sweep/. Every term in the guided score is
# z-scored across the k candidates, so lambda is literally the term's relative weight and 1.0
# means "same weight as the family-profile proposal term" (see design_campaign.py, point 3).
# The grid is log-spaced and centred on the all-equal point (1,1,1), spanning a quarter to four
# times it. lam_bright=0 and lam_edit=0 are included as controls: with the family support already
# hard constraining the alphabet, it is an open question whether either penalty is still needed.
#
# check_scale.py measures where that range actually puts the sampler, as normalized entropy of the
# selection softmax (1.0 = uniform over the k allowed candidates, 0.0 = deterministic):
#   lam 1/1/1   T=1    0.194     grid centre, decided but still stochastic
#   lam 4/4/4   T=1    0.031     grid corner, effectively argmax
#   lam 0.5/0/0 T=1    0.694     weakest cell, soft but far from uniform
#   lam 1/1/1   T=10   0.980     the archived negative control, reproduced
# The last row is the failure mode documented in ../archive/README.md (search degenerates into
# random sampling from the proposal's top-k). Running at T=1 keeps the whole grid clear of it.
#
# Budget: ~143 s per cell (both targets, 12 trials, 3 iters, measured on an RTX PRO 4500),
# so 125 cells is about 5 h -- inside a 15 h window with 3x headroom.
#
# Sequential so the single GPU is never contended. Resumable: the engine skips any cell whose
# CSV already holds >= --trials trials, so re-running after an interruption only fills gaps.
#
# Pseudo-perplexity is disabled in the driver (ESM-2 ppl is not this strategy's naturalness
# measure), so there is deliberately no --backfill-ppl pass.
#
#   bash run_sweep.sh                          # the whole grid
#   TRIALS=6 bash run_sweep.sh                 # shallower
#   LAM_BRIGHT="1 2" bash run_sweep.sh         # a slice
#   bash run_sweep.sh --pairs mOrange          # extra args are forwarded to design_campaign.py
#
# Launch detached so it survives disconnect:
#   setsid bash run_sweep.sh < /dev/null > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-/home/ubuntu/miniconda3/envs/esm2-fp-design/bin/python}"
TRIALS="${TRIALS:-12}"
ITERS="${ITERS:-3}"
PAIRS="${PAIRS:-EBFP,mOrange}"
LAM_PEAKS="${LAM_PEAKS:-0.25 0.5 1 2 4}"
LAM_BRIGHT="${LAM_BRIGHT:-0 0.5 1 2 4}"
LAM_EDIT="${LAM_EDIT:-0 0.5 1 2 4}"

mkdir -p designs logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/sweep_${TS}.log"
echo "$LOG" > .last_log

exec > "$LOG" 2>&1
echo "===== MSA-guided lambda sweep start $(date) ====="
echo "trials=$TRIALS iters=$ITERS pairs=$PAIRS"
echo "lam_ex/em: $LAM_PEAKS | lam_bright: $LAM_BRIGHT | lam_edit: $LAM_EDIT"
echo "extra args: $*"

total=$(( $(echo "$LAM_PEAKS" | wc -w) * $(echo "$LAM_BRIGHT" | wc -w) * $(echo "$LAM_EDIT" | wc -w) ))
echo "$total cells; at ~143 s/cell this is ~$(( total * 143 / 3600 )) h"

n=0
t_all=$SECONDS
for P in $LAM_PEAKS; do
  for B in $LAM_BRIGHT; do
    for E in $LAM_EDIT; do
      n=$((n + 1))
      echo ""
      echo "### [$n/$total] $(date) :: lam_ex=lam_em=$P lam_bright=$B lam_edit=$E"
      "$PY" -u design_campaign.py \
            --trials "$TRIALS" --iters "$ITERS" --pairs "$PAIRS" \
            --lam-ex "$P" --lam-em "$P" --lam-bright "$B" --lam-edit "$E" "$@"
      rc=$?
      el=$((SECONDS - t_all))
      echo "### [$n/$total] rc=$rc | elapsed ${el}s | eta $(( el * (total - n) / n ))s"
    done
  done
done
echo ""
echo "===== MSA-guided sweep done $(date) | $total cells | $((SECONDS - t_all))s ====="
