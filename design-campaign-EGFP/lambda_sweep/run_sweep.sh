#!/usr/bin/env bash
# EGFP -> mOrange lambda sweep: lam_ex=lam_em in {10,20,30} x lam_bright in {40,50,60}
# x lam_edit in {10,15,20} = 27 cells, 12 trials x 3 iterations each.
#
# Sequential so the single GPU is never contended. Resumable: cells whose CSV already holds
# >= --trials trials are skipped by the engine, so re-running after an interruption (or after
# import_existing.sh) only fills the gaps.
#
# Pseudo-perplexity is disabled in the driver, so there is deliberately NO --backfill-ppl pass
# here (that is what run_campaign.sh does in the other efforts).
#
#   bash run_sweep.sh                      # the whole grid
#   TRIALS=24 bash run_sweep.sh            # deeper
#   bash run_sweep.sh --pairs EBFP         # extra args are forwarded to design_campaign.py
#
# Launch detached so it survives disconnect:
#   setsid bash run_sweep.sh < /dev/null > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-/home/ubuntu/miniconda3/envs/esm2-fp-design/bin/python}"
TRIALS="${TRIALS:-12}"
ITERS="${ITERS:-3}"
PAIRS="${PAIRS:-mOrange}"
LAM_PEAKS="${LAM_PEAKS:-10 20 30}"
LAM_BRIGHT="${LAM_BRIGHT:-40 50 60}"
LAM_EDIT="${LAM_EDIT:-10 15 20}"

mkdir -p designs logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/sweep_${TS}.log"
echo "$LOG" > .last_log

exec > "$LOG" 2>&1
echo "===== lambda sweep start $(date) ====="
echo "trials=$TRIALS iters=$ITERS pairs=$PAIRS | lam_ex/em: $LAM_PEAKS | lam_bright: $LAM_BRIGHT | lam_edit: $LAM_EDIT"
echo "extra args: $*"

n=0
total=$(( $(echo "$LAM_PEAKS" | wc -w) * $(echo "$LAM_BRIGHT" | wc -w) * $(echo "$LAM_EDIT" | wc -w) ))
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
      echo "### [$n/$total] rc=$? elapsed ${SECONDS}s total"
    done
  done
done
echo ""
echo "===== lambda sweep done $(date) | $total cells | $((SECONDS - t_all))s ====="
