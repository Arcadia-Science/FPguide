#!/usr/bin/env bash
# EGFP ESM-2-guided lambda sweep at the MSA effort's scale: lam_ex=lam_em x lam_bright x lam_edit
# = 5 x 5 x 5 = 125 cells, 3 trials x 3 iterations each, both targets (EBFP and mOrange) in one
# process per cell.
#
# This is ../msa-guided/run_sweep.sh's grid run on the REAL ESM-2 masked-LM proposal instead of the
# family profile, so the two efforts differ only in where candidate residues come from. It is the
# matched-lambda control the campaign never ran: strategy 4's own sweep (../lambda_sweep/) used
# lam_ex/em in {10,20,30} x lam_bright in {40,50,60} x lam_edit in {10,15,20} at T=10, a grid that
# cannot express the lam_edit in {0, 0.5} region every one of strategy 5's mOrange winners came
# from. See design_campaign.py, "WHY THIS EFFORT EXISTS".
#
# Because lam_ex/lam_em never take 0 here while lam_bright and lam_edit both do, the 125 cells
# partition exactly onto three strategies -- 5 cells are strategy 2 (peaks only), 20 are strategy 3
# (peaks + edit), 100 are strategy 4 (peaks + brightness + edit). One sweep, not three; see
# design_campaign.py, "ONE SWEEP COVERS STRATEGIES 2, 3 AND 4".
#
# TRIALS defaults to 3, not the MSA effort's 12. That makes this a COARSE LOCALIZATION sweep: 9
# designs per cell against strategy 5's 36. It can answer "does the ESM-2 proposal have a good
# lambda region the T=10 grid missed?" but not "what is its best achievable peak error at equal
# budget", since best-of-N improves with N. Deepen the winning cells with TRIALS=12 for that.
#
# check_scale.py measures where this range puts the sampler on the ESM-2 proposal, as normalized
# entropy of the selection softmax (1.0 = uniform over the k allowed candidates, 0.0 = argmax):
#   lam 1/1/1     T=1     0.310   grid centre, decided but still stochastic
#   lam 4/4/4     T=1     0.040   grid corner, near-argmax
#   lam 0.25/0/0  T=1     0.772   weakest cell (strategy 2), soft but far from uniform
#   lam 2/2/0     T=1     0.292   where strategy 5's mOrange winners sat
#   lam 20/60/10  T=10    0.125   REFERENCE: strategy 4's OWN sweep centre -- sharper than this
#                                 grid's centre, so its 23.7 nm is not a degenerate-sampler artifact
#   lam 1/1/1     T=10    0.984   REFERENCE: the archived negative control, reproduced
# No cell of this grid is degenerate at either end. ESM-2's top-k on its own sits at 0.97 -- nearly
# uniform -- which is the calibration problem ../msa-guided/ was built to address, quantified.
#
# Budget: ~47 s per cell (both targets, 3 trials, 3 iters), from ../lambda_sweep/'s measured 81-83 s
# per cell for 12 trials on one target plus ~6 s of per-process model loading -- so 6.8 s per trial
# per target on an RTX PRO 4500. 125 cells is about 1 h 40 min.
#
# Sequential so the single GPU is never contended. Resumable: the engine skips any cell whose CSV
# already holds >= --trials trials, so re-running after an interruption only fills gaps.
#
# Pseudo-perplexity is disabled in the driver (as in both ../lambda_sweep/ and ../msa-guided/), so
# there is deliberately no --backfill-ppl pass.
#
#   bash run_sweep.sh                          # the whole grid
#   TRIALS=12 bash run_sweep.sh                # deepen (matches ../msa-guided/ per-cell depth)
#   LAM_EDIT="0 0.5" bash run_sweep.sh         # just the region strategy 5's winners came from
#   bash run_sweep.sh --pairs mOrange          # extra args are forwarded to design_campaign.py
#
# Launch detached so it survives disconnect:
#   setsid bash run_sweep.sh < /dev/null > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
TRIALS="${TRIALS:-3}"
ITERS="${ITERS:-3}"
PAIRS="${PAIRS:-EBFP,mOrange}"
TEMP="${TEMP:-1}"
LAM_PEAKS="${LAM_PEAKS:-0.25 0.5 1 2 4}"
LAM_BRIGHT="${LAM_BRIGHT:-0 0.5 1 2 4}"
LAM_EDIT="${LAM_EDIT:-0 0.5 1 2 4}"

mkdir -p designs logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/sweep_${TS}.log"
echo "$LOG" > .last_log

exec > "$LOG" 2>&1
echo "===== ESM-2-guided matched-lambda sweep start $(date) ====="
echo "trials=$TRIALS iters=$ITERS pairs=$PAIRS temp=$TEMP"
echo "lam_ex/em: $LAM_PEAKS | lam_bright: $LAM_BRIGHT | lam_edit: $LAM_EDIT"
echo "extra args: $*"

total=$(( $(echo "$LAM_PEAKS" | wc -w) * $(echo "$LAM_BRIGHT" | wc -w) * $(echo "$LAM_EDIT" | wc -w) ))
echo "$total cells; at ~47 s/cell this is ~$(( total * 47 / 60 )) min"

n=0
t_all=$SECONDS
for P in $LAM_PEAKS; do
  for B in $LAM_BRIGHT; do
    for E in $LAM_EDIT; do
      n=$((n + 1))
      echo ""
      echo "### [$n/$total] $(date) :: lam_ex=lam_em=$P lam_bright=$B lam_edit=$E temp=$TEMP"
      # --temp is passed explicitly even though the driver already defaults to 1.0, so the value
      # that produced each cell is recorded in this log and not only in the CSV.
      "$PY" -u design_campaign.py \
            --trials "$TRIALS" --iters "$ITERS" --pairs "$PAIRS" --temp "$TEMP" \
            --lam-ex "$P" --lam-em "$P" --lam-bright "$B" --lam-edit "$E" "$@"
      rc=$?
      el=$((SECONDS - t_all))
      echo "### [$n/$total] rc=$rc | elapsed ${el}s | eta $(( el * (total - n) / n ))s"
    done
  done
done
echo ""
echo "===== ESM-2-guided matched-lambda sweep done $(date) | $total cells | $((SECONDS - t_all))s ====="
