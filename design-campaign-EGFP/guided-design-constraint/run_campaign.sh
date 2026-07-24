#!/usr/bin/env bash
# EGFP guided+constraint design campaign: ESM-2 surrogate-guided generation with a lam_edit
# edit-penalty term, for the 4 EGFP->target pairs (EBFP/mEmerald/mOrange/mCherry), one pair at a
# time (T=10 k=10 lam_ex=lam_em=20 lam_edit=10, 3 iters). Output lands in a self-documenting
# designs_lam-edit10/ folder. Resumable at trial granularity (re-run with a larger --trials).
#
#   bash run_campaign.sh                       # ppl only scaffold + final (default), then backfill
#   PPL=all bash run_campaign.sh               # ppl every round
#   bash run_campaign.sh --trials 48           # extra args forwarded to design_campaign.py
#   bash run_campaign.sh --pairs EBFP,mOrange  # restrict to specific pairs
#
# Launch detached so it survives disconnect:
#   setsid bash run_campaign.sh < /dev/null > /dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-/home/ubuntu/miniconda3/envs/esm2-fp-design/bin/python}"
PPL="${PPL:-endpoints}"
mkdir -p logs

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/design_campaign_${TS}.log"
echo "$LOG" > .last_log

exec > "$LOG" 2>&1
echo "=== EGFP guided-constraint campaign start $(date) | ppl=${PPL} | extra args: $* ==="
"$PY" -u design_campaign.py --ppl "$PPL" "$@"
rc=$?
if [ "$rc" -eq 0 ] && [ "$PPL" = "endpoints" ]; then
    echo "=== backfilling intermediate-round ppl $(date) ==="
    # forward "$@" so --lam-edit/--pairs resolve to the SAME designs_lam-edit* folder
    "$PY" -u design_campaign.py --backfill-ppl "$@"
    rc=$?
fi
echo "=== EGFP guided-constraint campaign done $(date) | rc=${rc} ==="
exit "$rc"
