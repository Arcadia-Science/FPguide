#!/usr/bin/env bash
# EGFP guided design campaign: ESM-2 surrogate-guided generation for the 4 EGFP->target pairs
# (EBFP/mEmerald/mOrange/mCherry), one pair at a time (6 trials each, any-order masking,
# T=10 k=10 lam=20, 3 iters). Resumable: pairs whose designs/<pair>.csv already exists are skipped.
#
#   bash run_campaign.sh                 # ppl only scaffold + final (default), then backfill
#   PPL=all bash run_campaign.sh         # ppl every round
#   bash run_campaign.sh --pairs-limit 1 # extra args forwarded to design_campaign.py
#
# Launch detached so it survives disconnect:
#   setsid bash run_campaign.sh < /dev/null > /dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-/home/ubuntu/miniconda3/envs/esm2-fp-design/bin/python}"
PPL="${PPL:-endpoints}"
mkdir -p designs logs

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/design_campaign_${TS}.log"
echo "$LOG" > .last_log

exec > "$LOG" 2>&1
echo "=== EGFP guided campaign start $(date) | ppl=${PPL} | extra args: $* ==="
"$PY" -u design_campaign.py --ppl "$PPL" "$@"
rc=$?
if [ "$rc" -eq 0 ] && [ "$PPL" = "endpoints" ]; then
    echo "=== backfilling intermediate-round ppl $(date) ==="
    "$PY" -u design_campaign.py --backfill-ppl
    rc=$?
fi
echo "=== EGFP guided campaign done $(date) | rc=${rc} ==="
exit "$rc"
