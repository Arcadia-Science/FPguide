#!/usr/bin/env bash
# Conventional design campaign job: ESM-2 guided generation for the 24 scaffold->target pairs,
# one pair at a time (6 trials each, any-order masking, T=10 k=10 lam=20, 3 iters).
# Resumable: pairs whose designs/<pair>.csv already exists are skipped.
#
#   bash run_campaign.sh                 # ppl every round (default), all 24 pairs
#   PPL=endpoints bash run_campaign.sh   # ppl only scaffold + final
#   bash run_campaign.sh --pairs-limit 2 # extra args are forwarded to design_campaign.py
#
# Launch detached so it survives disconnect:
#   setsid bash run_campaign.sh < /dev/null > /dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-/home/ubuntu/miniconda3/envs/esm2-fp-design/bin/python}"
# Tier-B run: default to endpoint-only ppl during the design cycle for speed, then backfill the
# blank intermediate-round ppl cells in a single deduped pass afterwards (see design_campaign.py).
PPL="${PPL:-endpoints}"
mkdir -p designs logs

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/design_campaign_${TS}.log"
echo "$LOG" > .last_log

# route all output (this script + python) into the timestamped log
exec > "$LOG" 2>&1
echo "=== campaign start $(date) | ppl=${PPL} | extra args: $* ==="
"$PY" -u design_campaign.py --ppl "$PPL" "$@"
rc=$?
if [ "$rc" -eq 0 ] && [ "$PPL" = "endpoints" ]; then
    echo "=== backfilling intermediate-round ppl $(date) ==="
    "$PY" -u design_campaign.py --backfill-ppl
    rc=$?
fi
echo "=== campaign done $(date) | rc=${rc} ==="
exit "$rc"
