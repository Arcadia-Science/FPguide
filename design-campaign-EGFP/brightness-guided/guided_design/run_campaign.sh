#!/usr/bin/env bash
# EGFP brightness-guided design campaign: ESM-2 generation steered by the (ex, em) surrogate AND a
# classifier-predicted brightness, for the 4 EGFP->target pairs (EBFP/mEmerald/mOrange/mCherry), one
# pair at a time (6 trials each, any-order masking, T=10 k=10 lam_ex=lam_em=lam_bright=20, 3 iters).
# Resumable: pairs whose designs/<pair>.csv already exists are skipped.
#
#   bash run_campaign.sh                    # ppl only scaffold + final (default), then backfill
#   PPL=all bash run_campaign.sh            # ppl every round
#   bash run_campaign.sh --lam-bright 30    # extra args forwarded to design_campaign.py
#   bash run_campaign.sh --pairs-limit 1    # smoke: one pair only
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
echo "=== EGFP brightness-guided campaign start $(date) | ppl=${PPL} | extra args: $* ==="
"$PY" -u design_campaign.py --ppl "$PPL" "$@"
rc=$?
if [ "$rc" -eq 0 ] && [ "$PPL" = "endpoints" ]; then
    echo "=== backfilling intermediate-round ppl $(date) ==="
    # forward "$@" so --lam-bright/--lam-edit resolve to the SAME lam-suffixed designs_* folder
    "$PY" -u design_campaign.py --backfill-ppl "$@"
    rc=$?
fi
echo "=== EGFP brightness-guided campaign done $(date) | rc=${rc} ==="
exit "$rc"
