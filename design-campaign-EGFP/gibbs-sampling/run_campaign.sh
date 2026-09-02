#!/usr/bin/env bash
# EGFP Gibbs-sampling design campaign: pure ESM-2 masked-LM generation, TARGET-FREE. Gibbs draws
# from the masked-LM conditional and never uses a target, so this is ONE design effort per scaffold
# -- not one per EGFP->target pair -- and it writes a single designs/design_EGFP.csv. (An earlier
# version ran the four EGFP->EBFP/mEmerald/mOrange/mCherry pairs separately; those four CSVs were
# byte-identical apart from the target columns, so they were collapsed. See design_campaign.py.)
# Per trial: any-order masking, T=1 k=10, 3 iters. Per-position choice samples DIRECTLY from ESM-2's
# top-k masked-LM conditional (raw log-probs, T=1) -- no surrogate MAE term; the surrogate only
# records (ex, em)/peak_err as a diagnostic. Resumable at trial granularity: --trials defaults to 6,
# and re-running with a larger value appends the missing trials (the committed run reached 96).
#
#   bash run_campaign.sh                 # ppl only scaffold + final (default), then backfill
#   PPL=all bash run_campaign.sh         # ppl every round
#   bash run_campaign.sh --trials 12     # extra args forwarded to design_campaign.py
#
# Launch detached so it survives disconnect:
#   setsid bash run_campaign.sh < /dev/null > /dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
PPL="${PPL:-endpoints}"
mkdir -p designs logs

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/design_campaign_${TS}.log"
echo "$LOG" > .last_log

exec > "$LOG" 2>&1
echo "=== EGFP gibbs campaign start $(date) | ppl=${PPL} | extra args: $* ==="
"$PY" -u design_campaign.py --ppl "$PPL" "$@"
rc=$?
if [ "$rc" -eq 0 ] && [ "$PPL" = "endpoints" ]; then
    echo "=== backfilling intermediate-round ppl $(date) ==="
    "$PY" -u design_campaign.py --backfill-ppl
    rc=$?
fi
echo "=== EGFP gibbs campaign done $(date) | rc=${rc} ==="
exit "$rc"
