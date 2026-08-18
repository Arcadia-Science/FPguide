#!/usr/bin/env bash
# EGFP MSA-Gibbs campaign: UNGUIDED sampling from the 763-sequence family profile over the Tier-B
# window. Target-free, so one design effort per scaffold (one design_EGFP.csv), not one per pair.
# The surrogate only records (ex, em) as a diagnostic; it never steers. Resumable at trial
# granularity -- re-run with a larger --trials to append the missing trials.
#
#   bash run_campaign.sh --trials 96      # the run reported in README.md
#   bash run_campaign.sh --trials 96 --lam-edit 1   # opt-in penalty toward the scaffold residue
#
# No --ppl / --backfill-ppl step: ESM-2 pseudo-perplexity is disabled for this strategy (see
# design_campaign.py), unlike ../gibbs-sampling/run_campaign.sh.
#
# Launch detached so it survives disconnect:
#   setsid bash run_campaign.sh --trials 96 < /dev/null > /dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
mkdir -p designs logs

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/design_campaign_${TS}.log"
echo "$LOG" > .last_log

exec > "$LOG" 2>&1
echo "=== EGFP MSA-gibbs campaign start $(date) | args: $* ==="
"$PY" -u design_campaign.py "$@"
rc=$?
echo "=== EGFP MSA-gibbs campaign done $(date) | rc=${rc} ==="
exit "$rc"
