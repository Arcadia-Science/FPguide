#!/usr/bin/env bash
# Quick progress readout for the gibbs-sampling design campaign / expansion.
#   bash progress.sh            # expects 24 trials/pair (the current target)
#   bash progress.sh 6          # check against a different target trial count
#   watch -n 30 bash progress.sh   # live-updating view
set -uo pipefail
cd "$(dirname "$0")"
TARGET="${1:-24}"

# the actual worker is the python process; match its binary path so this script never self-matches
pid=$(pgrep -f "bin/python -u design_campaign.py" | head -1)
if [ -n "${pid:-}" ]; then
  el=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
  state="RUNNING (pid $pid, elapsed ${el:-?})"
else
  state="not running (finished or stopped)"
fi

done=0; total=0
for f in designs/*.csv; do
  [ -e "$f" ] || continue
  total=$((total + 1))
  n=$(tail -n +2 "$f" | cut -d, -f13 | sort -un | wc -l)
  [ "$n" -ge "$TARGET" ] && done=$((done + 1))
done

log=$(cat .last_log 2>/dev/null)
# average seconds/pair from finished log lines (each ends with "| <N>s")
avg=$(grep -oE '\| [0-9]+s$' "$log" 2>/dev/null | grep -oE '[0-9]+' | awk '{s+=$1;n++} END{if(n)printf "%.0f", s/n}')
rem=$((total - done)); eta="n/a"
[ -n "${avg:-}" ] && [ "$rem" -gt 0 ] && eta="~$((rem * avg / 60)) min"

echo "campaign : $state"
echo "pairs    : $done / $total at >=${TARGET} trials   (remaining $rem | avg ${avg:-?}s/pair | ETA $eta)"
echo "log      : $log"
echo "---- last 3 pair lines ----"
grep -E '^\[[0-9]+/' "$log" 2>/dev/null | tail -n 3
grep -E '^=== campaign done' "$log" 2>/dev/null | tail -n 1
exit 0
