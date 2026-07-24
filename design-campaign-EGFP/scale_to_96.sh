#!/usr/bin/env bash
# Scale the consolidated EGFP efforts to 96 trials (resumable: appends trials 48..95).
#   gibbs, spectra guide (mOrange), constrained spectra guide (mOrange), DMS guide (EBFP+mOrange).
# Sequential so the single GPU is never contended. Each wrapper writes its own logs/*.log + .last_log.
#   setsid bash scale_to_96.sh < /dev/null > scale_to_96.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

run() { echo "### $(date) :: $*"; ( cd "$1" && shift && bash run_campaign.sh "$@" ); echo "### $(date) :: rc=$? for above"; }

echo "===== scale_to_96 start $(date) ====="
run gibbs-sampling                     --trials 96
run guided-design                      --trials 96 --pairs mOrange
run guided-design-constraint           --trials 96 --lam-edit 10 --pairs mOrange
run brightness-guided/guided_design    --trials 96 --lam-bright 60 --lam-edit 10 --pairs EBFP,mOrange
echo "===== scale_to_96 done $(date) ====="
