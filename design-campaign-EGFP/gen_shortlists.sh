#!/usr/bin/env bash
# Watch the scale_to_96 chain and emit each per-strategy shortlist xlsx as soon as that run finishes.
# Completions are read from scale_to_96.out ("... for above" lines, in chain order):
#   1 gibbs -> mOrange_gibbs   2 spectra -> mOrange_spectra   3 constr -> mOrange_constr
#   4 DMS (EBFP+mOrange in one run) -> mOrange_DMS + EBFP_DMS
# Idempotent: a case is skipped once its shortlists/*.xlsx exists.
#   setsid bash gen_shortlists.sh < /dev/null > gen_shortlists.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-/home/ubuntu/miniconda3/envs/esm2-fp-design/bin/python}"
OUT=scale_to_96.out
SL=shortlists

gen(){ # <case> <outfile>
    [ -f "$SL/$2" ] && return 0
    echo ">>> $(date) generating $1"
    "$PY" -u make_shortlist_case.py "$1"
}

for _ in $(seq 1 240); do   # up to ~120 min
    ncomp=$(grep -c "for above" "$OUT" 2>/dev/null || echo 0)
    [ "$ncomp" -ge 1 ] && gen mOrange_gibbs   shortlist_mOrange_gibbs.xlsx
    [ "$ncomp" -ge 2 ] && gen mOrange_spectra shortlist_mOrange_spectra-guide.xlsx
    [ "$ncomp" -ge 3 ] && gen mOrange_constr  shortlist_mOrange_constrained-spectra-guide.xlsx
    if [ "$ncomp" -ge 4 ]; then
        gen mOrange_DMS shortlist_mOrange_DMS-guide.xlsx
        gen EBFP_DMS    shortlist_EBFP_DMS-guide.xlsx
    fi
    n_files=$(ls "$SL"/*.xlsx 2>/dev/null | wc -l)
    [ "$n_files" -ge 5 ] && { echo "=== all 5 shortlists generated $(date) ==="; break; }
    # bail early if the chain died without finishing
    if ! pgrep -f "scale_to_96.sh|design_campaign.py" >/dev/null && [ "$ncomp" -ge 4 ]; then break; fi
    sleep 30
done
echo "=== gen_shortlists exit $(date) | files: $(ls "$SL"/*.xlsx 2>/dev/null | wc -l)/5 ==="
