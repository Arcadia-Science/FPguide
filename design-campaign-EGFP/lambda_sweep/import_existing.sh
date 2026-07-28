#!/usr/bin/env bash
# Import the design CSVs that already exist at grid points, so run_sweep.sh skips those cells (the
# engine's trial_resume sees enough trials and reports "cached -> skip"). The skip is per PAIR, so a
# cell holding an imported mOrange CSV is still run for EBFP and vice versa.
#
# Every source was produced by the SAME driver, scaffold, window and pairs CSV at iters=3, T=10,
# k=10 and lam_ex=lam_em=20 -- identical settings to this sweep -- so they are copied verbatim.
# They do carry MORE trials than the sweep's 12 (24 / 48 / 96) and, unlike the fresh cells, they
# have a populated ppl column. See README.md.
set -euo pipefail
cd "$(dirname "$0")"

CAMP=".."
ARCH="$CAMP/archive/brightness-guided/guided_design"
LIVE="$CAMP/brightness-guided/guided_design"

copy() {   # copy <src-dir> <target> <cell-folder>
    local src="$1/design_EGFP-$2.csv" cell="designs/$3"
    if [ ! -f "$src" ]; then echo "MISSING source: $src" >&2; return 1; fi
    mkdir -p "$cell"
    cp -v "$src" "$cell/design_EGFP-$2.csv"
}

# ---- mOrange: three cells ----
copy "$ARCH/designs_lam-bright40_lam-edit10" mOrange "lam-ex20_lam-em20_lam-bright40_lam-edit10"
copy "$LIVE/designs_lam-bright60_lam-edit10" mOrange "lam-ex20_lam-em20_lam-bright60_lam-edit10"
copy "$ARCH/designs_lam-bright60_lam-edit20" mOrange "lam-ex20_lam-em20_lam-bright60_lam-edit20"

# ---- EBFP: two cells (the lam-bright60_lam-edit20 run was mOrange-only) ----
copy "$ARCH/designs_lam-bright40_lam-edit10" EBFP "lam-ex20_lam-em20_lam-bright40_lam-edit10"
copy "$LIVE/designs_lam-bright60_lam-edit10" EBFP "lam-ex20_lam-em20_lam-bright60_lam-edit10"

echo "imported 3 mOrange + 2 EBFP cells"
