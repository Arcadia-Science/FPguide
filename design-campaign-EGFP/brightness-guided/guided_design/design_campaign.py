#!/usr/bin/env python
"""EGFP brightness-guided design campaign (TIER-B window): ESM-2 guided generation steered by BOTH
the (ex, em) peak surrogate AND a classifier-predicted brightness, ONE scaffold->target pair at a time.

This mirrors design-campaign-EGFP/guided_design/design_campaign.py (same EGFP scaffold, same 4-colour
target palette, same Tier-B 5 A window, same guided engine) and changes ONLY the guidance signal: on
top of the peak terms we add a brightness term so the search is steered toward designs that are both
on-spectrum AND bright. Thin wrapper over the shared engine in ``fpdesign.campaign``
(strategy="guided"); the full procedure, CLI, and CSV schema are documented there. This file pins:

  * scaffold      = EGFP (idx 171, PDB 4EUL);
  * targets       = 4 commercially available, non-LSS FPs spanning blue/green/orange/red
                    (EBFP, mEmerald, mOrange, mCherry) -- see ../pairs/campaign_pairs_egfp.csv;
  * window file   = design_windows_egfp_tierB.json (local copy): 5 A window (pos2 -> aromatic,
                    Gly + catalytic Arg/Glu fixed) + Tier-B H-bond partners (EGFP: Q95,H149,T204)
                    restricted to {S,T,Y,N,Q,D,E,H,K,R,W} via position_constraints;
  * selection     = surrogate + brightness guided: score =
                        z(logp_ESM) - lam_ex*z(|d ex|) - lam_em*z(|d em|) + lam_bright*z(pred_bright),
                    sampled at T=10 (defaults lam_ex=lam_em=lam_bright=20, k=10). pred_bright is
                    logged per design in the extra ``pred_bright`` CSV column.

BRIGHTNESS MODEL (weight + architecture TBD): the brightness classifier is loaded from BRIGHTNESS_CKPT
below. The workflow reconstructs the architecture from the checkpoint's own metadata (any peak_models
out=1 model with train-split mean/std baked in), so finalizing the model later is a one-line path
change here -- nothing else in the campaign needs to move. It currently defaults to the avGFP-DMS
log10-brightness model so the pipeline is runnable end-to-end today.

Usage
-----
    python design_campaign.py --probe          # time ONE pair, project the total, EXIT
    python design_campaign.py --ppl endpoints  # full run, ppl only at scaffold + final round (fast)
    python design_campaign.py --backfill-ppl   # fill the blank intermediate-round ppl cells afterwards
    python design_campaign.py --lam-bright 30  # override the brightness guidance weight

Each run writes to a self-documenting folder named after its guidance weights, e.g.
``designs_lam-bright60_lam-edit10`` (defaults -> ``designs_lam-bright20_lam-edit0``), so different
lam settings never mix and lam_edit -- which has no CSV column -- is captured in the folder name.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent               # .../design-campaign-EGFP/brightness-guided/guided_design
CAMPAIGN = HERE.parent                                # .../design-campaign-EGFP/brightness-guided
REPO = HERE.parents[2]                                # .../spectrum-to-fp-design
sys.path.insert(0, str(REPO))
from fpdesign.campaign import CampaignConfig, DEFAULT_BRIGHTNESS, run     # noqa: E402

# Brightness classifier weight to steer the search. SWAP this when the final model/architecture is
# ready (any peak_models out=1 checkpoint -- the engine reads the arch + scaler from the file itself).
BRIGHTNESS_CKPT = DEFAULT_BRIGHTNESS

CFG = CampaignConfig(
    name="egfp-brightness-guided-tierB",
    strategy="guided",
    windows_json=HERE / "design_windows_egfp_tierB.json",
    pairs_csv=CAMPAIGN / "pairs" / "campaign_pairs_egfp.csv",
    # base output dir; outdir_lambda_suffix appends this run's guidance weights so each lam setting
    # lands in its own self-documenting folder, e.g. designs_lam-bright60_lam-edit10 (and a default
    # run -> designs_lam-bright20_lam-edit0). This avoids mixing runs and records lam_edit -- which
    # has no CSV column -- directly in the folder name.
    outdir=HERE / "designs",
    outdir_lambda_suffix=True,
    brightness_ckpt=BRIGHTNESS_CKPT,
    default_temp=10.0, default_k=10,
    default_lam_ex=20.0, default_lam_em=20.0, default_lam_bright=20.0,
    default_lam_edit=0.0,
    add_lam_args=True, add_lam_bright_arg=True, add_lam_edit_arg=True, add_rescore=True,
    record_lambda=True, record_brightness=True,
    per_trial_rng=True, trial_resume=True,   # per-trial Generator -> guided is trial-reproducible/resumable
    description="EGFP -> 4-colour palette, Tier-B guided (peaks + classifier-brightness) ESM-2 design.",
)

if __name__ == "__main__":
    run(CFG)
