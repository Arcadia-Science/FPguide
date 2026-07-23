#!/usr/bin/env python
"""EGFP design campaign (TIER-B window): ESM-2 guided generation, ONE scaffold->target pair
at a time.

Thin wrapper over the shared engine in ``fpdesign.campaign`` (strategy="guided"). The full
procedure, CLI, and CSV schema are documented there; this file only pins the config:

  * scaffold      = EGFP (idx 171, PDB 4EUL);
  * targets       = 4 commercially available, non-LSS FPs spanning blue/green/orange/red
                    (EBFP, mEmerald, mOrange, mCherry) -- see ../pairs/campaign_pairs_egfp.csv;
  * window file   = design_windows_egfp_tierB.json (local copy): 5 A window (pos2 -> aromatic,
                    Gly + catalytic Arg/Glu fixed) + Tier-B H-bond partners (EGFP: Q95,H149,T204)
                    restricted to {S,T,Y,N,Q,D,E,H,K,R,W} via position_constraints;
  * selection     = surrogate-guided: score = z(logp_ESM) - lam_ex*z(|d ex|) - lam_em*z(|d em|),
                    sampled at T=10 (defaults lam=20, k=10).

Usage
-----
    python design_campaign.py --probe          # time ONE pair, project the total, EXIT
    python design_campaign.py --ppl endpoints  # full run, ppl only at scaffold + final round (fast)
    python design_campaign.py --backfill-ppl   # fill the blank intermediate-round ppl cells afterwards
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent               # .../design-campaign-EGFP/guided_design
CAMPAIGN = HERE.parent                                # .../design-campaign-EGFP
REPO = CAMPAIGN.parent                                # .../spectrum-to-fp-design
sys.path.insert(0, str(REPO))
from fpdesign.campaign import CampaignConfig, run     # noqa: E402

CFG = CampaignConfig(
    name="egfp-guided-tierB",
    strategy="guided",
    windows_json=HERE / "design_windows_egfp_tierB.json",
    pairs_csv=CAMPAIGN / "pairs" / "campaign_pairs_egfp.csv",
    outdir=HERE / "designs",
    default_temp=10.0, default_k=10, default_lam_ex=20.0, default_lam_em=20.0,
    add_lam_args=True, add_rescore=True, record_lambda=True,
    per_trial_rng=True, trial_resume=True,   # per-trial Generator -> guided is trial-reproducible/resumable
    description="EGFP -> 4-colour palette, Tier-B guided (surrogate-steered) ESM-2 design.",
)

if __name__ == "__main__":
    run(CFG)
