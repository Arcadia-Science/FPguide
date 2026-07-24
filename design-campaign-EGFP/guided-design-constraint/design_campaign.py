#!/usr/bin/env python
"""EGFP design campaign (TIER-B window): ESM-2 surrogate-guided generation WITH an edit-penalty
constraint, ONE scaffold->target pair at a time.

Identical to design-campaign-EGFP/guided-design/design_campaign.py (same EGFP scaffold, same
4-colour target palette, same Tier-B 5 A window, same guided engine) and changes ONLY the score:
on top of the peak terms it subtracts an edit penalty so the search is steered toward designs that
are on-spectrum AND stay close to the (bright, well-folded) scaffold. This mirrors how the
brightness campaigns add their extra term -- here the extra term is lam_edit*z(is_edit). Thin
wrapper over the shared engine in ``fpdesign.campaign`` (strategy="guided"); the full procedure,
CLI, and CSV schema are documented there. This file pins:

  * scaffold      = EGFP (idx 171, PDB 4EUL);
  * targets       = 4 commercially available, non-LSS FPs spanning blue/green/orange/red
                    (EBFP, mEmerald, mOrange, mCherry) -- see ../pairs/campaign_pairs_egfp.csv;
  * window file   = design_windows_egfp_tierB.json (local copy): 5 A window (pos2 -> aromatic,
                    Gly + catalytic Arg/Glu fixed) + Tier-B H-bond partners (EGFP: Q95,H149,T204)
                    restricted to {S,T,Y,N,Q,D,E,H,K,R,W} via position_constraints;
  * selection     = surrogate-guided + edit penalty: score =
                        z(logp_ESM) - lam_ex*z(|d ex|) - lam_em*z(|d em|) - lam_edit*z(is_edit),
                    sampled at T=10 (defaults lam_ex=lam_em=20, lam_edit=10, k=10). The edit
                    penalty only bites where the scaffold residue is itself in the ESM top-k;
                    forced positions (e.g. pos2->aromatic) are unaffected.

Output goes to a self-documenting folder named after the active guidance weight, i.e.
``designs_lam-edit10`` (override with --lam-edit to write a sibling folder).

Usage
-----
    python design_campaign.py --probe          # time ONE pair, project the total, EXIT
    python design_campaign.py --ppl endpoints  # full run, ppl only at scaffold + final round (fast)
    python design_campaign.py --backfill-ppl   # fill the blank intermediate-round ppl cells afterwards
    python design_campaign.py --lam-edit 5     # override the edit-penalty weight
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent               # .../design-campaign-EGFP/guided-design-constraint
CAMPAIGN = HERE.parent                                # .../design-campaign-EGFP
REPO = CAMPAIGN.parent                                # .../spectrum-to-fp-design
sys.path.insert(0, str(REPO))
from fpdesign.campaign import CampaignConfig, run     # noqa: E402

CFG = CampaignConfig(
    name="egfp-guided-constraint-tierB",
    strategy="guided",
    windows_json=HERE / "design_windows_egfp_tierB.json",
    pairs_csv=CAMPAIGN / "pairs" / "campaign_pairs_egfp.csv",
    # base output dir; outdir_lambda_suffix appends the active edit weight (no brightness model
    # here) -> designs_lam-edit10. A different --lam-edit lands in its own sibling folder, and
    # lam_edit -- which has no CSV column -- is recorded in the folder name.
    outdir=HERE / "designs",
    outdir_lambda_suffix=True,
    default_temp=10.0, default_k=10, default_lam_ex=20.0, default_lam_em=20.0,
    default_lam_edit=10.0,
    add_lam_args=True, add_lam_edit_arg=True, add_rescore=True, record_lambda=True,
    per_trial_rng=True, trial_resume=True,   # per-trial Generator -> guided is trial-reproducible/resumable
    description="EGFP -> 4-colour palette, Tier-B guided (surrogate-steered) ESM-2 design "
                "with a lam_edit edit-penalty constraint toward the scaffold.",
)

if __name__ == "__main__":
    run(CFG)
