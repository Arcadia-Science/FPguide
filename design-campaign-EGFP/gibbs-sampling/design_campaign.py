#!/usr/bin/env python
"""EGFP design campaign (TIER-B window): pure ESM-2 masked-LM (Gibbs) design, per SCAFFOLD.

Thin wrapper over the shared engine in ``fpdesign.campaign`` (strategy="gibbs", target_free=True).
Gibbs sampling draws from the masked-LM conditional and NEVER uses a target, so a run is a single
design effort PER SCAFFOLD -- not one per scaffold->target pair. (Previously the four per-target
CSVs were byte-identical apart from the target columns.) This config pins:

  * scaffold      = EGFP (idx 171, PDB 4EUL); the pairs CSV is only used to resolve the scaffold,
                    its target rows are de-duplicated away (see ../pairs/campaign_pairs_egfp.csv);
  * window file   = design_windows_egfp_tierB.json (local copy): 5 A window (pos2 -> aromatic,
                    Gly + catalytic Arg/Glu fixed) + Tier-B H-bond partners (EGFP: Q95,H149,T204)
                    restricted to {S,T,Y,N,Q,D,E,H,K,R,W} via position_constraints;
  * selection     = pure ESM-2: sample DIRECTLY from the top-k masked-LM conditional
    softmax(logp / T) at T=1 -- a true Gibbs draw of p(x_i | x_{-i}); the surrogate is loaded only
    to record the design's own (ex, em) as a DIAGNOSTIC (never steers the search).

Output: one target-free ``designs/design_<scaffold>.csv`` (COLS_FREE schema; no target columns).
RESUMABLE AT TRIAL GRANULARITY: re-running with a larger --trials appends the missing trials.

Usage
-----
    python design_campaign.py --probe          # time ONE pair, project the total, EXIT
    python design_campaign.py --ppl endpoints  # full run, ppl only at scaffold + final round (fast)
    python design_campaign.py --backfill-ppl   # fill the blank intermediate-round ppl cells afterwards
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent               # .../design-campaign-EGFP/gibbs-sampling
CAMPAIGN = HERE.parent                                # .../design-campaign-EGFP
REPO = CAMPAIGN.parent                                # .../spectrum-to-fp-design
sys.path.insert(0, str(REPO))
from fpdesign.campaign import CampaignConfig, run     # noqa: E402

CFG = CampaignConfig(
    name="egfp-gibbs-tierB",
    strategy="gibbs",
    windows_json=HERE / "design_windows_egfp_tierB.json",
    pairs_csv=CAMPAIGN / "pairs" / "campaign_pairs_egfp.csv",
    outdir=HERE / "designs",
    default_temp=1.0, default_k=10,
    add_lam_args=False, add_rescore=False, record_lambda=False, trial_resume=True,
    target_free=True,   # gibbs ignores the target -> one design effort per scaffold, target-free CSV
    description="EGFP Tier-B pure-ESM-2 (Gibbs) design (target-free); surrogate diagnostic only.",
)

if __name__ == "__main__":
    run(CFG)
