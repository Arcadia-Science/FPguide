#!/usr/bin/env python
"""Gibbs-sampling design campaign (TIER-B window): pure ESM-2 masked-LM design, ONE scaffold->target
pair at a time.

Thin wrapper over the shared engine in ``fpdesign.campaign`` (strategy="gibbs"). The full procedure,
CLI, and CSV schema are documented there; this file only pins the config:

  * window file  = design_windows_24_tierB.json (local copy): 5 A window (pos2 -> aromatic, Gly +
    catalytic Arg/Glu fixed) + Tier-B H-bond partners restricted to {S,T,Y,N,Q,D,E,H,K,R,W} via
    position_constraints (enforced automatically from the window file);
  * pairs         = ./pairs_tierB.csv (LOCAL, reordered: the difficulty-spanning priority-5,
    peakdist ~176/99/48/28/18, come first so '--pairs-limit 5' runs them, then the rest);
  * selection     = pure ESM-2: sample DIRECTLY from the top-k masked-LM conditional
    softmax(logp / T) at T=1 -- a true Gibbs draw of p(x_i | x_{-i}); the surrogate is loaded only
    to record (ex, em)/peak_err as a DIAGNOSTIC (never steers the search). No lambda / MAE term, so
    lam_ex/lam_em CSV cells are left blank.

RESUMABLE AT TRIAL GRANULARITY: re-running with a larger --trials appends the missing trials to each
pair CSV (per-trial seeds make trial k reproducible regardless of when it is drawn).

Usage (unchanged from before)
-----------------------------
    python design_campaign.py --probe          # time ONE pair, project the total, EXIT
    python design_campaign.py --ppl endpoints  # full run, ppl only at scaffold + final round (fast)
    python design_campaign.py --backfill-ppl   # fill the blank intermediate-round ppl cells afterwards
    python design_campaign.py --trials 24      # EXPAND every pair to 24 trials (appends 6..23)
    python design_campaign.py --pairs-limit 5  # priority-5 scaffolds only
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent               # .../gibbs-sampling-tierB
CAMPAIGN = HERE.parent                                # .../design-campaign-conventional
REPO = CAMPAIGN.parent                                # .../spectrum-to-fp-design
sys.path.insert(0, str(REPO))
from fpdesign.campaign import CampaignConfig, run     # noqa: E402

CFG = CampaignConfig(
    name="gibbs-tierB",
    strategy="gibbs",
    windows_json=HERE / "design_windows_24_tierB.json",
    pairs_csv=HERE / "pairs_tierB.csv",
    outdir=HERE / "designs",
    default_temp=1.0, default_k=10,
    add_lam_args=False, add_rescore=False, record_lambda=False, trial_resume=True,
    description="Tier-B pure-ESM-2 (Gibbs) design; surrogate is diagnostic only.",
)

if __name__ == "__main__":
    run(CFG)
