#!/usr/bin/env python
"""Conventional design campaign (TIER-B window): ESM-2 guided generation, ONE scaffold->target pair
at a time.

Thin wrapper over the shared engine in ``fpdesign.campaign`` (strategy="guided"). The full
procedure, CLI, and CSV schema are documented there; this file only pins the config:

  * window file  = design_windows_24_tierB.json (local copy): on top of the 5 A window
    (pos2 -> aromatic, Gly + catalytic Arg/Glu fixed) each chromophore H-bond partner (side-chain
    N/O within 3.5 A of a chromophore N/O) is restricted to the H-bond-capable alphabet
    {S,T,Y,N,Q,D,E,H,K,R,W} via position_constraints -- enforced automatically from the window file;
  * pairs         = ../pairs/campaign_pairs_24.csv (the 24 campaign pairs);
  * selection     = surrogate-guided: score = z(logp_ESM) - lam_ex*z(|d ex|) - lam_em*z(|d em|),
                    sampled at T=10 (defaults lam=20, k=10).

Usage (unchanged from before)
-----------------------------
    python design_campaign.py --probe          # time ONE pair, project 24 pairs, EXIT
    python design_campaign.py --ppl endpoints  # full run, ppl only at scaffold + final round (fast)
    python design_campaign.py --backfill-ppl   # fill the blank intermediate-round ppl cells afterwards
    python design_campaign.py --trials 6 --iters 3 --temp 10 --k 10
    python design_campaign.py --pairs-limit 2  # first 2 pairs only
    python design_campaign.py --rescore        # re-fill pred_ex/pred_em/peak_err with the surrogate
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent               # .../guided_design_tierB
CAMPAIGN = HERE.parent                                # .../design-campaign-conventional
REPO = CAMPAIGN.parent                                # .../spectrum-to-fp-design
sys.path.insert(0, str(REPO))
from fpdesign.campaign import CampaignConfig, run     # noqa: E402

CFG = CampaignConfig(
    name="guided-tierB",
    strategy="guided",
    windows_json=HERE / "design_windows_24_tierB.json",
    pairs_csv=CAMPAIGN / "pairs" / "campaign_pairs_24.csv",
    outdir=HERE / "designs",
    default_temp=10.0, default_k=10, default_lam_ex=20.0, default_lam_em=20.0,
    add_lam_args=True, add_rescore=True, record_lambda=True, trial_resume=False,
    description="Tier-B guided (surrogate-steered) ESM-2 design over the 24 campaign pairs.",
)

if __name__ == "__main__":
    run(CFG)
