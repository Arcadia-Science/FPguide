#!/usr/bin/env python
"""EGFP lambda sweep (TIER-B window): the DMS-guide strategy run across a grid of guidance weights.

Same scaffold, window, target and engine as
``design-campaign-EGFP/brightness-guided/guided_design/design_campaign.py`` (strategy="guided",
peaks + classifier-brightness + edit penalty). Two things differ, both needed to sweep lambdas:

  1. OUTPUT DIR PER CELL. The engine's ``outdir_lambda_suffix`` encodes only lam_bright and
     lam_edit, so the three lam_ex/lam_em levels of a sweep would all collide in one folder --
     and because ``existing_pair`` validates only trials/rounds/temp/k, never the lambdas, the
     later cells would be silently skipped as "cached". This driver therefore names the folder
     itself from ALL FOUR weights and leaves outdir_lambda_suffix off.

  2. NO PSEUDO-PERPLEXITY. ``ppl_batched`` runs one masked forward per residue per sequence,
     which costs about as much as a whole design iteration (~49 s of the ~129 s per cell here).
     The column is written but read by nothing in this campaign, and it sat flat at 16.2-16.7
     across every setting already explored, so the sweep skips it: SweepCampaign.ppl_batched
     returns NaN and ``write_pair`` leaves the cell blank. Run ``--backfill-ppl`` later to fill
     it in if it is ever wanted.

Nothing in ``fpdesign/`` is modified, so the other campaigns are untouched.

Usage
-----
    python design_campaign.py --trials 12 --iters 3 --pairs mOrange \
        --lam-ex 20 --lam-em 20 --lam-bright 60 --lam-edit 10
    bash run_sweep.sh          # the whole 3 x 3 x 3 grid, sequentially
"""
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent               # .../design-campaign-EGFP/lambda_sweep
CAMPAIGN = HERE.parent                                # .../design-campaign-EGFP
REPO = HERE.parents[1]                                # .../spectrum-to-fp-design
sys.path.insert(0, str(REPO))
from fpdesign.campaign import (Campaign, CampaignConfig, DEFAULT_BRIGHTNESS,  # noqa: E402
                               build_argparser)

BRIGHTNESS_CKPT = DEFAULT_BRIGHTNESS

BASE = CampaignConfig(
    name="egfp-lambda-sweep-tierB",
    strategy="guided",
    windows_json=HERE / "design_windows_egfp_tierB.json",
    pairs_csv=CAMPAIGN / "pairs" / "campaign_pairs_egfp.csv",
    # base dir only; cell_dir() below appends the per-cell folder. outdir_lambda_suffix stays
    # False because its suffix omits lam_ex/lam_em (see the module docstring).
    outdir=HERE / "designs",
    outdir_lambda_suffix=False,
    brightness_ckpt=BRIGHTNESS_CKPT,
    default_temp=10.0, default_k=10,
    default_lam_ex=20.0, default_lam_em=20.0, default_lam_bright=60.0,
    default_lam_edit=10.0,
    add_lam_args=True, add_lam_bright_arg=True, add_lam_edit_arg=True, add_rescore=True,
    record_lambda=True, record_brightness=True,
    per_trial_rng=True, trial_resume=True,
    description="EGFP -> mOrange lambda sweep: lam_ex/em x lam_bright x lam_edit, Tier-B window.",
)


def cell_dir(args) -> Path:
    """One self-documenting folder per grid cell, carrying all four guidance weights."""
    return BASE.outdir / (f"lam-ex{args.lam_ex:g}_lam-em{args.lam_em:g}"
                          f"_lam-bright{args.lam_bright:g}_lam-edit{args.lam_edit:g}")


class SweepCampaign(Campaign):
    """The campaign with pseudo-perplexity disabled (see point 2 in the module docstring)."""

    def ppl_batched(self, seqlist, bs=None):
        return [float("nan")] * len(seqlist)


if __name__ == "__main__":
    # inline of fpdesign.campaign.run() so the ppl-free subclass is used
    args = build_argparser(BASE).parse_args()
    SweepCampaign(replace(BASE, outdir=cell_dir(args)), args).run()
