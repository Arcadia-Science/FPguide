#!/usr/bin/env python
"""EGFP ESM-2-guided design at the MSA sweep's LAMBDA SCALE (Tier-B window).

The point of this effort is a fair comparison, so this driver is deliberately a near-copy of
``../lambda_sweep/design_campaign.py`` -- same engine (strategy="guided"), same real ESM-2
masked-LM proposal, same surrogate, same brightness classifier, same Tier-B window, same pairs
CSV, same four-lambda per-cell output folder, same pseudo-perplexity-disabling subclass. THREE
things differ, and nothing else:

  1. OWN OUTPUT TREE. ``outdir = HERE/"designs"`` instead of ``lambda_sweep/designs``. This is
     not cosmetic. ``cell_dir()`` names a folder from the four lambdas but NOT from the
     temperature, and ``existing_pair`` validates trials/rounds/temp/k but not the lambdas, so
     writing T=1 cells into the T=10 sweep's tree would leave two different samplers' designs
     sitting in folders that a ``designs/*/design_EGFP-*.csv`` glob cannot tell apart. Every
     downstream consumer -- ``visualize_campaign.ipynb``, ``make_shortlist_case.py`` -- reads
     exactly that glob and would silently pool them into one "DMS guide" group.

  2. T = 1. See point 3 below; the lambda scale is only meaningful at this temperature.

  3. UNIT LAMBDA DEFAULTS (1.0 for all four, vs 20/20/60/10). Every term in the guided score is
     z-scored across the k candidates, so lambda IS the term's relative weight -- the same
     argument ``../msa-guided/design_campaign.py`` makes in its point 3. At the inherited
     lam_ex=lam_em=20, lam_bright=60, lam_edit=10, T=10 the effective weights are
     0.1 / 2 / 2 / 6 / 1: the proposal counts for a sixtieth of brightness, and the weakest edit
     penalty the T=10 grid can express still outweighs the proposal 10:1.

WHY THIS EFFORT EXISTS
----------------------
The campaign's headline mOrange result compares strategy 4 (``lambda_sweep/``, 27 cells over
lam_ex/em in {10,20,30} x lam_bright in {40,50,60} x lam_edit in {10,15,20} at T=10) against
strategy 5 (``msa-guided/``, 125 cells over {0.25,0.5,1,2,4} x {0,0.5,1,2,4} x {0,0.5,1,2,4} at
T=1) and reports 23.7 nm vs 3.4 nm. Those two grids do not overlap where it matters. Every one of
the mOrange MSA-guide top-10 came from lam_edit in {0, 0.5} -- four of them from lam_edit = 0
exactly -- and the T=10 grid cannot express either value: its weakest edit penalty is 10. The
README defends the comparison on POOL DEPTH (quadrupling strategy 4's pool moved its best error
24.2 -> 23.7 nm), which tests sampling density inside its own grid and says nothing about whether
that grid is centred in the right place.

So this driver runs strategy 5's exact grid on the ESM-2 proposal. Either the ESM-2 cells at
lam_edit ~ 0 reach single-digit nm while in-distribution and confidently bright -- and the
headline is a grid artifact -- or they do not, and the family profile's advantage is robust to
grid coverage rather than merely untested against it.

ONE SWEEP COVERS STRATEGIES 2 AND 4
-----------------------------------
lam_ex/lam_em never take 0 in this grid, while lam_bright and lam_edit both do, so the 125 cells
partition exactly onto the campaign's strategies with no overlap and no gaps:

    strategy 2 "spectra guide"              lam_bright = 0, lam_edit = 0              5 cells
    (retired)  "constrained spectra guide"  lam_bright = 0, lam_edit in {0.5,1,2,4}   20 cells
    strategy 4 "DMS guide"                  lam_bright in {0.5,1,2,4} x lam_edit any  100 cells

STRATEGY 3 WAS RETIRED from the campaign. Its 20 cells were run and are still on disk, but they
are EXCLUDED from every analysis -- ``analyze.py`` drops them before pooling and
``../benchmark_report.py`` has no row for them -- so both report 105 cells, not 125. Nothing reads
those folders; they are kept only as run output. Excluding them blanks four entries in the
heatmap's lam_bright=0 row and nothing else: all 100 lam_bright>0 cells keep full lam_edit
coverage, including the lam_edit in {0, 0.5} region this effort exists to probe.

In ``fpdesign.campaign._select_guided`` the brightness term is
``scores + lam_bright * _zc(Bk[sl, 0])``, so lam_bright = 0 contributes exactly nothing and
reproduces the peaks-only and peaks+edit strategies bit-for-bit -- the multinomial draw sees the
identical score vector and the identical per-trial Generator, and brightness inference is
deterministic under no_grad and consumes no RNG. The difference from running
``guided-design/`` and ``guided-design-constraint/`` separately is that those drivers set
``brightness_ckpt=None``, so they cannot RECORD ``pred_bright``. Keeping the classifier loaded at
lam_bright = 0 leaves the sampler untouched while logging the brightness column for all 125
cells, which is what lets the uniform ID-and-bright filter every other strategy is judged by
apply here too. Verified empirically -- see README.md, "lam_bright = 0 equivalence".

Note that the lam_bright > 0, lam_edit = 0 block (20 cells) is a setting the original strategy 4
never ran, since its grid fixed lam_edit in {10,15,20}. That block is precisely where strategy
5's winners live, so it is the most load-bearing part of the grid.

Nothing in ``fpdesign/``, ``lambda_sweep/`` or ``msa-guided/`` is modified.

Usage
-----
    python design_campaign.py --trials 3 --iters 3 --pairs EBFP,mOrange \
        --lam-ex 1 --lam-em 1 --lam-bright 1 --lam-edit 1 --temp 1
    bash run_sweep.sh          # the whole 5 x 5 x 5 grid, both targets
"""
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent               # .../design-campaign-EGFP/esm2_guided
CAMPAIGN = HERE.parent                                # .../design-campaign-EGFP
REPO = HERE.parents[1]                                # .../spectrum-to-fp-design
sys.path.insert(0, str(REPO))
from fpdesign.campaign import (Campaign, CampaignConfig, DEFAULT_BRIGHTNESS,  # noqa: E402
                               build_argparser)

BRIGHTNESS_CKPT = DEFAULT_BRIGHTNESS

BASE = CampaignConfig(
    name="egfp-esm2-guided-tierB",
    strategy="guided",
    windows_json=HERE / "design_windows_egfp_tierB.json",
    pairs_csv=CAMPAIGN / "pairs" / "campaign_pairs_egfp.csv",
    # base dir only; cell_dir() below appends the per-cell folder. Own tree, NOT lambda_sweep's --
    # see point 1 in the module docstring.
    outdir=HERE / "designs",
    outdir_lambda_suffix=False,
    brightness_ckpt=BRIGHTNESS_CKPT,
    # T=1 with unit lambdas: every z-scored term carries the same weight (point 3 above). These
    # are the same defaults ../msa-guided/design_campaign.py uses, so the two efforts differ only
    # in the proposal distribution.
    default_temp=1.0, default_k=10,
    default_lam_ex=1.0, default_lam_em=1.0, default_lam_bright=1.0,
    default_lam_edit=1.0,
    add_lam_args=True, add_lam_bright_arg=True, add_lam_edit_arg=True, add_rescore=True,
    record_lambda=True, record_brightness=True,
    per_trial_rng=True, trial_resume=True,
    description="EGFP ESM-2-guided lambda sweep at the MSA effort's scale (T=1, unit lambdas): "
                "peaks + brightness + edit penalty, Tier-B window.",
)


def cell_dir(args) -> Path:
    """One self-documenting folder per grid cell, carrying all four guidance weights."""
    return BASE.outdir / (f"lam-ex{args.lam_ex:g}_lam-em{args.lam_em:g}"
                          f"_lam-bright{args.lam_bright:g}_lam-edit{args.lam_edit:g}")


class SweepCampaign(Campaign):
    """The campaign with pseudo-perplexity disabled, as in ../lambda_sweep/design_campaign.py.

    ``ppl_batched`` costs about as much as a whole design iteration and nothing in this campaign
    reads the column; ../msa-guided/ leaves it blank too, so blanking it here keeps the two
    comparable. Run ``--backfill-ppl`` later if it is ever wanted.
    """

    def ppl_batched(self, seqlist, bs=None):
        return [float("nan")] * len(seqlist)


if __name__ == "__main__":
    # inline of fpdesign.campaign.run() so the ppl-free subclass is used
    args = build_argparser(BASE).parse_args()
    SweepCampaign(replace(BASE, outdir=cell_dir(args)), args).run()
