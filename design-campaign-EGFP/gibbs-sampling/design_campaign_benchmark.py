#!/usr/bin/env python
"""Strategy 1 (ESM-2 gibbs) at the equal-budget benchmark depth — SEPARATE output tree.

Identical to ``design_campaign.py`` in every respect that affects sampling: it imports that file's
``CFG`` and changes exactly two things.

  1. OUTPUT TREE. ``designs_benchmark375/`` instead of ``designs/``. This is the whole reason the
     file exists. ``designs/design_EGFP.csv`` is read by ``../make_shortlist_case.py`` (as
     ``GIBBS``) and by ``../visualize_campaign.ipynb``, and the engine is resumable
     (``trial_resume=True``), so running ``--trials 375`` against the normal driver would APPEND
     trials 96-374 to the shortlist's own input rather than error. Shortlist design names are
     rank-derived, so that would silently repoint existing names at different sequences. See
     PROVENANCE.md.

  2. NO PSEUDO-PERPLEXITY. ``ppl_batched`` costs about as much as a whole design iteration and
     would roughly double a multi-hour run for a column nothing in this campaign reads;
     ``../msa-gibbs/``, ``../msa-guided/`` and ``../esm2_guided/`` all skip it already.
     The column is left blank. Run ``--backfill-ppl`` against this tree if it is ever wanted.

WHY 375 TRIALS
--------------
The benchmark gives every strategy a comparable raw-design budget of >= 1,125 designs. Strategy 1
is target-free, so one run serves both targets and its per-target pool is the whole run:

    375 trials x 3 iterations = 1,125 raw designs

matching 125 cells x 3 trials x 3 iterations for the matched-lambda sweep and 125 x 12 x 3 / 4 for
strategy 5 at equal depth. Because trials are seeded independently
(``SEED + si*131 + trial*17``, ``fpdesign/campaign.py`` line 419), trials 0-95 here reproduce
``designs/design_EGFP.csv`` exactly apart from the blank ppl column.

Nothing in ``fpdesign/`` or in the normal driver is modified, and nothing downstream reads this
tree.

Usage
-----
    python design_campaign_benchmark.py --trials 375 --iters 3
"""
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from design_campaign import CFG                                   # noqa: E402
from fpdesign.campaign import Campaign, build_argparser           # noqa: E402

BENCH = replace(CFG,
                name="egfp-gibbs-tierB-benchmark375",
                outdir=HERE / "designs_benchmark375",
                description="EGFP Tier-B pure-ESM-2 (Gibbs) design, equal-budget benchmark depth "
                            "(target-free, ppl disabled); surrogate diagnostic only.")


class BenchCampaign(Campaign):
    """Pseudo-perplexity disabled (point 2 in the module docstring)."""

    def ppl_batched(self, seqlist, bs=None):
        return [float("nan")] * len(seqlist)


if __name__ == "__main__":
    args = build_argparser(BENCH).parse_args()
    BenchCampaign(BENCH, args).run()
