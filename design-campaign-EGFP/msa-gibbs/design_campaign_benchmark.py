#!/usr/bin/env python
"""Strategy 6 (MSA gibbs) at the equal-budget benchmark depth — SEPARATE output tree.

Identical to ``design_campaign.py`` in every respect that affects sampling: it imports that file's
``CFG`` and its ``MSAGibbsCampaign`` (so the family profile, the hard support constraint and the
disabled pseudo-perplexity all come from there unchanged) and alters exactly one thing.

  OUTPUT TREE. ``designs_benchmark375/`` instead of ``designs/``.

That is the entire reason this file exists, and it is not cosmetic.
``designs/design_EGFP.csv`` is the source of **batch 1's two wet-lab controls** (`B1_09`
`mOrange_MSAgib_01` and `B1_10` `mOrange_MSAgib_02`), and is read by
``../make_shortlist_case.py`` (as ``MGIB``) and ``../visualize_campaign.ipynb``. The engine is
resumable (``trial_resume=True``), so running ``--trials 375`` against the normal driver would
APPEND trials 96-374 to that file rather than error — enlarging the shortlist pool and, because
shortlist design names are rank-derived, silently repointing `mOrange_MSAgib_01` at a different
sequence. See PROVENANCE.md.

WHY 375 TRIALS
--------------
The benchmark gives every strategy a comparable raw-design budget of >= 1,125 designs. Strategy 6
is target-free, so one run serves both targets and its per-target pool is the whole run:

    375 trials x 3 iterations = 1,125 raw designs

Because trials are seeded independently (``SEED + si*131 + trial*17``,
``fpdesign/campaign.py`` line 419), trials 0-95 here are byte-identical to
``designs/design_EGFP.csv``, so the benchmark tree is a strict superset of the shortlist's input
rather than a different sample of it.

This run is cheap — the profile is a lookup and there is no ESM-2 forward pass or candidate
scoring, so the original 96 trials cost 5 s.

Nothing in ``fpdesign/``, in the normal driver, or in ``../msa-guided/`` is modified, and nothing
downstream reads this tree.

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
from design_campaign import CFG, MSAGibbsCampaign   # noqa: E402
from fpdesign.campaign import build_argparser       # noqa: E402

BENCH = replace(CFG,
                name="egfp-msa-gibbs-tierB-benchmark375",
                outdir=HERE / "designs_benchmark375",
                description="EGFP Tier-B MSA-Gibbs design, equal-budget benchmark depth "
                            "(target-free): unguided sampling from the family profile.")

if __name__ == "__main__":
    MSAGibbsCampaign(BENCH, build_argparser(BENCH).parse_args()).run()
