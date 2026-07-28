#!/usr/bin/env python
"""EGFP MSA-guided design campaign (Tier-B window): the family profile replaces ESM-2.

Same scaffold, window, targets, surrogate and brightness head as ``lambda_sweep/``. One thing
changes -- where the *proposal distribution* comes from -- and the lambda scale is redefined to
match.

WHY
---
ESM-2 650M is close to uninformative on this fold. Masked-marginal top-1 accuracy is 12.6% on
EGFP and 10.1% on avGFP, against 66-80% on ubiquitin, lysozyme and adenylate kinase
(``msa_conservation/results/esm_calibration.csv``). A near-flat proposal filtered to top-k is
close to an arbitrary 10-residue alphabet, and the audit of the existing campaigns shows the
cost: 22.6% of all 51,731 position-edits placed a residue no aligned FP uses at that column, and
12.9% buried a formal charge at RSA < 0.05 (``msa_conservation/results/design_audit_*.csv``).

The 763-sequence family alignment is decisive exactly where ESM-2 is not, so this strategy uses
it as the generative model directly.

WHAT CHANGES
------------
1. PROPOSAL. ``esm_logits_at`` no longer runs ESM-2. It returns the Henikoff-weighted family
   log-frequency of each residue at that column (``msa_pssm_egfp.json``, built by
   ``build_msa_pssm.py``). The profile is position-specific but context-independent -- a PSSM,
   not a conditional -- so it is constant across iterations and free to evaluate.

2. HARD SUPPORT CONSTRAINT. A residue with weighted family frequency 0 is removed from the
   position's alphabet, so it can never be selected. This is folded into ``pos_allowed`` at
   trial-build time (intersected with the Tier-B aromatic/H-bond constraints), which keeps
   ``k_eff = min(k, |allowed|)`` honest and lets the engine's selection code run unmodified.
   Across the window this blocks 211 of 500 position-residue combinations; alphabets run from 4
   residues (L61 -> LIVM, the family's only options in 763 sequences) to 20 (I168, genuinely
   permissive). EGFP's own residue always survives -- it is in the alignment by construction --
   so the edit penalty can always choose to stay put.

3. LAMBDA SCALE. Every term in the guided score is z-scored across the k candidates, so each
   already has unit variance and lambda IS the relative weight. The inherited defaults did not
   reflect that: at lam_ex=lam_em=20, lam_bright=60, lam_edit=10, T=10 the effective weights
   were 0.1 / 2 / 2 / 6 / 1, i.e. the proposal term counted for 1/60th of brightness. Here
   every default is 1.0 at T=1.0, so

       score = z(logp_MSA) - 1.0*z(|d ex|) - 1.0*z(|d em|) + 1.0*z(bright) - 1.0*z(is_edit)

   weights all five terms equally, and the sweep varies them around that centre.

4. NO PSEUDO-PERPLEXITY. ESM-2 ppl is not this strategy's naturalness measure and costs about as
   much as a design iteration, so it is disabled and the column is left blank. The family
   log-likelihood of a design is a pure function of the sequence and the PSSM, so it can be
   computed from the CSVs afterwards if wanted.

CAVEAT worth keeping in view: z-scoring the proposal term normalizes away how *decided* the
family is at a position. L61 (84% Leu over 4 allowed residues) and I168 (20 allowed, nearly
flat) both contribute one unit of variance. The hard support mask is what carries the strength
of the prior; the z-scored term only carries its shape. This matches how the ESM strategies
treat their own proposal term, so the comparison stays like-for-like.

Nothing in ``fpdesign/`` is modified.

Usage
-----
    python design_campaign.py --trials 12 --iters 3 --pairs mOrange \
        --lam-ex 1 --lam-em 1 --lam-bright 1 --lam-edit 1
    bash run_sweep.sh          # the whole 3 x 3 x 3 grid, both targets
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent               # .../design-campaign-EGFP/msa-guided
CAMPAIGN = HERE.parent                                # .../design-campaign-EGFP
REPO = HERE.parents[1]                                # .../spectrum-to-fp-design
sys.path.insert(0, str(REPO))
from fpdesign.campaign import (Campaign, CampaignConfig, DEFAULT_BRIGHTNESS,  # noqa: E402
                               build_argparser)

PSSM_JSON = HERE / "msa_pssm_egfp.json"
NEG_INF = float("-inf")

BASE = CampaignConfig(
    name="egfp-msa-guided-tierB",
    strategy="guided",
    windows_json=HERE / "design_windows_egfp_tierB.json",
    pairs_csv=CAMPAIGN / "pairs" / "campaign_pairs_egfp.csv",
    # base dir only; cell_dir() appends the per-cell folder, as in lambda_sweep/ -- the engine's
    # own suffix omits lam_ex/lam_em, which would collide across peak levels.
    outdir=HERE / "designs",
    outdir_lambda_suffix=False,
    brightness_ckpt=DEFAULT_BRIGHTNESS,
    # T=1 with unit lambdas: every z-scored term carries the same weight (see point 3 above).
    default_temp=1.0, default_k=10,
    default_lam_ex=1.0, default_lam_em=1.0, default_lam_bright=1.0, default_lam_edit=1.0,
    add_lam_args=True, add_lam_bright_arg=True, add_lam_edit_arg=True, add_rescore=True,
    record_lambda=True, record_brightness=True,
    per_trial_rng=True, trial_resume=True,
    description="EGFP MSA-guided design: family-profile proposal + peaks + brightness + edit "
                "penalty, Tier-B window, all terms equally weighted at lambda=1.",
)


def cell_dir(args) -> Path:
    """One self-documenting folder per grid cell, carrying all four guidance weights."""
    return BASE.outdir / (f"lam-ex{args.lam_ex:g}_lam-em{args.lam_em:g}"
                          f"_lam-bright{args.lam_bright:g}_lam-edit{args.lam_edit:g}")


class MSACampaign(Campaign):
    """Guided campaign whose proposal distribution is the family profile, not ESM-2."""

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        blob = json.load(open(PSSM_JSON))
        pssm = blob["pssm"]
        V = len(self.alphabet.all_toks)
        # per position: a full-vocab log-prob row (-inf off the family support) and the matching
        # boolean mask. Precomputed once -- the profile has no sequence context to depend on.
        self.msa_logits, self.msa_mask = {}, {}
        for p_str, ent in pssm.items():
            p = int(p_str)
            row = torch.full((V,), NEG_INF, device=self.dev)
            mask = torch.zeros(V, dtype=torch.bool, device=self.dev)
            for aa, pr in zip(ent["alphabet"], ent["probs"]):
                idx = self.alphabet.get_idx(aa)
                row[idx] = float(torch.log(torch.tensor(pr)))
                mask[idx] = True
            self.msa_logits[p] = row
            self.msa_mask[p] = mask
        m = blob["meta"]
        sizes = [len(e["alphabet"]) for e in pssm.values()]
        print(f"MSA profile: {m['n_sequences']} sequences, N_eff {m['n_eff_henikoff']}, "
              f"{len(pssm)} window positions | alphabet size min {min(sizes)} "
              f"median {sorted(sizes)[len(sizes)//2]} max {max(sizes)} "
              f"({sum(sizes)}/{20*len(sizes)} residues allowed)", flush=True)
        print("proposal = family profile (ESM-2 masked-LM not used for selection); "
              f"T={args.temp:g} lam_ex={args.lam_ex:g} lam_em={args.lam_em:g} "
              f"lam_bright={args.lam_bright:g} lam_edit={args.lam_edit:g}", flush=True)

    def build_trials(self, pr, trial_start=0, trial_end=None):
        """Engine trials, with every window position's alphabet replaced by the family support.

        The Tier-B constraints were already intersected in when the PSSM was built, so this
        strictly tightens ``pos_allowed`` -- it never re-permits something the window forbids.
        """
        insts = super().build_trials(pr, trial_start=trial_start, trial_end=trial_end)
        for t in insts:
            missing = [p for p in t["editable"] if int(p) not in self.msa_mask]
            if missing:
                raise SystemExit(f"no family profile for editable positions {missing}; "
                                 f"re-run build_msa_pssm.py")
            t["pos_allowed"] = {int(p): self.msa_mask[int(p)] for p in t["editable"]}
        return insts

    @torch.no_grad()
    def esm_logits_at(self, seqlist, positions, bs=None):
        """Family log-frequencies in place of ESM-2 masked-LM logits.

        Same contract as the engine's version -- (N, vocab) scored at each requested position --
        so ``run_iteration`` and ``_select_guided`` are used unmodified. ``seqlist`` is ignored:
        a profile is context-independent, which is the substantive difference from ESM-2.
        """
        return torch.stack([self.msa_logits[int(p)] for p in positions])

    def ppl_batched(self, seqlist, bs=None):
        """ESM-2 pseudo-perplexity disabled (point 4 in the module docstring)."""
        return [float("nan")] * len(seqlist)


if __name__ == "__main__":
    args = build_argparser(BASE).parse_args()
    MSACampaign(replace(BASE, outdir=cell_dir(args)), args).run()
