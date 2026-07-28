#!/usr/bin/env python
"""EGFP MSA-Gibbs design campaign (Tier-B window): unguided sampling from the family profile.

This is to [`../msa-guided/`](../msa-guided/) what [`../gibbs-sampling/`](../gibbs-sampling/) is to
the ESM-based guided strategies: **the same proposal distribution with the surrogate steering
removed**. Together the four make a 2x2 that isolates the two things that actually differ between
strategies in this campaign.

|                | unguided (gibbs)   | surrogate-guided      |
|----------------|--------------------|-----------------------|
| ESM-2 proposal | `gibbs-sampling/`  | `lambda_sweep/` etc.  |
| family profile | **this folder**    | `msa-guided/`         |

WHAT IT DOES
------------
Target-free, exactly like `gibbs-sampling/`: the search never sees a target, so one run is a single
design effort per scaffold, not one per pair, and it writes the target-free `COLS_FREE` schema. At
each visited position the residue is drawn from `softmax(top-k family log-frequency / T)` at T=1 --
the family analogue of a Gibbs draw from the masked-LM conditional. The surrogate is loaded only to
record each round's (ex, em) and is never consulted during selection.

Shares with `msa-guided/`:
  * the SAME PSSM file, `../msa-guided/msa_pssm_egfp.json`, referenced rather than copied so the
    two strategies provably sample the same distribution (the whole point of the comparison);
  * the hard support constraint -- zero family frequency means the residue is unselectable;
  * the Tier-B window (local copy, byte-identical to every other strategy's).

ONE PROPERTY TO UNDERSTAND BEFORE READING THE RESULTS
-----------------------------------------------------
A PSSM is **context-independent**. ESM-2's conditional changes as the sequence is edited, so its
Gibbs chain genuinely mixes and tends to stay near a self-consistent sequence; the family profile
does not move at all. Every iteration therefore re-draws each editable position independently from
the same fixed distribution, and the chain has no memory: round 3 is one independent sample from
the profile, not a refinement of round 2.

That makes this a clean sample of "what the family profile alone proposes for this window", which
is the baseline worth having -- but it also means the mutation load is set by the profile rather
than by the scaffold, and will be much higher than ESM-2 gibbs, which is pulled toward the sequence
it is conditioning on. `--lam-edit` is exposed (engine default 0) if you want the opt-in penalty
toward the scaffold residue; at 0 this is an exact draw from the profile's top-k.

No brightness term: the engine only supports `brightness_ckpt` with strategy="guided", and steering
is precisely what this control removes. `pred_bright` is therefore absent from the CSV and is
scored afterwards by the same `cnn-max-d2` classifier the other strategies use, exactly as is
already done for ESM gibbs.

Pseudo-perplexity is disabled for the same reason as in `msa-guided/`: ESM-2 ppl is not this
strategy's naturalness measure, and the family log-likelihood under the PSSM is recoverable from
the CSV without a GPU.

Nothing in ``fpdesign/`` is modified.

Usage
-----
    python design_campaign.py --trials 96 --iters 3
    bash run_campaign.sh --trials 96
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent               # .../design-campaign-EGFP/msa-gibbs
CAMPAIGN = HERE.parent                                # .../design-campaign-EGFP
REPO = HERE.parents[1]                                # .../spectrum-to-fp-design
sys.path.insert(0, str(REPO))
from fpdesign.campaign import Campaign, CampaignConfig, build_argparser  # noqa: E402

# Referenced, NOT copied: msa-guided and msa-gibbs must sample the identical profile for the
# guided-vs-unguided comparison to mean anything. Rebuild it with msa-guided/build_msa_pssm.py.
PSSM_JSON = CAMPAIGN / "msa-guided" / "msa_pssm_egfp.json"
NEG_INF = float("-inf")

CFG = CampaignConfig(
    name="egfp-msa-gibbs-tierB",
    strategy="gibbs",
    windows_json=HERE / "design_windows_egfp_tierB.json",
    pairs_csv=CAMPAIGN / "pairs" / "campaign_pairs_egfp.csv",
    outdir=HERE / "designs",
    # T=1 over the top-k family log-frequencies == a draw from the profile restricted to top-k,
    # matching gibbs-sampling/'s T=1 k=10 over the ESM-2 conditional.
    default_temp=1.0, default_k=10,
    add_lam_args=False, add_rescore=False, record_lambda=False,
    add_lam_edit_arg=True, default_lam_edit=0.0,   # opt-in; 0 = pure profile draw
    trial_resume=True,
    target_free=True,   # the profile ignores the target -> one target-free CSV per scaffold
    description="EGFP Tier-B MSA-Gibbs design (target-free): unguided sampling from the family "
                "profile; surrogate diagnostic only.",
)


class MSAGibbsCampaign(Campaign):
    """Gibbs campaign whose conditional is the family profile instead of ESM-2's masked LM."""

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        blob = json.load(open(PSSM_JSON))
        pssm = blob["pssm"]
        V = len(self.alphabet.all_toks)
        # per position: a full-vocab log-prob row (-inf off the family support) + its boolean mask.
        # Precomputed once; a profile has no sequence context to depend on.
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
        print(f"UNGUIDED: draw ~ softmax(top-{args.k} family log-freq / T), T={args.temp:g}, "
              f"lam_edit={getattr(args, 'lam_edit', 0.0):g}; surrogate records (ex, em) only",
              flush=True)

    def build_trials(self, pr, trial_start=0, trial_end=None):
        """Engine trials, with every window position's alphabet replaced by the family support.

        Tier-B was already intersected in when the PSSM was built, so this strictly tightens
        ``pos_allowed`` and never re-permits something the window forbids.
        """
        insts = super().build_trials(pr, trial_start=trial_start, trial_end=trial_end)
        for t in insts:
            missing = [p for p in t["editable"] if int(p) not in self.msa_mask]
            if missing:
                raise SystemExit(f"no family profile for editable positions {missing}; "
                                 f"re-run ../msa-guided/build_msa_pssm.py")
            t["pos_allowed"] = {int(p): self.msa_mask[int(p)] for p in t["editable"]}
        return insts

    @torch.no_grad()
    def esm_logits_at(self, seqlist, positions, bs=None):
        """Family log-frequencies in place of ESM-2 masked-LM logits.

        Same contract as the engine's version -- (N, vocab) scored at each requested position -- so
        ``run_iteration`` and ``_select_gibbs`` run unmodified. ``seqlist`` is ignored: a profile is
        context-independent, which is the substantive difference from ESM-2 (see the module
        docstring).
        """
        return torch.stack([self.msa_logits[int(p)] for p in positions])

    def ppl_batched(self, seqlist, bs=None):
        """ESM-2 pseudo-perplexity disabled; the PSSM log-likelihood is the analogue here."""
        return [float("nan")] * len(seqlist)


if __name__ == "__main__":
    MSAGibbsCampaign(CFG, build_argparser(CFG).parse_args()).run()
