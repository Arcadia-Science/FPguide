#!/usr/bin/env python
"""Is the selection softmax actually deciding anything at these lambdas, with an ESM-2 proposal?

``../msa-guided/check_scale.py`` asked this for the family-profile proposal and answered it (grid
centre 0.194 normalized entropy, corner 0.031, weakest cell 0.694, and the archived T=10 negative
control at 0.980). That evidence does NOT transfer to this effort, for a specific reason: the
selection score z-scores the proposal term, and ESM-2 is close to uninformative on this fold --
12.6% masked top-1 on EGFP against 66-80% on ordinary proteins of similar length
(``msa_conservation/results/esm_calibration.csv``). ``_zc(topv)`` divides a nearly flat spread of
top-k log-probs by a correspondingly tiny standard deviation, which rescales whatever structure is
there up to unit variance. Whether the resulting sampler is decided, degenerate, or effectively
random at lambda ~ 1 and T = 1 is an empirical question, and it is cheap to answer before spending
an hour and forty minutes of GPU time on the grid.

Same method as the MSA version: run one real design pass per setting and record the entropy of the
actual selection distribution softmax(scores/T), normalized against log(k_eff) so 1.0 is exactly
uniform over the k allowed candidates and 0.0 is a deterministic pick. Reports the mean normalized
entropy, the mean probability given to the chosen candidate, and the same entropy for the proposal
term alone (what ESM-2's top-k would give with no surrogate steering).

What to look for:
  * above ~0.9 at any grid cell -- that cell has no selection pressure and degenerates into random
    sampling from ESM-2's top-k, the failure mode ``../archive/README.md`` documents (22.6 of 25
    positions mutated, mOrange worse than the untouched scaffold).
  * below ~0.02 -- effectively argmax, so the cell contributes one deterministic trajectory per
    trial and extra trials buy only the any-order masking permutation.
Either is worth knowing per cell rather than discovering in the results.

The last two rows are reference points, not cells of this grid: they measure where strategy 4's
OWN sweep (``../lambda_sweep/``, lam 20/60/10 at T=10) and the archived unit-lambda-at-T=10
control actually put the same ESM-2 sampler.
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from design_campaign import BASE, SweepCampaign  # noqa: E402
from fpdesign.campaign import _zc, build_argparser  # noqa: E402

SETTINGS = [   # (lam_peaks, lam_bright, lam_edit, temp, label)
    (1.0, 1.0, 1.0, 1.0, "this grid, centre"),
    (4.0, 4.0, 4.0, 1.0, "this grid, corner"),
    (0.25, 0.0, 0.0, 1.0, "this grid, weakest cell (strategy 2)"),
    (2.0, 2.0, 0.0, 1.0, "this grid, where MSA winners sat"),
    (20.0, 60.0, 10.0, 10.0, "REF: strategy 4's own sweep centre"),
    (1.0, 1.0, 1.0, 10.0, "REF: archived negative control (T=10)"),
]


class ProbeCampaign(SweepCampaign):
    """Records the selection distribution at every visited position.

    An inline of ``fpdesign.campaign.Campaign._select_guided`` with the softmax captured before the
    draw. Kept in step with the engine by hand -- if the score formula there changes, this must
    change with it or the numbers below stop describing the real sampler.
    """

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.stats = []

    def _select_guided(self, sub, positions, logits):
        args = self.args
        cand_all, meta = [], []
        for i, t in enumerate(sub):
            pos = positions[i]
            mh = t["pos_allowed"].get(pos, self.AA_MASK)
            lg = logits[i].masked_fill(~mh, float("-inf"))
            logp = torch.log_softmax(lg, -1)
            k_eff = min(args.k, int(mh.sum().item()))
            topv, topi = torch.topk(logp, k_eff)
            aas = [self.alphabet.get_tok(int(x)) for x in topi.tolist()]
            for aa in aas:
                cand_all.append(t["seq"][:pos] + aa + t["seq"][pos + 1:])
            meta.append((t, pos, topv, aas, k_eff))
        Pk, Bk = self.peaks_and_brightness_batched(cand_all)
        off = 0
        for (t, pos, topv, aas, k_eff) in meta:
            sl = slice(off, off + k_eff); off += k_eff
            Pc = Pk[sl]
            ex_err = (Pc[:, 0] - t["tgt"][0]).abs(); em_err = (Pc[:, 1] - t["tgt"][1]).abs()
            scores = _zc(topv) - args.lam_ex * _zc(ex_err) - args.lam_em * _zc(em_err)
            if Bk is not None:
                scores = scores + args.lam_bright * _zc(Bk[sl, 0])
            if args.lam_edit:
                scaf_aa = t["scaffold"][pos]
                is_edit = torch.tensor([0.0 if aa == scaf_aa else 1.0 for aa in aas],
                                       device=scores.device)
                scores = scores - args.lam_edit * _zc(is_edit)
            p = torch.softmax(scores / args.temp, -1)
            p_prop = torch.softmax(topv, -1)              # ESM-2 top-k alone, unmodified
            if k_eff > 1:
                lk = float(np.log(k_eff))
                ent = float(-(p * torch.log(p.clamp_min(1e-12))).sum()) / lk
                ent_prop = float(-(p_prop * torch.log(p_prop.clamp_min(1e-12))).sum()) / lk
                self.stats.append((k_eff, ent, float(p.max()), ent_prop))
            ch = int(torch.multinomial(p, 1, generator=t["gen"]).item())
            t["seq"] = t["seq"][:pos] + aas[ch] + t["seq"][pos + 1:]


def main():
    print(f"{'setting':38s} {'lam(p/b/e)':>12s} {'T':>4s}  {'H/Hmax':>7s} {'max p':>7s} "
          f"{'ESM-2 H/Hmax':>15s}")
    print("-" * 92)
    for lam_p, lam_b, lam_e, temp, label in SETTINGS:
        argv = ["--trials", "4", "--iters", "1", "--pairs", "mOrange",
                "--lam-ex", str(lam_p), "--lam-em", str(lam_p),
                "--lam-bright", str(lam_b), "--lam-edit", str(lam_e), "--temp", str(temp)]
        args = build_argparser(BASE).parse_args(argv)
        c = ProbeCampaign(replace(BASE, outdir=HERE / "_probe"), args)
        insts = c.build_trials(c.units[0])
        c.run_iteration(insts)
        k, ent, mx, entp = map(np.array, zip(*c.stats))
        print(f"{label:38s} {lam_p:g}/{lam_b:g}/{lam_e:g}".ljust(52)
              + f"{temp:4g}  {ent.mean():7.3f} {mx.mean():7.3f} {entp.mean():15.3f}")
    print("\n1.000 = uniform over the k allowed candidates (no selection pressure); "
          "0.000 = deterministic.")
    print("'ESM-2 H/Hmax' is the masked-LM top-k on its own, before any surrogate steering; it is "
          "the same\nin every row and is the quantity ../msa-guided/ replaces with a family "
          "profile.")


if __name__ == "__main__":
    main()
