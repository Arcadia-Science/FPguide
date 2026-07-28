#!/usr/bin/env python
"""Is the selection softmax actually deciding anything at these lambdas?

``archive/README.md`` records a negative control worth not repeating: at lam=1 with T=10 every
z-scored term contributes ~0.1 of a unit, the softmax over the k candidates lands ~96% of the way
to uniform, and the "search" degenerates into random sampling from the proposal's top-k (22.6 of
25 positions mutated, mOrange worse than the untouched scaffold).

This campaign also uses lam=1, but at T=1 -- ten times sharper. That is an argument, not evidence,
so this script measures it: it runs one real design pass per requested lambda setting and records
the entropy of the actual selection distribution softmax(scores/T), normalized against log(k_eff)
so 1.0 is exactly uniform and 0.0 is a deterministic pick.

Reports for each setting the mean normalized entropy and the mean probability given to the chosen
candidate, plus the same numbers for the proposal term alone (what the family profile would give
on its own, with no surrogate steering).
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from design_campaign import BASE, MSACampaign  # noqa: E402
from fpdesign.campaign import _zc, build_argparser  # noqa: E402

SETTINGS = [   # (lam_peaks, lam_bright, lam_edit, temp, label)
    (1.0, 1.0, 1.0, 1.0, "this campaign, grid centre"),
    (4.0, 4.0, 4.0, 1.0, "this campaign, grid corner"),
    (0.5, 0.0, 0.0, 1.0, "this campaign, weakest cell"),
    (1.0, 1.0, 1.0, 10.0, "the archived negative control (T=10)"),
]


class ProbeCampaign(MSACampaign):
    """Records the selection distribution at every visited position."""

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
            p_prop = torch.softmax(topv, -1)              # family profile alone, unmodified
            if k_eff > 1:
                lk = float(np.log(k_eff))
                ent = float(-(p * torch.log(p.clamp_min(1e-12))).sum()) / lk
                ent_prop = float(-(p_prop * torch.log(p_prop.clamp_min(1e-12))).sum()) / lk
                self.stats.append((k_eff, ent, float(p.max()), ent_prop))
            ch = int(torch.multinomial(p, 1, generator=t["gen"]).item())
            t["seq"] = t["seq"][:pos] + aas[ch] + t["seq"][pos + 1:]


def main():
    print(f"{'setting':38s} {'lam(p/b/e)':>12s} {'T':>4s}  {'H/Hmax':>7s} {'max p':>7s} "
          f"{'profile H/Hmax':>15s}")
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
    print("'profile H/Hmax' is the family profile on its own, before any surrogate steering.")


if __name__ == "__main__":
    main()
