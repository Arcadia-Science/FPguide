#!/usr/bin/env python
"""Is the selection softmax actually deciding anything at lam=1, T=1 in THIS experiment?

The design-campaign-EGFP MSA-guided strategy this pipeline inherits its scale from documents a
negative control worth not repeating (``design-campaign-EGFP/msa-guided/check_scale.py``, and
``design-campaign-EGFP/archive/README.md``): at lam=1 with T=10 every z-scored term contributes
~0.1 of a unit, the softmax over the k candidates lands ~96% of the way to uniform, and the
search degenerates into random sampling from the proposal's top-k. That campaign's own check
measures H/Hmax = 0.194 at its grid centre, i.e. comfortably decided.

Its score has FIVE z-scored terms (profile + ex + em + brightness + edit penalty); ours has
THREE (profile + ex + em), so the score's spread across candidates is smaller at the same lambda
and the softmax is correspondingly softer. Whether that still lands in the decided regime is an
empirical question, which is what this script answers -- it re-runs the real selection step of
``design_knownstruct.py`` (same window, same PSSM, same surrogate, same fixed visit order) and
records the entropy of the actual selection distribution softmax(scores/T), normalized by
log(k_eff) so 1.0 is exactly uniform over the allowed candidates and 0.0 is deterministic.

Usage
-----
    python check_scale.py                  # 6 tasks, 1 round, the settings table below
    python check_scale.py --tasks 12
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "lib")]

import design_common as C
import peak_models as pm

K_TOP = 10
ESM_BS = 48

SETTINGS = [   # (lam_ex = lam_em, temp, label)
    (1.0, 1.0, "this pipeline (design_knownstruct.py)"),
    (2.0, 1.0, "sharper peak weight"),
    (1.0, 10.0, "the EGFP campaign's archived negative control"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=6, help="how many tasks to probe (default 6)")
    args = ap.parse_args()

    WINJ = json.load(open(C.WINDOWS_JSON))["windows"]
    d = C.load_dataset()
    rows, seqs, peaks = d["rows"], d["seqs"], d["peaks"]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = (dev.type == "cuda")
    AMP = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_fp16 else (lambda: nullcontext())

    _sb, s_meta = pm.load_model(C.SURR_CKPT, dev)
    surrogate = pm.wrap(_sb, s_meta["mean"], s_meta["std"], dev)
    esm_model, alphabet, bc = pm.get_esm(dev)
    V = len(alphabet.all_toks)

    @torch.no_grad()
    def surrogate_peaks(seqlist, bs=ESM_BS):
        outs = []
        for i in range(0, len(seqlist), bs):
            ch = seqlist[i:i + bs]
            _, _, tk = bc([(f"s{j}", s) for j, s in enumerate(ch)])
            with AMP():
                reps = esm_model(tk.to(dev), repr_layers=[pm.ESM_LAYER])["representations"][pm.ESM_LAYER]
            Lmax = max(len(s) for s in ch)
            H = torch.zeros(len(ch), Lmax, pm.D_IN, device=dev)
            mask = torch.zeros(len(ch), Lmax, dtype=torch.bool, device=dev)
            for j, s in enumerate(ch):
                n = len(s)
                H[j, :n] = reps[j, 1:1 + n].float()
                mask[j, :n] = True
            outs.append(surrogate(H, mask))
        return torch.cat(outs, 0)

    def _zc(t):
        return (t - t.mean()) / (t.std() + 1e-6)

    # ---- the same tasks the pipeline runs, first --tasks of the test cohort ----------------
    pairs = list(csv.DictReader(open(C.pairs_csv_path("knownstruct_Stest"))))[:args.tasks]
    base = []
    for r in pairs:
        si, ti = int(r["scaffold_idx"]), int(r["target_idx"])
        w = WINJ[rows[si]["name"]]
        c1, p2 = w["chromophore"]["pos1_0based"], w["chromophore"]["pos2_0based"]
        pos_allowed, pssm_vec = {}, {}
        for p_str, ent in w["pssm"].items():
            p = int(p_str)
            mask = torch.zeros(V, dtype=torch.bool)
            vec = torch.full((V,), float("-inf"))
            for aa, pr in zip(ent["alphabet"], ent["probs"]):
                idx = alphabet.get_idx(aa)
                mask[idx] = True
                vec[idx] = float(np.log(max(pr, 1e-12)))
            pos_allowed[p] = mask.to(dev)
            pssm_vec[p] = vec.to(dev)
        base.append(dict(seq=seqs[si], scaffold=seqs[si], editable=list(w["editable_0based"]),
                         c1=c1, p2=p2, pos_allowed=pos_allowed, pssm_vec=pssm_vec,
                         tgt=torch.tensor(peaks[ti], device=dev)))
    print(f"probing {len(base)} tasks | k={K_TOP} | surrogate {C.SURR_CKPT.name}")
    print(f"{'setting':44s} {'lam':>5s} {'T':>4s}  {'H/Hmax':>7s} {'p(chosen)':>10s} {'profile H/Hmax':>15s}")
    print("-" * 92)

    for lam, temp, label in SETTINGS:
        torch.manual_seed(C.SEED)
        T = [dict(t, seq=t["scaffold"]) for t in base]     # every setting starts from the scaffold
        stats = []
        # one design cycle, exactly design_knownstruct.py's PHASE_PLAN: chromophore pos 1-2 first,
        # then the rest of the pocket, both in fixed positional order (no random permutation).
        for nums, use_pocket in [({1, 2}, False), (set(), True)]:
            for t in T:
                t["_ed"] = ([{1: t["c1"], 2: t["p2"]}[n] for n in sorted(nums)] +
                            ([p for p in t["editable"] if p not in (t["c1"], t["p2"])] if use_pocket else []))
            for j in range(max(len(t["_ed"]) for t in T)):
                sub = [t for t in T if j < len(t["_ed"])]
                positions = [t["_ed"][j] for t in sub]
                cand_all, meta = [], []
                for i, t in enumerate(sub):
                    pos = positions[i]
                    mh = t["pos_allowed"][pos]
                    lg = t["pssm_vec"][pos].masked_fill(~mh, float("-inf"))
                    logp = torch.log_softmax(lg, -1)
                    k_eff = min(K_TOP, int(mh.sum().item()))
                    topv, topi = torch.topk(logp, k_eff)
                    aas = [alphabet.get_tok(int(x)) for x in topi.tolist()]
                    for aa in aas:
                        cand_all.append(t["seq"][:pos] + aa + t["seq"][pos + 1:])
                    meta.append((t, pos, topv, aas, k_eff))
                Pk = surrogate_peaks(cand_all)
                off = 0
                for (t, pos, topv, aas, k_eff) in meta:
                    Pc = Pk[off:off + k_eff]; off += k_eff
                    ex_err = (Pc[:, 0] - t["tgt"][0]).abs(); em_err = (Pc[:, 1] - t["tgt"][1]).abs()
                    scores = _zc(topv) - lam * _zc(ex_err) - lam * _zc(em_err)
                    p = torch.softmax(scores / temp, -1)
                    p_prop = torch.softmax(topv, -1)        # family profile alone, no steering
                    if k_eff > 1:
                        lk = float(np.log(k_eff))
                        stats.append((float(-(p * torch.log(p.clamp_min(1e-12))).sum()) / lk,
                                      float(p.max()),
                                      float(-(p_prop * torch.log(p_prop.clamp_min(1e-12))).sum()) / lk))
                    ch = int(torch.multinomial(p, 1).item())
                    t["seq"] = t["seq"][:pos] + aas[ch] + t["seq"][pos + 1:]
        ent, mx, entp = map(np.array, zip(*stats))
        print(f"{label:44s} {lam:5g} {temp:4g}  {ent.mean():7.3f} {mx.mean():10.3f} {entp.mean():15.3f}"
              f"   (n={len(stats)})")

    print("\n1.000 = uniform over the k allowed candidates (no selection pressure); 0.000 = deterministic.")
    print("'profile H/Hmax' is the family profile on its own, before any surrogate steering.")
    print("EGFP campaign reference (5 terms, lam=1, T=1): H/Hmax 0.194, p(chosen) 0.868; "
          "its T=10 control: 0.980 / 0.189.")


if __name__ == "__main__":
    main()
