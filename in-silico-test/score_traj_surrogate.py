#!/usr/bin/env python
"""Post-hoc: score every trajectory sequence with the SURROGATE, for the notebook viewers.

A design run logs only the ORACLE prediction per round (``pred_ex``/``pred_em``/
``peak_err``): the surrogate is consulted inside a round, on the k=10 single-mutation candidates
at each editable position, and only the accepted candidate's sequence survives -- its surrogate
prediction is never written out. So the trajectory CSVs cannot show what the search *thought*
it was doing, only what the held-out judge says it did.

This script fills that gap by re-running the same surrogate (``C.SURR_CKPT``, the CV-selected
cnn-max-d1 refit) over each round's ``designed_seq`` and caching the result next to the design
CSVs as ``surrogate_traj.csv``. Cheap: ~n_tasks x (1 + n_rounds) sequences, one ESM-2 pass each.

Read alongside the oracle curve, the surrogate curve is the surrogate/oracle DISAGREEMENT per
task -- the gap is the model error the search is actually flying blind on, and a surrogate
error that falls while the oracle error rises is the search exploiting its own scorer.

Multi-trial runs are handled: when the design CSVs carry a ``trial`` column, every trajectory is
scored and the cache is keyed by (example, trial, round), so a caller merges on all three.

Usage
-----
    python score_traj_surrogate.py --arm esm2_t2_rand3   # task 2 guided (3.1)     -> PIPE_OUT_ESM2_T2_R3
    python score_traj_surrogate.py --arm gibbs_t2_r12    # task 2 null (3.2)       -> PIPE_OUT_GIBBS_T2_R12
    python score_traj_surrogate.py                       # = --arm esm2_t2_rand3 (the default)

    # the archived task-1 arms, whose CSVs are still on disk:
    python score_traj_surrogate.py --arm msa_rand3       # task-1 MSA, 3 trials    -> PIPE_OUT_R3
    python score_traj_surrogate.py --arm esm2_rand3      # task-1 ESM-2, 3 trials  -> PIPE_OUT_ESM2_R3
    python score_traj_surrogate.py --arm gibbs_r12       # task-1 unguided control -> PIPE_OUT_GIBBS_R12
    python score_traj_surrogate.py --force               # recompute even if the cache exists
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "lib")]

import design_common as C
import peak_models as pm

CACHE_NAME = "surrogate_traj.csv"
ESM_BS = 32


# The live arms are the task-2 pair; the rest are the archived task-1 runs, whose design CSVs
# moved to archive/peak_designs/ with the code -- archive/'s script + notebooks still read them
# through the constants below (archive/README.md).
ARMS = {# task set 2 (random target per scaffold): 3.1 guided, 3.2 unguided null
        "esm2_t2_rand3": C.PIPE_OUT_ESM2_T2_R3, "gibbs_t2_r12": C.PIPE_OUT_GIBBS_T2_R12,
        # --- archived task set 1 ---
        "msa": C.PIPE_OUT, "pssm": C.PIPE_OUT,          # "pssm" kept as an alias for "msa"
        "esm2": C.PIPE_OUT_ESM2,
        "msa_rand3": C.PIPE_OUT_R3, "esm2_rand3": C.PIPE_OUT_ESM2_R3,
        # the unguided control never consulted the surrogate during the search, so here the
        # surrogate curve is not "what the search thought" but a counterfactual: what it would
        # have said about sequences chosen without it -- and the axis a surrogate-selected
        # best-of-12 has to be computed on, to price trial selection against the guided arms
        "gibbs_r12": C.PIPE_OUT_GIBBS_R12}


def pipe_dir(arm):
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; choose from {sorted(set(ARMS))}")
    return str(ARMS[arm])


def cache_path(arm):
    return os.path.join(pipe_dir(arm), CACHE_NAME)


def compute(arm="msa", force=False, verbose=True):
    """Return the surrogate-per-round table, computing + caching it on first call."""
    out_fn = cache_path(arm)
    if os.path.exists(out_fn) and not force:
        return pd.read_csv(out_fn)

    files = sorted(glob.glob(os.path.join(pipe_dir(arm), "*", "design_*.csv")))
    if not files:
        raise SystemExit(f"no design CSVs under {pipe_dir(arm)}; run the design stage first")
    traj = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # `trial` is absent in the single-trial first pass and present in the 3-trial reruns; carry it
    # through when it is there so the cache keys each trajectory separately
    keys = ["example"] + (["trial"] if "trial" in traj.columns else []) + ["round"]
    traj = traj[keys + ["designed_seq", "target_ex", "target_em"]].copy()

    dev = (torch.device("cuda") if torch.cuda.is_available()
           else torch.device("mps") if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
           else torch.device("cpu"))
    use_fp16 = (dev.type == "cuda")
    AMP = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_fp16 else (lambda: nullcontext())
    if verbose:
        print(f"device {dev} | fp16 {use_fp16} | {len(traj)} sequences from {len(files)} tasks")

    _sb, s_meta = pm.load_model(C.SURR_CKPT, dev)
    surrogate = pm.wrap(_sb, s_meta["mean"], s_meta["std"], dev)
    esm_model, alphabet, bc = pm.get_esm(dev)

    @torch.no_grad()
    def surrogate_peaks(seqlist, bs=ESM_BS):
        # same embedding path the design run used: ESM-2 layer-33 residues -> surrogate, fp16 AMP
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
            outs.append(surrogate(H, mask).float().cpu())
        return torch.cat(outs, 0).numpy()

    t0 = time.time()
    P = surrogate_peaks(list(traj["designed_seq"]))
    traj["surr_ex"] = np.round(P[:, 0], 1)
    traj["surr_em"] = np.round(P[:, 1], 1)
    # same error definition the design run logs for the oracle: mean of the two absolute offsets
    traj["surr_err"] = np.round(
        0.5 * ((traj.surr_ex - traj.target_ex).abs() + (traj.surr_em - traj.target_em).abs()), 2)
    out = traj[keys + ["surr_ex", "surr_em", "surr_err"]]
    out.to_csv(out_fn, index=False)
    if verbose:
        print(f"wrote {out_fn} ({len(out)} rows) | {time.time()-t0:.0f}s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(set(ARMS)), default="esm2_t2_rand3",
                    help="which design run to score; default is stage 3.1, the live guided arm")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    compute(args.arm, force=args.force)
