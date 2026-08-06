#!/usr/bin/env python
"""Three-arm comparison on the tasks all three arms share -- reproduces the README's table.

3.1 and 3.2 ran 108 tasks x 3 trials; 3.3 ran the 72 S-train + S-test tasks x 12. Comparing the
published 108-task means against the null is therefore wrong, and wrong in a direction that
flatters the guided arms (S-val starts closer). Everything here is restricted to the shared 72
and recomputed, which is the whole reason this script exists rather than a note saying "be
careful". See the README's "3.3's cohorts are a strict subset" gotcha.

Reported per arm, on the final cycle:

  mean of trials    per-task mean over trials, then averaged -- the only trial-count-independent
                    statistic here, so the primary axis for 3-trial vs 12-trial arms
  surrogate-sel.    the trial whose SURROGATE error is lowest (the rule 3.1/3.2 report), from
                    ``score_traj_surrogate.py``'s cache -- run it for every arm first
  oracle-best       the trial whose ORACLE error is lowest: unobtainable, a bound, and strongly
                    favouring whichever arm drew more trials
  spread            within-task max - min: also grows with trial count, hence the matched
                    "first 3" row for the null

Usage
-----
    python score_traj_surrogate.py --arm msa_rand3      # once per arm, if not already cached
    python score_traj_surrogate.py --arm esm2_rand3
    python score_traj_surrogate.py --arm gibbs_r12
    python 3.3_design_run_gibbs/compare_arms.py
    python 3.3_design_run_gibbs/compare_arms.py --by-cohort
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd
from scipy.stats import wilcoxon

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT]

import design_common as C

# (label, pipeline dir, trials to keep) -- None keeps every trial the arm ran
ARMS = [("3.1 MSA PSSM (3 trials)", C.PIPE_OUT_R3, None),
        ("3.2 ESM-2 guided (3 trials)", C.PIPE_OUT_ESM2_R3, None),
        ("3.3 Gibbs unguided (12 trials)", C.PIPE_OUT_GIBBS_R12, None),
        ("3.3 Gibbs unguided (first 3)", C.PIPE_OUT_GIBBS_R12, 3)]


def load(pipe):
    files = sorted(glob.glob(os.path.join(str(pipe), "*", "design_*.csv")))
    if not files:
        raise SystemExit(f"no design CSVs under {pipe}; run that arm first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cache = os.path.join(str(pipe), "surrogate_traj.csv")
    if not os.path.exists(cache):
        raise SystemExit(f"missing {cache}; run score_traj_surrogate.py for this arm first")
    return df.merge(pd.read_csv(cache), on=["example", "trial", "round"], how="left")


def summarize(df, examples, n_trials=None):
    """Per-task statistics on the final cycle, restricted to `examples`."""
    df = df[df.example.isin(examples)]
    fin = df[df["round"] == df["round"].max()]
    if n_trials is not None:
        fin = fin[fin.trial < n_trials]
    g = fin.groupby("example")
    scaf = df[df["round"] == 0].groupby("example").peak_err.first()
    sel = fin.loc[g.surr_err.idxmin()].set_index("example").peak_err
    return dict(scaffold=scaf, mean_tr=g.peak_err.mean(), sel=sel, best=g.peak_err.min(),
                spread=g.peak_err.agg(lambda s: s.max() - s.min()),
                ident=g.ident_to_scaffold.mean(), fam=g.fam_logp.mean(),
                improved=int(sum(sel[e] < scaf[e] for e in sel.index)))


def report(loaded, examples, title):
    print(f"\n=== {title} (n={len(examples)}) ===")
    out = {}
    for label, pipe, nt in ARMS:
        s = summarize(loaded[str(pipe)], examples, nt)
        out[label] = s
        print(f"{label:32s} scaffold={s['scaffold'].mean():6.1f}  "
              f"mean-of-trials={s['mean_tr'].mean():6.1f}  surr-sel={s['sel'].mean():6.1f}  "
              f"oracle-best={s['best'].mean():6.1f}  improved={s['improved']:3d}/{len(examples)}  "
              f"spread={s['spread'].mean():5.1f}  ident={s['ident'].mean():.3f}  "
              f"fam_logp={s['fam'].mean():7.1f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-cohort", action="store_true", help="also break out S-train / S-test")
    args = ap.parse_args()

    loaded = {str(pipe): load(pipe) for _, pipe, _ in ARMS}
    null = loaded[str(C.PIPE_OUT_GIBBS_R12)]
    shared = set.intersection(*(set(df.example) for df in loaded.values()))
    print(f"{len(shared)} tasks shared by all arms "
          + ", ".join(f"{lab.split()[0]}={loaded[str(p)].example.nunique()}" for lab, p, _ in ARMS[:3]))

    R = report(loaded, shared, "all matched tasks")

    # paired tests on the two axes that mean anything: guided vs null
    print()
    guided = ["3.1 MSA PSSM (3 trials)", "3.2 ESM-2 guided (3 trials)"]
    nulls = ["3.3 Gibbs unguided (12 trials)", "3.3 Gibbs unguided (first 3)"]
    for axis in ("mean_tr", "sel"):
        for g in guided:
            for n in nulls:
                x = R[g][axis].sort_index(); y = R[n][axis].reindex(x.index)
                print(f"{n.split('(')[1][:-1]:>9s} null - {g.split()[0]} ({axis:7s}): "
                      f"{(y - x).mean():+6.1f} nm | guided better on {(y > x).sum():2d}/{len(x)} "
                      f"| Wilcoxon p={wilcoxon(x, y).pvalue:.2g}")

    if args.by_cohort:
        for coh in sorted(null.cohort.unique()):
            report(loaded, shared & set(null[null.cohort == coh].example), coh)


if __name__ == "__main__":
    main()
