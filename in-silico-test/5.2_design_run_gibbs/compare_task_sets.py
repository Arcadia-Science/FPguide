#!/usr/bin/env python
"""Task set 1 vs task set 2 -- does the guided arm's advantage survive ordinary targets?

Four arms, two per task set, each guided arm read against its OWN null:

    task 1 (furthest target)   3.2 ESM-2 guided (3 trials)   vs  3.3 Gibbs unguided (12 trials)
    task 2 (random target)     5.1 ESM-2 guided (3 trials)   vs  5.2 Gibbs unguided (12 trials)

Everything is identical between the two rows except which target each scaffold was paired with
(``2_design_task_specification/curate_pairs.py`` takes the most spectrally distant qualifying
target; ``4_design_task2/curate_pairs_task2.py`` draws uniformly from the same qualifying set).

WHY THE FRACTION MATTERS MORE THAN THE nm. Task 2's scaffolds start ~70 nm from their targets
where task 1's start ~133 nm, so absolute error is not comparable across rows -- an arm can look
better on task 2 purely by having less distance to cover. The comparable quantity is the SHARE of
the scaffold's initial error a run closes, and the comparable *guidance* quantity is the gap
between a guided arm and its own null, which is what this script pairs.

Within a task set the guided/null comparison is paired per task and tested with a Wilcoxon signed
rank; across task sets it is not paired (different tasks, mostly different scaffolds), so those
rows are reported side by side without a test.

Trial counts differ by arm (3 guided, 12 null), so the primary axis is the per-task MEAN over
trials -- the only statistic here that does not improve just by drawing more trials. The
surrogate-selected design (the rule the guided arms report) is shown alongside, and needs
``score_traj_surrogate.py`` to have been run for every arm.

Usage
-----
    for a in esm2_rand3 gibbs_r12 esm2_t2_rand3 gibbs_t2_r12; do
        python score_traj_surrogate.py --arm $a
    done
    python 5.2_design_run_gibbs/compare_task_sets.py
    python 5.2_design_run_gibbs/compare_task_sets.py --by-condition
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
from scipy.stats import wilcoxon

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT]

import design_common as C

TASK_SETS = [
    ("task 1 -- furthest target", [("3.2 ESM-2 guided", C.PIPE_OUT_ESM2_R3),
                                   ("3.3 Gibbs unguided", C.PIPE_OUT_GIBBS_R12)]),
    ("task 2 -- random target", [("5.1 ESM-2 guided", C.PIPE_OUT_ESM2_T2_R3),
                                 ("5.2 Gibbs unguided", C.PIPE_OUT_GIBBS_T2_R12)]),
]


def load(pipe):
    files = sorted(glob.glob(os.path.join(str(pipe), "*", "design_*.csv")))
    if not files:
        raise SystemExit(f"no design CSVs under {pipe}; run that arm first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cache = os.path.join(str(pipe), "surrogate_traj.csv")
    if not os.path.exists(cache):
        raise SystemExit(f"missing {cache}; run score_traj_surrogate.py for this arm first")
    return df.merge(pd.read_csv(cache), on=["example", "trial", "round"], how="left")


def summarize(df, examples):
    """Per-task statistics on the final cycle, restricted to `examples`."""
    df = df[df.example.isin(examples)]
    fin = df[df["round"] == df["round"].max()]
    g = fin.groupby("example")
    scaf = df[df["round"] == 0].groupby("example").peak_err.first()
    sel = fin.loc[g.surr_err.idxmin()].set_index("example").peak_err
    mean_tr = g.peak_err.mean()
    return dict(scaffold=scaf, mean_tr=mean_tr, sel=sel, best=g.peak_err.min(),
                spread=g.peak_err.agg(lambda s: s.max() - s.min()),
                ident=g.ident_to_scaffold.mean(),
                # share of the scaffold's own initial error closed -- the axis that survives the
                # two task sets starting at different distances
                frac=(1 - mean_tr / scaf.reindex(mean_tr.index)),
                improved=int(sum(sel[e] < scaf[e] for e in sel.index)))


def report(title, arms, examples):
    print(f"\n=== {title} (n={len(examples)}) ===")
    out = {}
    for label, df in arms:
        s = summarize(df, examples)
        out[label] = s
        print(f"{label:22s} scaffold={s['scaffold'].mean():6.1f}  mean-of-trials={s['mean_tr'].mean():6.1f}  "
              f"surr-sel={s['sel'].mean():6.1f}  oracle-best={s['best'].mean():6.1f}  "
              f"closed={s['frac'].mean():+.2f}  improved={s['improved']:3d}/{len(examples)}  "
              f"spread={s['spread'].mean():5.1f}  ident={s['ident'].mean():.3f}")
    (gl, gs), (nl, ns) = list(out.items())
    for axis in ("mean_tr", "sel"):
        x = gs[axis].sort_index(); y = ns[axis].reindex(x.index)
        print(f"  guidance ({axis:7s}): {(y - x).mean():+6.1f} nm | guided better on "
              f"{(y > x).sum():2d}/{len(x)} | Wilcoxon p={wilcoxon(x, y).pvalue:.2g}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-condition", action="store_true",
                    help="also break out S-pool (in the surrogate's training pool) vs S-test")
    args = ap.parse_args()

    results = {}
    for title, arms in TASK_SETS:
        loaded = [(label, load(pipe)) for label, pipe in arms]
        shared = sorted(set.intersection(*(set(df.example) for _, df in loaded)))
        results[title] = report(title, loaded, shared)
        if args.by_condition:
            coh = loaded[0][1].drop_duplicates("example").set_index("example").cohort
            for cond in C.CONDITIONS:
                ex = [e for e in shared if C.condition(coh[e]) == cond]
                if ex:
                    report(f"{title} | {C.CONDITION_LABEL[cond]}", loaded, ex)

    # the cross-task-set read: not paired (different tasks), so only the two summaries side by side
    print("\n=== task 1 vs task 2 (unpaired -- different tasks) ===")
    print(f"{'':26s} {'start':>7s} {'guided':>8s} {'null':>7s} {'guidance':>9s} {'closed(g)':>10s} {'closed(n)':>10s}")
    for title, out in results.items():
        (gl, gs), (nl, ns) = list(out.items())
        print(f"{title:26s} {gs['scaffold'].mean():7.1f} {gs['mean_tr'].mean():8.1f} "
              f"{ns['mean_tr'].mean():7.1f} {(ns['mean_tr'] - gs['mean_tr']).mean():+9.1f} "
              f"{gs['frac'].mean():+10.2f} {ns['frac'].mean():+10.2f}")
    print("\n(guided/null columns are the per-task mean over trials; `closed` is the mean share of "
          "the\n scaffold's own initial error closed, the axis comparable across the two task sets.)")


if __name__ == "__main__":
    main()
