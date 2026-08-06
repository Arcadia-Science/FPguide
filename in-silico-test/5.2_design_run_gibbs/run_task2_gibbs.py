#!/usr/bin/env python
"""Stage 5.2 -- the 3.3 unguided Gibbs control, run on TASK SET 2 (a random qualifying target per
scaffold, from ``4_design_task2/curate_pairs_task2.py``).

5.1's null. Same windows, same ESM-2 proposal, same k = 10 / T = 1.0 / 2 cycles / random visit
order / seeding, with lam_ex = lam_em = 0 so the target spectrum never enters the search and the
surrogate is not loaded at all. 12 trials, as in 3.3, because this arm's output is a distribution
rather than a point.

It matters more here than in 3.3, not less. Task 1's targets are 100-300 nm from their scaffolds,
so a search that merely resamples the pocket drifts toward the dataset's centre of mass and
closes 20% of the scaffold error for free. Task 2's targets are drawn uniformly and sit much
closer, which changes that free gain -- possibly its sign, since a design starting near its
target has room to wander AWAY from it. Without this arm, 5.1's numbers cannot be read at all.

Cohorts are stage 4's ``knownstruct_Spool`` (36) + ``knownstruct_Stest`` (36) -- the same 72
tasks 5.1 runs, so the two arms are directly paired (unlike 3.3, which is a strict 72-of-108
subset of 3.1/3.2).

This is a RUNNER, not a second implementation: it calls
``3.3_design_run_gibbs/design_knownstruct_gibbs.py`` with the task-2 manifests and output root.
Extra arguments are passed through and override the defaults below.

Usage
-----
    python 5.2_design_run_gibbs/run_task2_gibbs.py
    python 5.2_design_run_gibbs/run_task2_gibbs.py --smoke 2 --trials 2   # wiring probe
    python 5.2_design_run_gibbs/run_task2_gibbs.py --lam-ex 1 --lam-em 1  # reproduces 5.1
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT]

import design_common as C

SCRIPT = os.path.join(_ROOT, "3.3_design_run_gibbs", "design_knownstruct_gibbs.py")
DEFAULTS = ["--pairs-dir", str(C.PAIRS_DIR_T2),
            "--cohorts", *C.TASK2_COHORTS,
            "--outdir", str(C.PIPE_OUT_GIBBS_T2_R12),
            "--no-ppl"]


def main():
    cmd = [sys.executable, SCRIPT, *DEFAULTS, *sys.argv[1:]]
    print("$ " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
