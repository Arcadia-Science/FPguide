#!/usr/bin/env python
"""Stage 3.2 -- the unguided Gibbs control, run on TASK SET 2 (a random qualifying target per
scaffold, from ``2_design_task_specification/curate_pairs_task2.py``).

3.1's null. Same windows, same ESM-2 proposal, same k = 10 / T = 1.0 / 2 cycles / random visit
order / seeding, with lam_ex = lam_em = 0 so the target spectrum never enters the search and the
surrogate is not loaded at all. 12 trials, as in the archived task-1 control, because this arm's
output is a distribution rather than a point.

It matters more here than it did on task 1, not less. Task 1's targets are 100-300 nm from
their scaffolds, so a search that merely resamples the pocket drifts toward the dataset's centre
of mass and closes 20% of the scaffold error for free. Task 2's targets are drawn uniformly and sit much
closer, which changes that free gain -- possibly its sign, since a design starting near its
target has room to wander AWAY from it. Without this arm, 3.1's numbers cannot be read at all.

Cohorts are stage 2's ``knownstruct_Spool`` (36) + ``knownstruct_Stest`` (36) -- the same 72
tasks 3.1 runs, so the two arms are directly paired (unlike the archived task-1 control, which is
a strict 72-of-108 subset of the task-1 guided arms).

This is a RUNNER, not a second implementation: it calls ``design_knownstruct_gibbs.py`` beside it
-- the same engine the archived task-1 control ran -- with the task-2 manifests and output root.
Extra arguments are passed through and override the defaults below.

Usage
-----
    python 3.2_design_run_gibbs/run_task2_gibbs.py
    python 3.2_design_run_gibbs/run_task2_gibbs.py --smoke 2 --trials 2   # wiring probe
    python 3.2_design_run_gibbs/run_task2_gibbs.py --lam-ex 1 --lam-em 1  # reproduces 3.1
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT]

import design_common as C

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "design_knownstruct_gibbs.py")
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
