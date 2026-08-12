#!/usr/bin/env python
"""Stage 5.1 -- the 3.2 ESM-2 guided arm, run on TASK SET 2 (a random qualifying target per
scaffold, from ``4_design_task2/curate_pairs_task2.py``).

Nothing about the search changes: same Tier-B windows, same deployed surrogate and oracle, same
ESM-2 650M masked-LM proposal, same ``z(logp_esm) - z(|ex_err|) - z(|em_err|)`` at 1/1/1, same
k = 10 / T = 1.0 / 2 cycles / 3 trials / random visit order / seed. The ONLY difference from 3.2
is the pair manifest, which is the point -- it makes 3.2 vs 5.1 a controlled read on whether the
guided arm's behaviour depends on its targets being the most spectrally distant available.

Cohorts are stage 4's two merged pools -- ``knownstruct_Spool`` (36, S-train + S-val: inside the
refit surrogate's training pool) and ``knownstruct_Stest`` (36, never trained on) -- so this arm
runs 72 tasks x 3 trials against 3.2's 108 x 3. Comparisons should be made on the conditions,
which are directly matched, not on the task counts.

This is a RUNNER, not a second implementation: it calls
``3.2_design_run_ESM2/design_knownstruct_esm2.py`` with the task-2 manifests and output root, so
the two task sets cannot drift apart in anything but their pairs. Extra arguments are passed
through and override the defaults below.

Usage
-----
    python 5.1_design_run_ESM2/run_task2_esm2.py
    python 5.1_design_run_ESM2/run_task2_esm2.py --smoke 2 --trials 1   # wiring probe
    python 5.1_design_run_ESM2/run_task2_esm2.py --trials 6            # more trials
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT]

import design_common as C

SCRIPT = os.path.join(_ROOT, "3.2_design_run_ESM2", "design_knownstruct_esm2.py")
# --no-ppl: pseudo-perplexity costs about a whole design cycle per round and is not used in any
# comparison here; fam_logp is free and still written, so the naturalness axis is intact.
DEFAULTS = ["--pairs-dir", str(C.PAIRS_DIR_T2),
            "--cohorts", *C.TASK2_COHORTS,
            "--outdir", str(C.PIPE_OUT_ESM2_T2_R3),
            "--no-ppl"]


def main():
    cmd = [sys.executable, SCRIPT, *DEFAULTS, *sys.argv[1:]]
    print("$ " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
