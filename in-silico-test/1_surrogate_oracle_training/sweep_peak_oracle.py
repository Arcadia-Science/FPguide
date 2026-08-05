#!/usr/bin/env python3
"""Architectural sweep for ex/em peak regression, run on the NESTED oracle/surrogate split
(``make_dual_split.py``: oracle 80/10/10 of the whole dataset, surrogate 70/15/15 carved
from oracle-train only) instead of the main pipeline's coordinated dual split.

Thin wrapper over ``sweep_peak_oracle_base.py`` (a local copy of the shared sweep
implementation, so this folder stays self-contained) -- identical grid, protocol and code, only
``CUR`` (-> ./data, the curated dataset + this folder's own dual_splits.csv) and ``OUT_BASE``
(-> ./trained_models) point here instead. See that file's docstring for the full protocol
description; not duplicated here.

Usage: identical CLI to the source script, e.g.
    python sweep_peak_oracle.py --role both --dry-run
    python sweep_peak_oracle.py --role surrogate --limit 3 --seeds 0   # smoke/timing probe
    python sweep_peak_oracle.py --role both --seeds 0                 # the real sweep
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # experiment root
# lib/ must be importable BEFORE the base module is exec'd below -- it does `import peak_models`.
sys.path[:0] = [HERE, os.path.join(HERE, "lib")]

# Loaded via an explicit spec rather than `import sweep_peak_oracle_base` only so the module
# object is available before its CUR/OUT_BASE are repointed below; the distinct sys.modules key
# also keeps this file (itself importable as "sweep_peak_oracle", e.g. from a notebook doing
# `import sweep_peak_oracle as swp`) from ever resolving to itself.
_spec = importlib.util.spec_from_file_location(
    "sweep_peak_oracle_base", os.path.join(HERE, "lib", "sweep_peak_oracle_base.py"))
base = importlib.util.module_from_spec(_spec)
sys.modules["sweep_peak_oracle_base"] = base
_spec.loader.exec_module(base)

base.CUR = os.path.join(HERE, "data")
base.OUT_BASE = os.path.join(HERE, "trained_models")
# load_data's `cur=CUR` default was bound to the OLD path at import time (Python evaluates
# default args once, at def time) -- rebinding base.CUR above does not retroactively change
# it, so the default must be patched explicitly too.
base.load_data.__defaults__ = (base.CUR, False)

# Re-exported so notebooks/scripts can `import sweep_peak_oracle as swp` here and call
# swp.load_data(...)/swp.device()/etc. against THIS folder's data+split.
device = base.device
load_data = base.load_data
make_configs = base.make_configs
label = base.label
out_dir = base.out_dir
ROLE_CFG = base.ROLE_CFG

if __name__ == "__main__":
    base.main()
