#!/usr/bin/env python3
"""Add seeds for a specified set of sweep configs (skips seeds already trained).

Targets one role's sweep (default: oracle / ProstT5; use --role surrogate for ESM-2), reusing
sweep_peak_oracle's data loader, protocol, and per-role output dir.
"""
import argparse
import os

import numpy as np
import torch

import peak_models as pm
import sweep_peak_oracle as swp

SPECS = [
    {"arch": "cnn", "pool": "max", "n_conv": 2, "depth": 2},
    {"arch": "cnn", "pool": "concat", "n_conv": 2, "depth": 2},
    {"arch": "cnn", "pool": "concat", "n_conv": 1, "depth": 1},
    {"arch": "cnn", "pool": "max", "n_conv": 3, "depth": 3},
    {"arch": "cnn", "pool": "max", "n_conv": 1, "depth": 1},
]
SEEDS = [0, 1, 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", default="oracle", choices=swp.ROLES,
                    help="which sweep's dir/embedding/split to use (default: oracle / ProstT5)")
    role = ap.parse_args().role

    dev = swp.device(); print(f"device: {dev} | role: {role} ({swp.ROLE_CFG[role]['emb']})", flush=True)
    D = swp.load_data(role, dev, to_gpu=True)
    OUT = swp.out_dir(role); os.makedirs(OUT, exist_ok=True)
    for spec in SPECS:
        for seed in SEEDS:
            path = os.path.join(OUT, f"{swp.label(spec)}_s{seed}.pt")
            if os.path.exists(path):
                print(f"[cached ] {swp.label(spec):16} s{seed}", flush=True); continue
            m, base = swp.train_eval(spec, seed, D, dev, 200, 20)
            pm.save_model(path, base, {**spec, "d_in": D["d_in"], "out": 2, "seed": seed,
                                       "role": role, "emb": D["emb"],
                                       "mean": D["mean"], "std": D["std"], **m})
            print(f"[trained] {swp.label(spec):16} s{seed}  val {m['val_mae']:.2f}  test {m['test_mae']:.2f}", flush=True)

    print("\n=== mean +/- std over 3 seeds ===", flush=True)
    rows = []
    for spec in SPECS:
        vs, ts = [], []
        for seed in SEEDS:
            ck = torch.load(os.path.join(OUT, f"{swp.label(spec)}_s{seed}.pt"),
                            map_location="cpu", weights_only=False)
            vs.append(ck["val_mae"]); ts.append(ck["test_mae"])
        vs, ts = np.array(vs), np.array(ts)
        rows.append((swp.label(spec), vs.mean(), vs.std(), ts.mean(), ts.std()))
    for lab, vm, vsd, tm, tsd in sorted(rows, key=lambda r: r[1]):
        print(f"{lab:16} val {vm:.2f} ± {vsd:.2f}   test {tm:.2f} ± {tsd:.2f}", flush=True)


if __name__ == "__main__":
    main()
