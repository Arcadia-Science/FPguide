#!/usr/bin/env python3
"""Confirm the top-3 sweep configs at 3 seeds (reuses seed-0 checkpoints; trains seeds 1-2).

Targets one role's sweep (default: oracle / ProstT5; use --role surrogate for ESM-2). Saves all
checkpoints into that role's sweep dir and writes results_top3.csv with mean +/- std over seeds.
Does NOT touch the full-sweep results.csv.
"""
import argparse
import csv
import os

import numpy as np
import torch

import peak_models as pm
import sweep_peak_oracle as swp

TOP = [
    {"arch": "cnn", "pool": "concatstd", "n_conv": 1, "depth": 1},
    {"arch": "cnn", "pool": "concatstd", "n_conv": 2, "depth": 2},
    {"arch": "cnn", "pool": "concat", "n_conv": 3, "depth": 3},
]
SEEDS = [0, 1, 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", default="oracle", choices=swp.ROLES,
                    help="which sweep's dir/embedding/split to use (default: oracle / ProstT5)")
    role = ap.parse_args().role

    dev = swp.device(); print(f"device: {dev} | role: {role} ({swp.ROLE_CFG[role]['emb']})", flush=True)
    OUT = swp.out_dir(role); os.makedirs(OUT, exist_ok=True)
    D = swp.load_data(role, dev, to_gpu=True)
    for spec in TOP:
        for seed in SEEDS:
            path = os.path.join(OUT, f"{swp.label(spec)}_s{seed}.pt")
            if os.path.exists(path):
                print(f"[cached ] {swp.label(spec):20} s{seed}", flush=True); continue
            m, base = swp.train_eval(spec, seed, D, dev, 200, 20)
            pm.save_model(path, base, {**spec, "d_in": D["d_in"], "out": 2, "seed": seed,
                                       "role": role, "emb": D["emb"],
                                       "mean": D["mean"], "std": D["std"], **m})
            print(f"[trained] {swp.label(spec):20} s{seed}  val {m['val_mae']:.2f}  test {m['test_mae']:.2f}", flush=True)

    rows = []
    for spec in TOP:
        vs, ts = [], []
        for seed in SEEDS:
            ck = torch.load(os.path.join(OUT, f"{swp.label(spec)}_s{seed}.pt"),
                            map_location="cpu", weights_only=False)
            vs.append(ck["val_mae"]); ts.append(ck["test_mae"])
        vs, ts = np.array(vs), np.array(ts)
        rows.append({"role": role, "emb": D["emb"], "label": swp.label(spec), "n_seeds": len(SEEDS),
                     "val_mae_mean": round(float(vs.mean()), 3), "val_mae_std": round(float(vs.std()), 3),
                     "test_mae_mean": round(float(ts.mean()), 3), "test_mae_std": round(float(ts.std()), 3)})
    rows.sort(key=lambda r: r["val_mae_mean"])
    with open(os.path.join(OUT, "results_top3.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\n=== top-3 confirmation ({role}/{D['emb']}, mean +/- std over 3 seeds) ===", flush=True)
    for r in rows:
        print(f"{r['label']:20} val {r['val_mae_mean']:.2f} ± {r['val_mae_std']:.2f}   "
              f"test {r['test_mae_mean']:.2f} ± {r['test_mae_std']:.2f}", flush=True)
    print(f"\nwinner: {rows[0]['label']}", flush=True)


if __name__ == "__main__":
    main()
