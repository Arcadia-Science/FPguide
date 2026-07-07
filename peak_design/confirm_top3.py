#!/usr/bin/env python3
"""Confirm the top-3 sweep configs at 3 seeds (reuses seed-0 checkpoints; trains seeds 1-2).

Saves all checkpoints into trained_models/oracle_sweep/ and writes results_top3.csv with
mean +/- std over seeds. Does NOT touch the full-sweep results.csv.
"""
import csv, os
from collections import defaultdict

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
    dev = swp.device(); print("device:", dev, flush=True)
    os.makedirs(swp.OUT, exist_ok=True)
    D = swp.load_data(dev)
    for spec in TOP:
        for seed in SEEDS:
            path = os.path.join(swp.OUT, f"{swp.label(spec)}_s{seed}.pt")
            if os.path.exists(path):
                print(f"[cached ] {swp.label(spec):20} s{seed}", flush=True); continue
            m, base = swp.train_eval(spec, seed, D, dev, 200, 20)
            pm.save_model(path, base, {**spec, "out": 2, "seed": seed,
                                       "mean": D["mean"], "std": D["std"], **m})
            print(f"[trained] {swp.label(spec):20} s{seed}  val {m['val_mae']:.2f}  test {m['test_mae']:.2f}", flush=True)

    rows = []
    for spec in TOP:
        vs, ts = [], []
        for seed in SEEDS:
            ck = torch.load(os.path.join(swp.OUT, f"{swp.label(spec)}_s{seed}.pt"),
                            map_location="cpu", weights_only=False)
            vs.append(ck["val_mae"]); ts.append(ck["test_mae"])
        vs, ts = np.array(vs), np.array(ts)
        rows.append({"label": swp.label(spec), "n_seeds": len(SEEDS),
                     "val_mae_mean": round(float(vs.mean()), 3), "val_mae_std": round(float(vs.std()), 3),
                     "test_mae_mean": round(float(ts.mean()), 3), "test_mae_std": round(float(ts.std()), 3)})
    rows.sort(key=lambda r: r["val_mae_mean"])
    with open(os.path.join(swp.OUT, "results_top3.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\n=== top-3 confirmation (mean +/- std over 3 seeds) ===", flush=True)
    for r in rows:
        print(f"{r['label']:20} val {r['val_mae_mean']:.2f} ± {r['val_mae_std']:.2f}   "
              f"test {r['test_mae_mean']:.2f} ± {r['test_mae_std']:.2f}", flush=True)
    print(f"\nwinner: {rows[0]['label']}", flush=True)


if __name__ == "__main__":
    main()
