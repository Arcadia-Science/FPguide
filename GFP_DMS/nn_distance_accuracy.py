#!/usr/bin/env python3
"""Bright/dim classifier accuracy stratified by the campaigns' in-distribution (ID) NN distance.

The design campaigns call a sequence in-distribution when its nearest-neighbour L2 distance to the
40k GFP-DMS cloud, in z-scored ESM-2 max-pool space, is at most the cloud's own 99th-percentile
self-excluded NN distance (`make_shortlist_case.py`). This script turns that same statistic on the
cloud itself: every one of the 40k sequences gets its self-excluded NN distance to the other 39,999,
those distances are converted to percentiles, and classifier accuracy is reported per percentile
bin -- with the top bins (95-100) being the sparsest, most outlying corner of the DMS cloud, i.e.
the regime where a design sitting right at the ID cutoff would be scored.

Accuracy uses the val-F1-optimal threshold from visualize_sweep.ipynb. Train rows are reported for
completeness but are fitted data; read the val/test columns for the honest number.

Usage:
    python nn_distance_accuracy.py                                   # sub40k + the 40k checkpoint
    python nn_distance_accuracy.py --ckpt trained_models/cnn_max_d2_40k/cnn-max-d2_s0.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "esm2_design"))
import peak_models as pm  # noqa: E402
import sweep_classify_parallel as sw  # noqa: E402

MAXPOOL = os.path.join(HERE, "DMS_data", "esm_maxpool_4scaffold_10k.npz")   # row-aligned with sub40k
SCAFFOLDS = ["avGFP", "amacGFP", "cgreGFP", "ppluGFP"]


def self_nn_distance(Z, dev, chunk=2048):
    """Self-excluded 1-NN L2 distance of every row of Z to the rest of the cloud."""
    T = torch.as_tensor(Z, dtype=torch.float32, device=dev)
    out = np.empty(len(Z), dtype=np.float64)
    for i in range(0, len(T), chunk):
        d = torch.cdist(T[i:i + chunk], T)
        d[torch.arange(len(d), device=dev), torch.arange(i, i + len(d), device=dev)] = float("inf")
        out[i:i + chunk] = d.min(dim=1).values.double().cpu().numpy()
    return out


def _auroc(y, p):
    return sw._auroc(np.asarray(y), np.asarray(p)) if 0 < np.sum(y) < len(y) else np.nan


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (accuracy CI)."""
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_split_cache(stem):
    """sub{20,40}k cache with the residue embeddings left on disk (memmap)."""
    emb, lenp, csvp = sw.data_paths(stem)
    meta = pd.read_csv(csvp)
    split = meta["split"].to_numpy()
    H = np.load(emb, mmap_mode="r")
    D = dict(H=H, Ls=np.load(lenp).astype(np.int64), ar=torch.arange(H.shape[1]),
             y=meta["label"].to_numpy(np.float32),
             tr=np.where(split == "train")[0], va=np.where(split == "val")[0],
             te=np.where(split == "test")[0])
    return D, meta, split


def plot(tab, dist, pct, p99, y, correct, split, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = tab[tab.bin != "ALL"]
    x = np.arange(len(d))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))

    a1.bar(x, d.acc_heldout, color="#2b6c8f", edgecolor="k", alpha=0.85, label="val+test accuracy")
    err = np.clip([d.acc_heldout - d.acc_lo, d.acc_hi - d.acc_heldout], 0, None)
    a1.errorbar(x, d.acc_heldout, yerr=err, fmt="none", ecolor="k", capsize=3)
    a1.plot(x, d.bal_acc, "o-", color="#f6a200", label="balanced accuracy")
    a1.plot(x, d.majority_acc, "s--", color="#9a9a9a", label="majority-class baseline")
    a1.plot(x, d.auroc, "^-", color="#c5474b", label="AUROC")
    for i, n in enumerate(d.n_heldout):
        a1.text(i, 0.03, f"n={n}", ha="center", fontsize=8, rotation=90)
    a1.set_xticks(x); a1.set_xticklabels(d["bin"], rotation=20)
    a1.set_ylim(0, 1.05); a1.set_xlabel("percentile of NN distance within the 40k DMS cloud")
    a1.set_title("bright/dim accuracy on held-out rows by ID-distance percentile")
    a1.legend(loc="lower left", fontsize=9)

    ok = split != "train"
    a2.hist([dist[ok & (correct == 1)], dist[ok & (correct == 0)]], bins=60, stacked=True,
            color=["#2b6c8f", "#c5474b"], label=["correct", "wrong"])
    a2.axvline(p99, ls="--", color="k", lw=1.5, label=f"campaign ID cutoff (p99 = {p99:.2f})")
    a2.set_yscale("log"); a2.set_xlabel("NN distance to the 40k cloud (z-scored max-pool space)")
    a2.set_ylabel("held-out rows"); a2.set_title("NN-distance distribution"); a2.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", default="sub40k")
    ap.add_argument("--ckpt", default="trained_models/cnn_max_d2_40k/cnn-max-d2_s0.pt")
    ap.add_argument("--out", default="figures/nn_distance_accuracy")
    a = ap.parse_args()

    dev = sw.device()
    D, meta, split = load_split_cache(a.stem)
    print(f"cache {a.stem}: {len(meta)} rows | train {len(D['tr'])} val {len(D['va'])} test {len(D['te'])}")

    # ---- campaign ID statistic: self-excluded NN distance in z-scored max-pool space ----
    z = np.load(MAXPOOL, allow_pickle=True)
    mp = z["mp"]
    assert (z["scaf"] == meta["scaffold"].values).all(), "max-pool cache is not row-aligned with the CSV"
    Z = (mp - mp.mean(0)) / (mp.std(0) + 1e-6)
    dist = self_nn_distance(Z, dev)
    p99 = float(np.percentile(dist, 99))                      # the campaigns' in-distribution cutoff
    pct = 100.0 * (np.argsort(np.argsort(dist)) + 0.5) / len(dist)
    print(f"NN distance: median {np.median(dist):.3f}  p95 {np.percentile(dist,95):.3f}  "
          f"p99 (ID cutoff) {p99:.3f}  max {dist.max():.3f}")

    # ---- score every row (train rows are fitted; kept for reference) ----
    net, ck = pm.load_model(os.path.join(HERE, a.ckpt), dev, out=1)
    print(f"checkpoint {a.ckpt}: {ck.get('arch')}-{ck.get('pool')}-d{ck.get('depth')} n_train={ck.get('n_train')}",
          flush=True)
    logits, idx = sw.predict_logits(net, D, np.arange(len(meta)), dev)
    prob = 1 / (1 + np.exp(-logits.astype(np.float64)))
    prob = prob[np.argsort(idx)]                              # predict_logits sorts; restore row order
    y = D["y"].astype(int)

    def f1(yy, yhat):
        tp = ((yhat == 1) & (yy == 1)).sum()
        return 2 * tp / max(2 * tp + ((yhat == 1) & (yy == 0)).sum() + ((yhat == 0) & (yy == 1)).sum(), 1)

    grid = np.linspace(prob[D["va"]].min(), prob[D["va"]].max(), 200)
    thr = float(grid[int(np.argmax([f1(y[D["va"]], (prob[D["va"]] >= t).astype(int)) for t in grid]))])
    correct = ((prob >= thr).astype(int) == y).astype(int)
    ho = split != "train"                                     # val + test = never fitted
    print(f"val-opt F1 threshold {thr:.3f} | held-out acc {correct[ho].mean():.4f} "
          f"AUROC {_auroc(y[ho], prob[ho]):.4f}  n={ho.sum()} | test acc {correct[D['te']].mean():.4f}")

    # ---- stratify by percentile of the NN-distance distribution ----
    edges = np.arange(95, 101, 1)
    masks = [("0-95", pct < edges[0])]
    masks += [(f"{lo}-{hi}", (pct >= lo) & (pct < hi)) for lo, hi in zip(edges[:-1], edges[1:])]
    masks += [("ALL", np.ones(len(pct), bool))]
    rows = []
    for name, m in masks:
        h = m & ho
        n = int(h.sum())
        lo_ci, hi_ci = wilson(int(correct[h].sum()), n)
        br, dm = correct[h & (y == 1)], correct[h & (y == 0)]
        rows.append(dict(bin=name, n_all=int(m.sum()), n_heldout=n, dist_lo=dist[m].min(), dist_hi=dist[m].max(),
                         bright_pct=100 * y[h].mean() if n else np.nan,
                         acc_heldout=correct[h].mean() if n else np.nan, acc_lo=lo_ci, acc_hi=hi_ci,
                         acc_test=correct[m & (split == "test")].mean() if (m & (split == "test")).any() else np.nan,
                         acc_train=correct[m & (split == "train")].mean() if (m & (split == "train")).any() else np.nan,
                         bal_acc=0.5 * (br.mean() + dm.mean()) if len(br) and len(dm) else np.nan,
                         bright_recall=br.mean() if len(br) else np.nan,
                         dim_recall=dm.mean() if len(dm) else np.nan,
                         auroc=_auroc(y[h], prob[h]) if n else np.nan,
                         majority_acc=max(y[h].mean(), 1 - y[h].mean()) if n else np.nan))
    tab = pd.DataFrame(rows)
    print("\n=== accuracy by percentile of NN distance within the 40k cloud (held-out = val+test) ===")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== scaffold composition per bin (all 40k rows) ===")
    comp = pd.DataFrame([{"bin": n, **{s: int(((meta["scaffold"].values == s) & m).sum()) for s in SCAFFOLDS}}
                         for n, m in masks])
    print(comp.to_string(index=False))

    os.makedirs(os.path.join(HERE, os.path.dirname(a.out)), exist_ok=True)
    plot(tab, dist, pct, p99, y, correct, split, os.path.join(HERE, a.out + ".png"))
    tab.to_csv(os.path.join(HERE, a.out + ".csv"), index=False)
    pd.DataFrame(dict(scaffold=meta["scaffold"], split=split, nn_dist=dist, nn_pct=pct,
                      y=y, prob=prob, correct=correct)).to_csv(os.path.join(HERE, a.out + "_per_row.csv"),
                                                              index=False)
    json.dump(dict(stem=a.stem, ckpt=a.ckpt, threshold=thr, id_cutoff_p99=p99,
                   heldout_acc=float(correct[ho].mean()), heldout_auroc=float(_auroc(y[ho], prob[ho]))),
              open(os.path.join(HERE, a.out + "_meta.json"), "w"), indent=2)
    print(f"\nwrote {a.out}.csv / _per_row.csv / _meta.json")


if __name__ == "__main__":
    main()
