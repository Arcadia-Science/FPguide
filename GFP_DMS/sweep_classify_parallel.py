#!/usr/bin/env python3
"""Multi-GPU bright/dim CLASSIFIER sweep over the 4-scaffold 20k subsample.

The regression analogue is `sweep_brightness.py`; this trades the log-brightness regression head
for a binary **bright vs dim** classifier (single logit, BCE-with-logits) and swaps the metrics to
AUROC / AUPRC / accuracy / F1. It runs a CNN/MLP backbone x pooling x depth grid (24 configs; no
transformer) but **in parallel across GPUs**: the launcher assigns configs round-robin to one worker per GPU
(each pinned via CUDA_VISIBLE_DEVICES), every worker trains its share, and the launcher merges the
per-config checkpoints into a ranked leaderboard.

Data: the compact `DMS_data/sub20k_*` cache built by `build_subsample.py` (5k rows x 4 scaffolds,
padded to a common Lmax, label bright=1/dim=0, per-scaffold 70/15/15 split). Small enough (~12 GB
fp16) that each worker just holds it in RAM.

Outputs:
    trained_models/sweep_class4/<arch>-<pool>-d<depth>_s<seed>.pt   per fit (weights + metrics)
    trained_models/sweep_class4/results.csv / results.json          leaderboard (ranked by val AUROC)

Usage:
    python sweep_classify_parallel.py --dry-run             # list configs + shard assignment
    python sweep_classify_parallel.py --gpus 0,1,2,3        # full 24-config sweep on 4 GPUs
    python sweep_classify_parallel.py --configs cnn-max-d1  # a single config (benchmark)
    python sweep_classify_parallel.py --limit 4             # first 4 configs (smoke test)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from fpdesign import peak_models as pm  # noqa: E402

DMS = os.path.join(HERE, "DMS_data")


def data_paths(stem="sub20k"):
    """(emb, len, csv) cache paths for a given subsample stem (built by build_subsample.py)."""
    return (os.path.join(DMS, f"{stem}_esm_residue_fp16.npy"),
            os.path.join(DMS, f"{stem}_esm_residue_len.npy"),
            os.path.join(DMS, f"{stem}_sequences.csv"))


EMB, LEN, CSV = data_paths()                        # defaults (sub20k); notebook reads sw.CSV
OUT = os.path.join(HERE, "trained_models", "sweep_class4")

D_IN = 1280
POOLS = ["mean", "max", "concat", "concatstd", "attn", "cov"]
DEPTHS = (1, 2, 3)
BS, LR, WD = 128, 1e-3, 1e-4
EVAL_BS = 256


def device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def make_configs():
    cfgs = []
    for pool in POOLS:
        cfgs.append({"arch": "mlp", "pool": pool, "depth": 0})
        for d in DEPTHS:
            cfgs.append({"arch": "cnn", "pool": pool, "n_conv": d, "depth": d})
    return cfgs                                     # 6 + 18 = 24 (CNN + MLP only, no transformer)


def label(spec):
    return f"{spec['arch']}-{spec['pool']}-d{spec['depth']}"


# --------------------------------------------------------------------------- data
def load_data(dev, stem=None):
    """Load a compact subsample cache into RAM + its persisted per-scaffold split. Returns a dict."""
    emb, lenp, csvp = data_paths(stem) if stem else (EMB, LEN, CSV)
    for p in (emb, lenp, csvp):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {p}\nRun: python build_subsample.py")
    rows = list(csv.DictReader(open(csvp)))
    y = np.array([int(r["label"]) for r in rows], dtype=np.float32)
    split = np.array([r["split"] for r in rows])
    H = np.ascontiguousarray(np.load(emb, mmap_mode="r"))          # (N, Lmax, 1280) fp16 -> RAM
    Ls = np.load(lenp).astype(np.int64)
    N, Lmax = H.shape[0], H.shape[1]
    tr = np.where(split == "train")[0]; va = np.where(split == "val")[0]; te = np.where(split == "test")[0]
    return dict(N=N, Lmax=Lmax, H=H, Ls=Ls, ar=torch.arange(Lmax),
                y=y, Y=torch.tensor(y, device=dev).unsqueeze(1), tr=tr, va=va, te=te)


def batches(D, idx, dev, bs, shuffle=False, rng=None):
    idx = np.array(idx)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        Hb = torch.from_numpy(np.ascontiguousarray(D["H"][b])).float().to(dev)
        mk = (D["ar"].unsqueeze(0) < torch.tensor(D["Ls"][b]).unsqueeze(1)).to(dev)
        yield Hb, mk, b


@torch.no_grad()
def predict_logits(net, D, idx, dev):
    """Return (logits, idx) over idx (sorted for cache locality)."""
    idx = np.sort(np.asarray(idx))
    net.eval(); ps = []
    for Hb, mk, b in batches(D, idx, dev, EVAL_BS):
        ps.append(net(Hb, mk).cpu().numpy())
    return np.concatenate(ps).ravel(), idx


# --------------------------------------------------------------------------- metrics
def _auroc(y, s):
    """Rank-based AUROC (no sklearn dependency)."""
    y = np.asarray(y); s = np.asarray(s)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sr = s[order]; i = 0
    while i < len(sr):                                  # average ranks over ties
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _auprc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]; tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(y.sum(), 1)
    rec = np.concatenate([[0], rec]); prec = np.concatenate([[1], prec])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def clf_metrics(logits, y, thr=0.5):
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    y = np.asarray(y)
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    acc = (tp + tn) / max(len(y), 1)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(auroc=_auroc(y, p), auprc=_auprc(y, p), acc=acc, f1=f1, prec=prec, rec=rec)


# --------------------------------------------------------------------------- train
def train_eval(spec, seed, D, dev, max_epochs=200, patience=20):
    """Fit one classifier config; early-stop on val BCE, restore best weights, score val+test."""
    torch.manual_seed(seed); rng = np.random.default_rng(100 + seed)
    net = pm.build_base({**spec, "d_in": D_IN}, dev, out=1, drop=spec.get("drop", 0.2))
    npos = float(D["y"][D["tr"]].sum()); nneg = float(len(D["tr"]) - npos)
    pos_weight = torch.tensor([nneg / max(npos, 1.0)], device=dev)     # counter class imbalance
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), LR, weight_decay=WD)

    best = {"val": float("inf"), "state": None, "epoch": -1}; bad = 0
    for ep in range(max_epochs):
        net.train()
        for Hb, mk, b in batches(D, D["tr"], dev, BS, shuffle=True, rng=rng):
            opt.zero_grad()
            lossf(net(Hb, mk), D["Y"][torch.as_tensor(b, device=dev)]).backward()
            opt.step()
        vl, vi = predict_logits(net, D, D["va"], dev)
        with torch.no_grad():
            v = float(torch.nn.functional.binary_cross_entropy_with_logits(
                torch.tensor(vl), torch.tensor(D["y"][vi]), pos_weight=pos_weight.cpu()))
        if v < best["val"] - 1e-4:
            best = {"val": v, "state": copy.deepcopy({k: x.cpu() for k, x in net.state_dict().items()}),
                    "epoch": ep}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best["state"])

    vl, vi = predict_logits(net, D, D["va"], dev); vm = clf_metrics(vl, D["y"][vi])
    tl, ti = predict_logits(net, D, D["te"], dev); tm = clf_metrics(tl, D["y"][ti])
    m = {f"val_{k}": v for k, v in vm.items()}
    m.update({f"test_{k}": v for k, v in tm.items()})
    m.update(epoch_best=best["epoch"], n_train=int(len(D["tr"])))
    return m, net


MK = ("val_auroc", "val_auprc", "val_acc", "val_f1",
      "test_auroc", "test_auprc", "test_acc", "test_f1", "epoch_best", "n_train")


def _select(cfgs, limit, labels):
    if labels:
        by = {label(c): c for c in cfgs}
        picked = []
        for lab in labels:
            if lab not in by:
                raise SystemExit(f"unknown config {lab!r}; known: {', '.join(by)}")
            picked.append(by[lab])
        return picked
    return cfgs[:limit] if limit else cfgs


# --------------------------------------------------------------------------- worker
def run_worker(a):
    dev = device()
    out_dir = a.out or OUT
    cfgs = _select(make_configs(), a.limit, [s for s in a.configs.split(",") if s] if a.configs else None)
    mine = cfgs[a.shard_id::a.shards]                              # round-robin: mixes cheap/expensive archs
    if not mine:
        print(f"[worker {a.shard_id}] no configs", flush=True); return
    print(f"[worker {a.shard_id}] gpu {os.environ.get('CUDA_VISIBLE_DEVICES','?')} ({dev}) "
          f"stem={a.data_stem or 'sub20k'} -> {len(mine)} configs: {', '.join(label(c) for c in mine)}", flush=True)
    D = load_data(dev, stem=a.data_stem)
    os.makedirs(out_dir, exist_ok=True)
    for spec in mine:
        for seed in a.seeds:
            path = os.path.join(out_dir, f"{label(spec)}_s{seed}.pt")
            if os.path.exists(path) and not a.force:
                print(f"[worker {a.shard_id}] cached  {label(spec)} s{seed}", flush=True); continue
            t0 = time.time()
            m, net = train_eval(spec, seed, D, dev, a.max_epochs, a.patience)
            pm.save_model(path, net, {**spec, "d_in": D_IN, "out": 1, "seed": seed,
                                      "probe_dim": pm.COV_PROBE_DIM, "task": "bright_dim_classify",
                                      "data_stem": a.data_stem or "sub20k",
                                      "scaffolds": "avGFP+amacGFP+cgreGFP+ppluGFP", **m})
            print(f"[worker {a.shard_id}] {label(spec):22} s{seed}  "
                  f"val AUROC {m['val_auroc']:.4f} acc {m['val_acc']:.3f} | "
                  f"test AUROC {m['test_auroc']:.4f} acc {m['test_acc']:.3f}  "
                  f"@ep{m['epoch_best']}  {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- launcher
def merge_leaderboard(cfgs, seeds, out_dir=OUT):
    recs = []
    for spec in cfgs:
        for seed in seeds:
            path = os.path.join(out_dir, f"{label(spec)}_s{seed}.pt")
            if not os.path.exists(path):
                continue
            ck = torch.load(path, map_location="cpu", weights_only=False)
            recs.append({"arch": spec["arch"], "pool": spec["pool"], "depth": spec["depth"],
                         "seed": seed, "label": label(spec), **{k: ck[k] for k in MK},
                         "ckpt": os.path.relpath(path, HERE)})
    recs.sort(key=lambda r: -r["val_auroc"])                      # rank by val AUROC (higher better)
    if not recs:
        print("no checkpoints to merge"); return
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
    json.dump({"task": "bright_dim_classify", "scaffolds": "avGFP+amacGFP+cgreGFP+ppluGFP",
               "results": recs}, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    print(f"\n=== bright/dim classifier leaderboard (val AUROC) ===")
    print(f"{'config':22} {'val_AUROC':>9} {'val_acc':>8} {'test_AUROC':>10} {'test_acc':>8} {'test_F1':>8}")
    for r in recs[:12]:
        print(f"{r['label']:22} {r['val_auroc']:9.4f} {r['val_acc']:8.3f} "
              f"{r['test_auroc']:10.4f} {r['test_acc']:8.3f} {r['test_f1']:8.3f}")
    b = recs[0]
    print(f"best: {b['label']}  val AUROC {b['val_auroc']:.4f}  test AUROC {b['test_auroc']:.4f}")


def run_launcher(a):
    cfgs = _select(make_configs(), a.limit, [s for s in a.configs.split(",") if s] if a.configs else None)
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip() != ""]
    K = len(gpus)
    print(f"configs: {len(cfgs)} x {len(a.seeds)} seed(s) | GPUs {gpus} ({K} workers)")
    for i, g in enumerate(gpus):
        mine = cfgs[i::K]
        print(f"  worker {i} (gpu {g}): {len(mine)} configs -> {', '.join(label(c) for c in mine)}")
    if a.dry_run:
        print("dry-run: not launching"); return

    out_dir = a.out or OUT
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    seeds_s = ",".join(str(s) for s in a.seeds)
    procs = []
    for i, g in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        log = open(os.path.join(HERE, "logs", f"sweep_class4_worker{i}_gpu{g}_{stamp}.log"), "w")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               "--shard-id", str(i), "--shards", str(K), "--seeds", seeds_s,
               "--max-epochs", str(a.max_epochs), "--patience", str(a.patience)]
        if a.configs:
            cmd += ["--configs", a.configs]
        if a.data_stem:
            cmd += ["--data-stem", a.data_stem]
        if a.out:
            cmd += ["--out", a.out]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        if a.force:
            cmd += ["--force"]
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, log))
        print(f"launched worker {i} on gpu {g} (pid {p.pid}) -> {log.name}", flush=True)

    t0 = time.time()
    try:
        while any(p.poll() is None for p, _ in procs):
            time.sleep(20)
            alive = sum(p.poll() is None for p, _ in procs)
            done = len([f for f in os.listdir(out_dir) if f.endswith(".pt")])
            print(f"  ... {alive}/{K} workers alive, {done}/{len(cfgs)*len(a.seeds)} checkpoints, "
                  f"elapsed {(time.time()-t0)/60:.1f}m", flush=True)
    except KeyboardInterrupt:
        print("interrupted; terminating workers"); [p.terminate() for p, _ in procs]

    codes = [p.wait() for p, _ in procs]
    for _, log in procs:
        log.close()
    print(f"worker exit codes: {codes}  total {(time.time()-t0)/60:.1f}m")
    merge_leaderboard(cfgs, a.seeds, out_dir)
    if not all(c == 0 for c in codes):
        sys.exit("one or more workers failed; see logs/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default="0,1,2,3", help="comma-separated GPU ids, one worker each")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="only the first N configs")
    ap.add_argument("--configs", type=str, default=None, help="comma-separated config labels")
    ap.add_argument("--force", action="store_true", help="retrain even if a checkpoint exists")
    ap.add_argument("--data-stem", default=None, help="subsample cache stem (default sub20k)")
    ap.add_argument("--out", default=None, help="checkpoint output dir (default trained_models/sweep_class4)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--shard-id", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--shards", type=int, default=1, help=argparse.SUPPRESS)
    a = ap.parse_args()
    (run_worker if a.worker else run_launcher)(a)


if __name__ == "__main__":
    main()
