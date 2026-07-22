#!/usr/bin/env python3
"""Architecture sweep for avGFP DMS log-brightness regression (single prediction model).

Predicts the Sarkisyan et al. (2016) avGFP DMS **log10 median brightness** from ESM-2 650M
per-residue embeddings. This mirrors scalar_design/sweep_scalar.py -- the same backbone x
pooling grid, single output (out=1), standardized-target MSE, early stopping on val MAE -- but
there is only ONE model here (no surrogate/oracle split), the target is already log10 so it is
used as-is (no log1p), and the 31 GB embedding cache is read through a numpy **memmap** (it does
not fit in RAM), copying only each mini-batch to the GPU.

Inputs (built beforehand):
    DMS_data/avgfp_dms_sequences.csv    rows aligned to the embedding cache; column
                                        `logMedianBrightness` is the regression target
    DMS_data/esm_residue_fp16.npy       (N, Lmax, 1280) fp16  per-residue ESM-2 embeddings
    DMS_data/esm_residue_len.npy        (N,) int64            sequence lengths
    (produced by transform_dms.py and embed_dms.py)

Grid (scalar sweep grid + the covariance-probe pool): backbone in {mlp (pool-only), cnn,
transformer} x pooling in {mean, max, concat, concatstd, attn, cov} x depth in {1,2,3}
= 6 + 18 + 18 = 42 configs. 'cov' is peak_models' learned second-order covariance-probe readout.

Outputs:
    trained_models/sweep/<arch>-<pool>-d<depth>_s<seed>.pt      every fit (weights + metrics + scaler)
    trained_models/sweep/results.csv / results.json             leaderboard (ranked by val MAE)
    DMS_data/split_70_15_15.csv                                 persisted split (index, split)

Resumable (skips fits whose checkpoint exists). Metrics reported in log10-brightness units:
MAE, RMSE, Pearson r, R^2 on both val and test.

Two modes:
  * SWEEP (default): fit the whole grid, validating every epoch with early stopping; ranked
    leaderboard written to trained_models/sweep/. Use --subsample for a fast architecture search.
  * FULL-DATA (--top-k / --configs): take the winners off the sweep leaderboard and train them on
    ALL rows with a fixed epoch budget and NO per-epoch validation -- val + test are scored ONCE at
    the endpoint (a single memmap pass each). Output goes to trained_models/full/.

Usage:
    python sweep_brightness.py --dry-run              # list the 42 configs, no training
    python sweep_brightness.py --limit 2              # first 2 configs (smoke test)
    python sweep_brightness.py --subsample 10000      # architecture search on a 10k random subset
    python sweep_brightness.py                        # full grid sweep (needs the embedding cache)
    python sweep_brightness.py --top-k 3 --max-epochs 30   # train top-3 winners on the full dataset
    python sweep_brightness.py --configs cnn-cov-d3,cnn-mean-d2 --max-epochs 30
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# peak_models.py (ESM-2 embedding utils + PeakCNN/PeakTransformer/Readout) lives in esm2_design/.
sys.path.insert(0, os.path.join(HERE, "..", "esm2_design"))
import peak_models as pm  # noqa: E402

DMS = os.path.join(HERE, "DMS_data")
CSV = os.path.join(DMS, "avgfp_dms_sequences.csv")
EMB = os.path.join(DMS, "esm_residue_fp16.npy")
LEN = os.path.join(DMS, "esm_residue_len.npy")
OUT = os.path.join(HERE, "trained_models", "sweep")
FULL = os.path.join(HERE, "trained_models", "full")
SPLIT = os.path.join(DMS, "split_70_15_15.csv")

TARGET_COL = "logMedianBrightness"
UNIT = "log10-brightness"
D_IN = 1280
POOLS = ["mean", "max", "concat", "concatstd", "attn", "cov"]  # 'cov' = covariance-probe pool
DEPTHS = (1, 2, 3)
BS, LR, WD = 32, 1e-3, 1e-4
EVAL_BS = 128
RAM_CACHE_GB = 18          # cache the WHOLE working set in RAM when it fits under this
TRAIN_CACHE_GB = 22        # else cache just the train split in RAM (val/test stream from memmap)

_BUILD_HINT = ("Build the inputs first:\n"
               "  python avGFP_DMS/transform_dms.py     # -> DMS_data/avgfp_dms_sequences.csv\n"
               "  python avGFP_DMS/embed_dms.py         # -> DMS_data/esm_residue_fp16.npy (+_len)")


def device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def make_configs():
    cfgs = []
    for pool in POOLS:
        cfgs.append({"arch": "mlp", "pool": pool, "depth": 0})
        for d in DEPTHS:
            cfgs.append({"arch": "cnn", "pool": pool, "n_conv": d, "depth": d})
            cfgs.append({"arch": "transformer", "pool": pool, "nlayers": d, "depth": d})
    return cfgs                                     # 6 + 18 + 18 = 42


def label(spec):
    return f"{spec['arch']}-{spec['pool']}-d{spec['depth']}"


def load_data(dev, subsample=None, cache_ram=None, split_seed=0):
    """Load the log-brightness target + ESM-2 embeddings + a fixed-seed 70/15/15 split.

    All indices below live in the GLOBAL row space (CSV / embedding-cache order). The 29 GiB
    (N, Lmax, 1280) fp16 cache is opened as a memmap; per-row random reads from it are slow, so we
    cache the hot rows in RAM under a three-way policy (auto by size, or forced via `cache_ram`):

      * whole working set < RAM_CACHE_GB  -> cache every row in RAM (e.g. a 10k subsample ~ 6 GB);
      * else train split < TRAIN_CACHE_GB -> cache only TRAIN in RAM (val/test read from the memmap,
        which is fine because full-mode evaluates them once at the endpoint);
      * else                              -> stream everything from the memmap.

    `D["cmap"]` maps a global row index -> its position in the RAM cache `D["Hc"]` (or -1 = read the
    memmap `D["H"]`), so `batches` transparently serves each row from wherever it lives.
    """
    for p, what in [(CSV, "sequence/target CSV"), (EMB, "ESM-2 embedding cache"), (LEN, "length cache")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{what} not found: {p}\n{_BUILD_HINT}")

    rows = list(csv.DictReader(open(CSV)))
    y = np.array([float(r[TARGET_COL]) for r in rows], dtype=np.float32)
    H = np.load(EMB, mmap_mode="r")                                  # (N, Lmax, 1280) fp16 memmap
    Ls = np.load(LEN).astype(np.int64)
    N = len(y)
    if not (H.shape[0] == N == len(Ls)):
        raise ValueError(f"row mismatch: csv={N} emb={H.shape[0]} len={len(Ls)}")
    if H.shape[2] != D_IN:
        raise ValueError(f"embedding dim {H.shape[2]} != expected {D_IN}")
    Lmax = int(H.shape[1])
    row_gb = Lmax * D_IN * 2 / 1e9

    universe = np.arange(N)
    if subsample and subsample < N:                                 # architecture search on a subset
        universe = np.sort(np.random.default_rng(0).choice(N, size=subsample, replace=False))

    # fixed-seed 70/15/15 split over the working universe
    perm = np.random.default_rng(split_seed).permutation(universe)
    n_te = int(round(0.15 * len(perm))); n_va = int(round(0.15 * len(perm)))
    te, va, tr = np.sort(perm[:n_te]), np.sort(perm[n_te:n_te + n_va]), np.sort(perm[n_te + n_va:])
    mean = np.array([y[tr].mean()], np.float32); std = np.array([y[tr].std() + 1e-6], np.float32)

    # three-way RAM caching (see docstring)
    full_gb, train_gb = len(universe) * row_gb, len(tr) * row_gb
    cmap = np.full(N, -1, dtype=np.int64)
    if cache_ram is False:
        cache_rows = np.empty(0, dtype=np.int64)
        print(f"streaming all splits from the on-disk memmap ({full_gb:.1f} GB)", flush=True)
    elif cache_ram is True or full_gb < RAM_CACHE_GB:
        cache_rows = np.sort(universe)
        print(f"caching all {len(cache_rows)} rows in RAM (~{full_gb:.1f} GB)...", flush=True)
    elif train_gb < TRAIN_CACHE_GB:
        cache_rows = tr
        print(f"caching {len(tr)} TRAIN rows in RAM (~{train_gb:.1f} GB); val/test stream from memmap",
              flush=True)
    else:
        cache_rows = np.empty(0, dtype=np.int64)
        print(f"train split ~{train_gb:.1f} GB > cache limit; streaming from memmap", flush=True)

    Hc = None
    if len(cache_rows):
        Hc = np.ascontiguousarray(np.asarray(H[cache_rows]))        # (len, Lmax, 1280) fp16 in RAM
        cmap[cache_rows] = np.arange(len(cache_rows))

    role = np.full(N, "unused", dtype=object)
    role[tr] = "train"; role[va] = "val"; role[te] = "test"
    with open(SPLIT, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["index", "split", TARGET_COL])
        for i in range(N):
            w.writerow([i, role[i], f"{y[i]:.6f}"])

    return dict(unit=UNIT, N=N, Lmax=Lmax, H=H, Hc=Hc, cmap=cmap, Ls=Ls, ar=torch.arange(Lmax),
                y=y, Y=torch.tensor(y, device=dev).unsqueeze(1),
                mean=mean, std=std, sd=torch.tensor(std, device=dev),
                tr=tr, va=va, te=te, split_seed=split_seed)


def _gather(D, b):
    """Fetch embeddings for global row indices `b` from the RAM cache and/or the memmap."""
    if D["Hc"] is None:
        return np.ascontiguousarray(D["H"][b])
    locs = D["cmap"][b]
    if (locs >= 0).all():                                            # all cached (e.g. a train batch)
        return np.ascontiguousarray(D["Hc"][locs])
    Hb = np.empty((len(b), D["Lmax"], D_IN), dtype=np.float16)
    cached = locs >= 0
    if cached.any():
        Hb[cached] = D["Hc"][locs[cached]]
    if (~cached).any():
        Hb[~cached] = D["H"][b[~cached]]                            # memmap read
    return Hb


def batches(D, idx, dev, bs, shuffle=False, rng=None):
    idx = np.array(idx)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        Hb = torch.from_numpy(_gather(D, b)).float().to(dev)
        mk = (D["ar"].unsqueeze(0) < torch.tensor(D["Ls"][b]).unsqueeze(1)).to(dev)
        yield Hb, mk, b


@torch.no_grad()
def predict(net, D, idx, dev):
    """Predict over `idx`; returns (P, idx_used). idx is sorted so memmap-backed splits read
    monotonically. Metrics are order-agnostic, and callers index truth with the returned idx."""
    idx = np.sort(np.asarray(idx))
    net.eval(); ps = []
    for Hb, mk, b in batches(D, idx, dev, EVAL_BS):
        ps.append(net(Hb, mk).cpu().numpy())
    return np.concatenate(ps).ravel(), idx


def metrics(P, T):
    """Regression metrics in target units: MAE, RMSE, Pearson r, R^2."""
    P = np.asarray(P, float); T = np.asarray(T, float)
    err = P - T
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    if P.std() > 0 and T.std() > 0:
        r = float(np.corrcoef(P, T)[0, 1])
    else:
        r = float("nan")
    ss_res = float((err ** 2).sum()); ss_tot = float(((T - T.mean()) ** 2).sum())
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return mae, rmse, r, r2


def _split_metrics(net, D, dev, prefix, idx):
    P, idx = predict(net, D, idx, dev)                              # idx sorted -> align truth the same way
    mae, rmse, r, r2 = metrics(P, D["y"][idx])
    return {f"{prefix}_mae": mae, f"{prefix}_rmse": rmse, f"{prefix}_pearson": r, f"{prefix}_r2": r2}


def train_eval(spec, seed, D, dev, max_epochs=200, patience=20, endpoint=False):
    """Fit one config. Two regimes:

      * endpoint=False (sweep): validate every epoch, early-stop on val MAE, restore best weights.
      * endpoint=True  (full):  no per-epoch validation -- train a fixed `max_epochs` budget, keep the
        final weights, and evaluate val + test ONCE at the endpoint (a single memmap pass each).
    """
    torch.manual_seed(seed); rng = np.random.default_rng(100 + seed)
    base = pm.build_base({**spec, "d_in": D_IN}, dev, out=1, drop=spec.get("drop", 0.2))
    net = pm.wrap(base, D["mean"], D["std"], dev)                   # forward returns brightness in log units
    opt = torch.optim.Adam(net.parameters(), LR, weight_decay=WD)
    best = {"val": float("inf"), "state": None, "epoch": -1}; bad = 0
    for ep in range(max_epochs):
        net.train()
        for Hb, mk, b in batches(D, D["tr"], dev, BS, shuffle=True, rng=rng):
            opt.zero_grad()
            (((net(Hb, mk) - D["Y"][torch.as_tensor(b, device=dev)]) / D["sd"]) ** 2).mean().backward()
            opt.step()
        if endpoint:
            continue                                               # no per-epoch val; train the full budget
        P, vi = predict(net, D, D["va"], dev)
        v = float(np.abs(P - D["y"][vi]).mean())
        if v < best["val"] - 1e-4:
            best = {"val": v, "state": copy.deepcopy({k: x.cpu() for k, x in net.base.state_dict().items()}),
                    "epoch": ep}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if not endpoint:
        net.base.load_state_dict(best["state"])
    m = {**_split_metrics(net, D, dev, "val", D["va"]),
         **_split_metrics(net, D, dev, "test", D["te"]),
         "epoch_best": (max_epochs - 1 if endpoint else best["epoch"]), "n_train": int(len(D["tr"]))}
    return m, net.base


MK = ("val_mae", "val_rmse", "val_pearson", "val_r2",
      "test_mae", "test_rmse", "test_pearson", "test_r2", "epoch_best", "n_train")


def select_configs(top_k=None, labels=None):
    """Pick configs for full-dataset training from the sweep leaderboard (trained_models/sweep/results.csv),
    reconstructing each spec (incl. n_conv/nlayers) from make_configs() by matching its label."""
    by_label = {label(c): c for c in make_configs()}
    res = os.path.join(OUT, "results.csv")
    if not os.path.exists(res):
        raise FileNotFoundError(f"sweep leaderboard not found: {res}\nRun the --subsample sweep first.")
    ranked = [r["label"] for r in sorted(csv.DictReader(open(res)), key=lambda r: float(r["val_mae"]))]
    seen, order = set(), []
    for lab in (ranked if labels is None else labels):             # dedupe, keep leaderboard order
        if lab in seen:
            continue
        if lab not in by_label:
            raise ValueError(f"unknown config label: {lab}\nknown: {', '.join(by_label)}")
        seen.add(lab); order.append(lab)
    if labels is None and top_k:
        order = order[:top_k]
    return [by_label[lab] for lab in order]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="seeds per config (default: 0)")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="only the first N configs (smoke)")
    ap.add_argument("--subsample", type=int, default=None, help="train/select on a random N-row subset")
    ap.add_argument("--split-seed", type=int, default=0, help="seed for the 70/15/15 split (default: 0)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="FULL-DATA mode: train the top-K configs from the sweep leaderboard on all rows "
                         "(fixed budget, no early stopping; val+test scored once at the endpoint)")
    ap.add_argument("--configs", type=str, default=None,
                    help="FULL-DATA mode: comma-separated config labels to train (e.g. cnn-mean-d2,cnn-cov-d3)")
    ap.add_argument("--cache-ram", dest="cache_ram", action="store_true", default=None,
                    help="force caching the working embeddings in RAM (default: auto by size)")
    ap.add_argument("--no-cache-ram", dest="cache_ram", action="store_false",
                    help="force streaming from the on-disk memmap instead of caching in RAM")
    ap.add_argument("--force", action="store_true", help="retrain even if a checkpoint exists")
    ap.add_argument("--dry-run", action="store_true", help="list configs, no training")
    a = ap.parse_args()

    full_mode = a.top_k is not None or a.configs is not None
    if full_mode:
        labels = [s.strip() for s in a.configs.split(",")] if a.configs else None
        cfgs = select_configs(top_k=a.top_k, labels=labels)
        out_dir = FULL
        endpoint = True                                            # train + validate-at-endpoint
    else:
        cfgs = make_configs()
        if a.limit:
            cfgs = cfgs[:a.limit]
        out_dir = OUT
        endpoint = False

    if a.dry_run:
        for c in cfgs:
            print(" ", label(c))
        mode = "FULL-DATA (endpoint eval)" if full_mode else "SWEEP"
        print(f"[{mode}] {len(cfgs)} configs x {len(a.seeds)} seed(s) = {len(cfgs) * len(a.seeds)} fits")
        return

    dev = device(); print(f"device: {dev}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    D = load_data(dev, subsample=(None if full_mode else a.subsample),
                  cache_ram=a.cache_ram, split_seed=a.split_seed)
    print(f"[avGFP-DMS] target={UNIT} | N={D['N']} Lmax={D['Lmax']} | split_seed={a.split_seed} | "
          f"split train={len(D['tr'])} val={len(D['va'])} test={len(D['te'])}"
          + (" | FULL-DATA endpoint eval" if full_mode else
             (f" | subsample={a.subsample}" if a.subsample else "")), flush=True)

    per = []
    for ci, spec in enumerate(cfgs):
        for seed in a.seeds:
            path = os.path.join(out_dir, f"{label(spec)}_s{seed}.pt")
            if os.path.exists(path) and not a.force:
                ck = torch.load(path, map_location="cpu", weights_only=False)
                m = {k: ck[k] for k in MK}; tag = "cached "
            else:
                m, base = train_eval(spec, seed, D, dev, a.max_epochs, a.patience, endpoint=endpoint)
                pm.save_model(path, base, {**spec, "d_in": D_IN, "out": 1, "seed": seed,
                                           "probe_dim": pm.COV_PROBE_DIM,  # self-describing 'cov' readout
                                           "trait": "avgfp_dms_brightness", "unit": UNIT,
                                           "split_seed": a.split_seed, "full_data": full_mode,
                                           "mean": D["mean"], "std": D["std"], **m})
                tag = "trained"
            per.append({"arch": spec["arch"], "pool": spec["pool"], "depth": spec["depth"],
                        "seed": seed, "label": label(spec), **m, "ckpt": os.path.relpath(path, HERE)})
            print(f"[{ci+1:2}/{len(cfgs)}] {tag} {label(spec):22} s{seed} "
                  f"val MAE {m['val_mae']:.3f} r {m['val_pearson']:.3f} R2 {m['val_r2']:.3f} | "
                  f"test MAE {m['test_mae']:.3f} r {m['test_pearson']:.3f}  @ep{m['epoch_best']}", flush=True)

    per.sort(key=lambda r: r["val_mae"])
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0].keys())); w.writeheader(); w.writerows(per)
    json.dump({"trait": "avgfp_dms_brightness", "unit": UNIT, "full_data": full_mode,
               "split_seed": a.split_seed, "results": per},
              open(os.path.join(out_dir, "results.json"), "w"), indent=2)

    print(f"\n=== avGFP DMS brightness leaderboard (val MAE, {UNIT}) ===", flush=True)
    print(f"{'config':22} {'val_MAE':>8} {'val_r':>7} {'val_R2':>7} {'test_MAE':>9} {'test_r':>7}")
    for r in per[:12]:
        print(f"{r['label']:22} {r['val_mae']:8.3f} {r['val_pearson']:7.3f} {r['val_r2']:7.3f} "
              f"{r['test_mae']:9.3f} {r['test_pearson']:7.3f}", flush=True)
    b = per[0]
    print(f"best: {b['label']}  val MAE {b['val_mae']:.3f} {UNIT}  r {b['val_pearson']:.3f}  R2 {b['val_r2']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
