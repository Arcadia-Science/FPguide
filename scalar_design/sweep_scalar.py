#!/usr/bin/env python3
"""Architecture sweep for a scalar FP target (brightness or pKa).

Same backbone x pooling grid as the peak sweep, but:
  - single output (out=1)
  - depth up to 3 backbone layers
  - 1 seed per config
  - a plain 70/15/15 train/val/test split (seed 0) -- NOT the coordinated dual split

brightness is modeled in log space (log1p; MAE reported in log units); pKa directly (MAE in pH units).
Resumable (skips fits whose checkpoint exists). Saves every fit + results.csv/json.

Usage:
    python sweep_scalar.py --trait brightness
    python sweep_scalar.py --trait pka
    python sweep_scalar.py --trait pka --dry-run
"""
from __future__ import annotations

import argparse, copy, csv, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# peak_models.py lives in the sibling peak_design/ folder.
sys.path.insert(0, os.path.join(HERE, "..", "peak_design"))
import peak_models as pm
POOLS = ["mean", "max", "concat", "concatstd", "attn"]
DEPTHS = (1, 2, 3)
BS, LR, WD = 32, 1e-3, 1e-4

TRAITS = {
    "brightness": dict(cur="../dataset_pipeline/data/brightness/curated", target="brightness.npy",
                       assign="brightness_assignments.csv", log=True, unit="log-brightness",
                       tdm="trained_models_brightness"),
    "pka": dict(cur="../dataset_pipeline/data/pka/curated", target="pka.npy",
                assign="pka_assignments.csv", log=False, unit="pKa (pH)",
                tdm="trained_models_pka"),
}


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
    return cfgs                                   # 5 + 15 + 15 = 35


def label(spec):
    return f"{spec['arch']}-{spec['pool']}-d{spec['depth']}"


def load_data(trait, dev):
    cfg = TRAITS[trait]
    cur = os.path.join(HERE, cfg["cur"])
    rows = sorted(csv.DictReader(open(os.path.join(cur, cfg["assign"]))), key=lambda r: int(r["index"]))
    y_raw = np.load(os.path.join(cur, cfg["target"])).astype(np.float32).ravel()
    y = np.log1p(y_raw).astype(np.float32) if cfg["log"] else y_raw
    H = np.load(os.path.join(cur, "esm_residue_fp16.npy"))
    Ls = np.load(os.path.join(cur, "esm_residue_len.npy")).astype(np.int64)
    N = len(y)

    # plain 70/15/15 split (seed 0)
    perm = np.random.default_rng(0).permutation(N)
    n_te = int(round(0.15 * N)); n_va = int(round(0.15 * N))
    te, va, tr = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
    mean = np.array([y[tr].mean()], np.float32); std = np.array([y[tr].std() + 1e-6], np.float32)

    # persist the split for provenance
    role = np.empty(N, dtype=object)
    role[tr] = "train"; role[va] = "val"; role[te] = "test"
    with open(os.path.join(cur, "split_70_15_15.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["index", "name", "role"])
        for i, r in enumerate(rows):
            w.writerow([i, r["name"], role[i]])

    Lmax = int(H.shape[1])
    return dict(unit=cfg["unit"], log=cfg["log"], N=N, Lmax=Lmax,
                Ht=torch.tensor(H), Ls=Ls, ar=torch.arange(Lmax),
                y=y, Y=torch.tensor(y, device=dev).unsqueeze(1),
                mean=mean, std=std, sd=torch.tensor(std, device=dev),
                tr=tr, va=va, te=te)


def batches(D, idx, dev, shuffle=False, rng=None):
    idx = np.array(idx)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), BS):
        b = idx[i:i + BS]
        mk = (D["ar"].unsqueeze(0) < torch.tensor(D["Ls"][b]).unsqueeze(1)).to(dev)
        yield D["Ht"][b].float().to(dev), mk, b


@torch.no_grad()
def eval_mae(net, D, idx, dev):
    net.eval(); ps = []
    for Hb, mk, b in batches(D, idx, dev):
        ps.append(net(Hb, mk).cpu().numpy())
    P = np.concatenate(ps).ravel()
    return float(np.abs(P - D["y"][idx]).mean())


def train_eval(spec, seed, D, dev, max_epochs=200, patience=20):
    torch.manual_seed(seed); rng = np.random.default_rng(100 + seed)
    base = pm.build_base(spec, dev, out=1, drop=spec.get("drop", 0.2))
    net = pm.wrap(base, D["mean"], D["std"], dev)
    opt = torch.optim.Adam(net.parameters(), LR, weight_decay=WD)
    best = {"val": float("inf"), "state": None, "epoch": -1}; bad = 0
    for ep in range(max_epochs):
        net.train()
        for Hb, mk, b in batches(D, D["tr"], dev, shuffle=True, rng=rng):
            opt.zero_grad()
            (((net(Hb, mk) - D["Y"][torch.as_tensor(b, device=dev)]) / D["sd"]) ** 2).mean().backward()
            opt.step()
        v = eval_mae(net, D, D["va"], dev)
        if v < best["val"] - 1e-4:
            best = {"val": v, "state": copy.deepcopy({k: x.cpu() for k, x in net.base.state_dict().items()}),
                    "epoch": ep}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.base.load_state_dict(best["state"])
    m = {"val_mae": eval_mae(net, D, D["va"], dev), "test_mae": eval_mae(net, D, D["te"], dev),
         "epoch_best": best["epoch"], "n_train": int(len(D["tr"]))}
    return m, net.base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trait", choices=list(TRAITS), required=True)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfgs = make_configs()
    if a.dry_run:
        for c in cfgs:
            print(" ", label(c))
        print(f"{len(cfgs)} configs x 1 seed = {len(cfgs)} fits")
        return

    dev = device(); print(f"device: {dev}", flush=True)
    OUT = os.path.join(HERE, TRAITS[a.trait]["tdm"], "sweep"); os.makedirs(OUT, exist_ok=True)
    D = load_data(a.trait, dev)
    print(f"[{a.trait}] target={D['unit']} | N={D['N']} Lmax={D['Lmax']} | "
          f"split train={len(D['tr'])} val={len(D['va'])} test={len(D['te'])}", flush=True)

    per = []
    for ci, spec in enumerate(cfgs):
        path = os.path.join(OUT, f"{label(spec)}_s0.pt")
        if os.path.exists(path) and not a.force:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            m = {k: ck[k] for k in ("val_mae", "test_mae", "epoch_best", "n_train")}; tag = "cached "
        else:
            m, base = train_eval(spec, 0, D, dev, a.max_epochs, a.patience)
            pm.save_model(path, base, {**spec, "out": 1, "seed": 0, "trait": a.trait,
                                       "mean": D["mean"], "std": D["std"], **m})
            tag = "trained"
        per.append({"arch": spec["arch"], "pool": spec["pool"], "depth": spec["depth"],
                    "label": label(spec), **m, "ckpt": os.path.relpath(path, HERE)})
        print(f"[{ci+1:2}/{len(cfgs)}] {tag} {label(spec):22} "
              f"val {m['val_mae']:.3f}  test {m['test_mae']:.3f}  @ep{m['epoch_best']}", flush=True)

    per.sort(key=lambda r: r["val_mae"])
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0].keys())); w.writeheader(); w.writerows(per)
    json.dump({"trait": a.trait, "unit": D["unit"], "results": per},
              open(os.path.join(OUT, "results.json"), "w"), indent=2)

    print(f"\n=== {a.trait} leaderboard (val MAE, {D['unit']}) ===", flush=True)
    for r in per[:12]:
        print(f"{r['label']:22} val {r['val_mae']:.3f}  test {r['test_mae']:.3f}", flush=True)
    print(f"best: {per[0]['label']}  val {per[0]['val_mae']:.3f} {D['unit']}", flush=True)


if __name__ == "__main__":
    main()
