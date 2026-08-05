#!/usr/bin/env python3
"""Train the FINAL deployable surrogate: cnn-max-d1, the config the 3-fold CV confirmed as best
(see sweep_results.ipynb), refit on the full surrogate train+val pool (515 rows) instead of just
train (424).

CV was for architecture + epoch-budget selection, not deployment -- the 3 fold-models themselves
were never saved (each only saw 2/3 of the pool anyway). There's no held-out val left to early-stop
on once train+val are both used for fitting, so the epoch count is instead FIXED to the mean
epoch_best across cnn-max-d1's 3 CV folds (56, 69, 88 -> round(71.0) = 71), rather than reserving
part of train+val purely to decide when to stop.

The 91-row surrogate test is left completely untouched (not trained on) and evaluated once at the
end as a final honest check -- same fixed yardstick used throughout the CV work.

Usage:
    python train_final_surrogate.py
"""
import os

import numpy as np
import pandas as pd
import torch

# --- stage-folder bootstrap: put the experiment root (design_common), lib/ (vendored
# --- modules) and msa/ (family alignment code) on the import path.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, "lib"), _os.path.join(_ROOT, "msa")]

import peak_models as pm
import sweep_peak_oracle as swp

SPEC = {"arch": "cnn", "pool": "max", "n_conv": 1, "depth": 1}
CV_CSV = os.path.join(_ROOT, "trained_models", "surrogate_cv3.csv")
OUT = os.path.join(_ROOT, "trained_models", "surrogate_final", "cnn-max-d1_trainval.pt")


def main():
    cv = pd.read_csv(CV_CSV)
    epoch_budget = int(round(cv[cv.label == swp.label(SPEC)]["epoch_best"].mean()))
    print(f"epoch budget (mean CV epoch_best across 3 folds) = {epoch_budget}")

    dev = swp.device()
    D = swp.load_data("surrogate", dev, to_gpu=True)
    pool = np.concatenate([D["tr"], D["va"]])           # 515: everything except the held-out test
    test_idx = D["te"]                                   # 91: untouched throughout
    print(f"train+val pool = {len(pool)} | held-out test (untouched) = {len(test_idx)}")

    peaks_np = D["Pk"].cpu().numpy()
    mean = peaks_np[pool].mean(0).astype(np.float32); std = (peaks_np[pool].std(0) + 1e-6).astype(np.float32)

    torch.manual_seed(0); rng = np.random.default_rng(100)
    base = pm.build_base({**SPEC, "d_in": D["d_in"]}, dev, out=2, drop=SPEC.get("drop", 0.2))
    net = pm.wrap(base, mean, std, dev)
    sd = torch.tensor(std, device=dev)
    opt = torch.optim.Adam(net.parameters(), swp.base.LR, weight_decay=swp.base.WD)

    for ep in range(epoch_budget):
        net.train()
        for Hb, mk, b in swp.base.batches(D, pool, dev, shuffle=True, rng=rng):
            opt.zero_grad()
            (((net(Hb, mk) - D["Pk"][torch.as_tensor(b, device=dev)]) / sd) ** 2).mean().backward()
            opt.step()
        if (ep + 1) % 10 == 0 or ep == epoch_budget - 1:
            te = swp.base.eval_mae(net, D, test_idx, dev)[0]
            print(f"  epoch {ep + 1}/{epoch_budget}: held-out test MAE {te:.2f} nm", flush=True)

    te_mae, te_ex, te_em = swp.base.eval_mae(net, D, test_idx, dev)
    print(f"FINAL held-out test MAE {te_mae:.2f} nm (ex {te_ex:.2f} / em {te_em:.2f})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pm.save_model(OUT, net.base, {
        **SPEC, "d_in": D["d_in"], "out": 2, "seed": 0, "role": "surrogate_final", "emb": "esm",
        "mean": mean, "std": std, "n_train": int(len(pool)), "epoch_budget": epoch_budget,
        "cv_epoch_bests": cv[cv.label == swp.label(SPEC)]["epoch_best"].tolist(),
        "test_mae": float(te_mae), "test_ex": float(te_ex), "test_em": float(te_em),
        "note": "final refit on surrogate train+val (515), epoch budget fixed from 3-fold CV; "
                "test (91) never trained on.",
    })
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
