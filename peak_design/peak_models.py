"""Shared FP peak-wavelength surrogate/oracle models + ESM-2 per-residue embedding utilities.

The *peak_design* analog of ``../design/fp_models.py``. Instead of predicting the full
1002-dim excitation/emission spectrum, every model here regresses just the **two peak
wavelengths** ``(ex_max, em_max)`` (nm) taken over the full spectrum — i.e. we condition
design on the peaks rather than the whole curve.

Backbones: a **CNN** (1-D convolutions over residues) and a **Transformer encoder**, each
with a ``max | min | mean | concat`` masked-pooling readout over the ESM-2 per-residue
embeddings. Used by ``surrogate_oracle_peak_dual.ipynb`` and ``guided_design_peak.ipynb`` so
saved checkpoints reconstruct identically wherever they are loaded.
"""
from __future__ import annotations
import torch
import torch.nn as nn

D_IN = 1280          # ESM-2 650M (esm2_t33_650M_UR50D) embedding dim
ESM_LAYER = 33
N_PEAKS = 2          # (ex_max, em_max)


def masked_pool(x, mask, kind):
    """Reduce (B, L, C) over valid residues. kind in {mean, max, min, std}."""
    mk = mask.unsqueeze(-1)
    cnt = mk.sum(1).clamp(min=1)
    mean = (x * mk).sum(1) / cnt
    if kind == "mean":
        return mean
    if kind == "max":
        return x.masked_fill(~mk, float("-inf")).max(1).values
    if kind == "min":
        return x.masked_fill(~mk, float("inf")).min(1).values
    if kind == "std":
        var = (((x - mean.unsqueeze(1)) ** 2) * mk).sum(1) / cnt
        return torch.sqrt(var + 1e-8)
    raise ValueError(kind)


# pooling readouts exposed to the notebooks
POOLS = ["mean", "min", "max", "concat"]
CONCAT = ["mean", "max", "min"]          # what 'concat' expands to (no std, matching the requested set)


def _pool_kinds(pool):
    return CONCAT if pool == "concat" else [pool]


class PeakCNN(nn.Module):
    """Conv1d over per-residue embeddings -> masked pool -> MLP head -> (ex_max, em_max).

    pool: 'mean'|'min'|'max', or 'concat' (concatenate mean+max+min).
    """

    def __init__(self, pool, d_in=D_IN, conv_ch=128, k=5, hidden=256, nl=2, drop=0.2, out=N_PEAKS):
        super().__init__()
        self.pool = pool
        self.kinds = _pool_kinds(pool)
        self.conv = nn.Sequential(nn.Conv1d(d_in, conv_ch, k, padding=k // 2), nn.ReLU(),
                                  nn.Conv1d(conv_ch, conv_ch, k, padding=k // 2), nn.ReLU())
        feat = conv_ch * len(self.kinds)
        layers = [nn.Linear(feat, hidden), nn.ReLU()]
        for _ in range(nl - 1):
            layers += [nn.Dropout(drop), nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, out)]
        self.head = nn.Sequential(*layers)

    def forward(self, Hb, mask):
        x = self.conv(Hb.transpose(1, 2)).transpose(1, 2)
        pooled = torch.cat([masked_pool(x, mask, kk) for kk in self.kinds], dim=-1)
        return self.head(pooled)


class PeakTransformer(nn.Module):
    """Project embeddings -> Transformer encoder over residues -> masked pool -> MLP head -> (ex_max, em_max).

    Self-attention refines the (already positionally-informed) ESM-2 embeddings; the padding mask keeps
    attention/pooling on valid residues only. pool: 'mean'|'min'|'max'|'concat' (mean+max+min).
    """

    def __init__(self, pool, d_in=D_IN, d_model=128, nhead=4, nlayers=2, ff=256,
                 hidden=256, nl=2, drop=0.2, out=N_PEAKS):
        super().__init__()
        self.pool = pool
        self.kinds = _pool_kinds(pool)
        self.proj = nn.Linear(d_in, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=ff, dropout=drop,
                                         batch_first=True, activation="relu")
        # enable_nested_tensor=False keeps the padding-masked path on GPU (the nested-tensor
        # fast-path falls back to CPU on MPS); weights are unchanged either way.
        self.encoder = nn.TransformerEncoder(enc, nlayers, enable_nested_tensor=False)
        feat = d_model * len(self.kinds)
        layers = [nn.Linear(feat, hidden), nn.ReLU()]
        for _ in range(nl - 1):
            layers += [nn.Dropout(drop), nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, out)]
        self.head = nn.Sequential(*layers)

    def forward(self, Hb, mask):
        x = self.proj(Hb)
        x = self.encoder(x, src_key_padding_mask=~mask)        # True = ignore (padding)
        pooled = torch.cat([masked_pool(x, mask, kk) for kk in self.kinds], dim=-1)
        return self.head(pooled)


class StandardizedPeaks(nn.Module):
    """Wrap a base net that predicts standardized peaks so forward() returns peaks in **nm**.

    z = base(H, mask)  ->  z * peak_std + peak_mean.
    The base net (trainable) is what gets saved/loaded; the (mean, std) are held as buffers and
    re-attached from the dataset's train-split statistics wherever the model is reconstructed
    (mirrors how NormalizedSpectrum re-attaches its PCA basis in ../design).
    """

    def __init__(self, base, mean, std):
        super().__init__()
        self.base = base
        self.register_buffer("peak_mean", torch.as_tensor(mean, dtype=torch.float32))  # (2,)
        self.register_buffer("peak_std", torch.as_tensor(std, dtype=torch.float32))    # (2,)

    def forward(self, Hb, mask):
        return self.base(Hb, mask) * self.peak_std + self.peak_mean


# ---- checkpoint save/load ----
def save_model(path, net, meta):
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, **meta}, path)


def build_base(spec, dev, out=N_PEAKS, drop=0.2):
    """Construct the trainable base net (predicts standardized peaks) from a spec dict."""
    if spec["arch"] == "cnn":
        return PeakCNN(spec["pool"], conv_ch=spec.get("conv_ch", 128), k=spec.get("k", 5),
                       hidden=spec.get("hidden", 256), nl=spec.get("nl", 2), drop=drop, out=out).to(dev)
    return PeakTransformer(spec["pool"], d_model=spec.get("d_model", 128), nhead=spec.get("nhead", 4),
                           nlayers=spec.get("nlayers", 2), ff=spec.get("ff", 256),
                           hidden=spec.get("hidden", 256), nl=spec.get("nl", 2), drop=drop, out=out).to(dev)


def load_model(path, dev, out=N_PEAKS):
    """Reconstruct the base net (standardized-peak predictor) from a checkpoint. Returns (net, ck)."""
    # weights_only=False: our own checkpoints carry a metadata dict (metrics, scaler stats) with
    # numpy scalars, which the PyTorch 2.6+ weights_only=True default refuses to unpickle.
    ck = torch.load(path, map_location=dev, weights_only=False)
    out = ck.get("out", out)
    net = build_base(ck, dev, out=out, drop=ck.get("drop", 0.2))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


def wrap(base, mean, std, dev):
    """Bake the inverse-standardization into the model so forward() returns nm."""
    return StandardizedPeaks(base, mean, std).to(dev)


def load_wrapped(path, mean, std, dev, out=N_PEAKS):
    base, ck = load_model(path, dev, out=out)
    return wrap(base, mean, std, dev), ck


# ---- ESM-2 per-residue embeddings (lazy-loaded, for novel sequences) ----
_ESM = None


def get_esm(dev):
    global _ESM
    if _ESM is None:
        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.eval().to(dev)
        for p in model.parameters():
            p.requires_grad_(False)
        _ESM = (model, alphabet, alphabet.get_batch_converter())
    return _ESM


@torch.no_grad()
def resid_embed(seqs, dev, bs=16):
    """List[str] (or str) -> (H:(B,Lmax,1280), mask:(B,Lmax) bool) on `dev`."""
    if isinstance(seqs, str):
        seqs = [seqs]
    model, alphabet, bc = get_esm(dev)
    Lmax = max(len(s) for s in seqs)
    H = torch.zeros(len(seqs), Lmax, D_IN)
    mask = torch.zeros(len(seqs), Lmax, dtype=torch.bool)
    for i in range(0, len(seqs), bs):
        ch = seqs[i:i + bs]
        _, _, tk = bc([(f"s{j}", s) for j, s in enumerate(ch)])
        reps = model(tk.to(dev), repr_layers=[ESM_LAYER])["representations"][ESM_LAYER]
        for j, s in enumerate(ch):
            n = len(s)
            H[i + j, :n] = reps[j, 1:1 + n].float().cpu()
            mask[i + j, :n] = True
    return H.to(dev), mask.to(dev)


def peaks_fn(net, dev):
    """Return f(seqs)->(B,2) tensor of (ex_max, em_max) in nm: embeds novel sequences and runs the wrapped net."""
    @torch.no_grad()
    def f(seqs):
        H, mask = resid_embed(seqs, dev)
        return net(H, mask)
    return f
