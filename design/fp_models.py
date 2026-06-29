"""Shared FP-spectrum surrogate/oracle models + ESM-2 per-residue embedding utilities.

Used by surrogate_model_design.ipynb, surrogate_model_design_dual.ipynb, and guided_design_approach1.ipynb so
the saved checkpoints reconstruct identically wherever they are loaded.
"""
from __future__ import annotations
import torch
import torch.nn as nn

D_IN = 1280          # ESM-2 650M (esm2_t33_650M_UR50D) embedding dim
ESM_LAYER = 33


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


def peak_normalize(spec, G, eps=1e-8):
    """Per-half (excitation|emission) peak-normalize a (..., 2G) spectrum so each half maxes at 1.

    Works for torch tensors (differentiable) and numpy arrays.
    """
    ex, em = spec[..., :G], spec[..., G:]
    if isinstance(spec, torch.Tensor):
        ex = ex / ex.amax(-1, keepdim=True).clamp_min(eps)
        em = em / em.amax(-1, keepdim=True).clamp_min(eps)
        return torch.cat([ex, em], dim=-1)
    import numpy as np
    ex = ex / np.clip(ex.max(-1, keepdims=True), eps, None)
    em = em / np.clip(em.max(-1, keepdims=True), eps, None)
    return np.concatenate([ex, em], axis=-1)


def cosine_loss(pred, target, G):
    """1 - mean over the batch of the average per-half cosine similarity. pred/target: (B, 2G)."""
    ex = nn.functional.cosine_similarity(pred[..., :G], target[..., :G], dim=-1)
    em = nn.functional.cosine_similarity(pred[..., G:], target[..., G:], dim=-1)
    return (1.0 - 0.5 * (ex + em)).mean()


class SpectrumNet(nn.Module):
    """MLP (pool raw embeddings) or CNN (conv over residues -> pool) -> spectrum.

    pool: 'mean'|'min'|'max'|'std', or 'concat' (concatenate all four).
    """
    CONCAT = ["mean", "max", "min", "std"]

    def __init__(self, pool, use_cnn, d_in=D_IN, conv_ch=128, k=5, hidden=256, nl=2, drop=0.2, out=1002):
        super().__init__()
        self.pool = pool
        self.use_cnn = use_cnn
        self.kinds = self.CONCAT if pool == "concat" else [pool]
        if use_cnn:
            self.conv = nn.Sequential(nn.Conv1d(d_in, conv_ch, k, padding=k // 2), nn.ReLU(),
                                      nn.Conv1d(conv_ch, conv_ch, k, padding=k // 2), nn.ReLU())
            base = conv_ch
        else:
            self.conv = None
            base = d_in
        feat = base * len(self.kinds)
        layers = [nn.Linear(feat, hidden), nn.ReLU()]
        for _ in range(nl - 1):
            layers += [nn.Dropout(drop), nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, out)]
        self.head = nn.Sequential(*layers)

    def forward(self, Hb, mask):
        x = Hb
        if self.use_cnn:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        pooled = torch.cat([masked_pool(x, mask, k) for k in self.kinds], dim=-1)
        return self.head(pooled)


class PooledMLP(nn.Module):
    """Pool per-residue embeddings (mean/min/max/std, or 'concat'=all four) -> MLP -> spectrum.

    Pools internally so it shares SpectrumNet's forward(H, mask) interface (works with spectrum_fn).
    """
    CONCAT = ["mean", "max", "min", "std"]

    def __init__(self, input, hidden, nl, d_in=D_IN, out=1002, drop=0.2):
        super().__init__()
        self.input = input
        self.kinds = self.CONCAT if input == "concat" else [input]
        d = d_in * len(self.kinds)
        layers = [nn.Linear(d, hidden), nn.ReLU()]
        for _ in range(nl - 1):
            layers += [nn.Dropout(drop), nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, Hb, mask):
        feat = torch.cat([masked_pool(Hb, mask, k) for k in self.kinds], dim=-1)
        return self.net(feat)


class NormalizedSpectrum(nn.Module):
    """Wrap a PCA-coefficient net so forward() returns a peak-normalized spectrum.

    coeffs = base(H, mask)  ->  PCA inverse_transform (fixed linear: coef @ components + mean)
                            ->  per-half (ex|em) max-normalization.
    The base net (trainable) is what gets saved/loaded; the PCA basis is held as buffers and
    re-attached via the dataset's fitted PCA wherever the model is reconstructed.
    """

    def __init__(self, base, components, mean, G):
        super().__init__()
        self.base = base
        self.G = G
        self.register_buffer("components", torch.as_tensor(components, dtype=torch.float32))  # (n_comp, 2G)
        self.register_buffer("pca_mean", torch.as_tensor(mean, dtype=torch.float32))          # (2G,)

    def forward(self, Hb, mask):
        coef = self.base(Hb, mask)
        spec = coef @ self.components + self.pca_mean      # PCA inverse_transform
        return peak_normalize(spec, self.G)


# ---- checkpoint save/load ----
def save_model(path, net, meta):
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, **meta}, path)


def load_model(path, dev, out=1002):
    ck = torch.load(path, map_location=dev)
    out = ck.get("out", out)   # prefer the output dim saved with the checkpoint (e.g. PCA n_components)
    if ck.get("kind") == "pooledmlp":
        net = PooledMLP(ck["input"], ck["hidden"], ck["nl"], out=out, drop=ck.get("drop", 0.2)).to(dev)
    else:  # SpectrumNet (MLP or CNN over per-residue embeddings)
        net = SpectrumNet(ck["pool"], ck["arch"] == "CNN", conv_ch=ck.get("conv_ch", 128), k=ck.get("k", 5),
                          hidden=ck["hidden"], nl=ck["nl"], drop=ck["drop"], out=out).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


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


def spectrum_fn(net, dev):
    """Return f(seqs)->(B,out) tensor that embeds novel sequences and runs the net."""
    @torch.no_grad()
    def f(seqs):
        H, mask = resid_embed(seqs, dev)
        return net(H, mask)
    return f


def pca_spectrum_fn(net, pca, dev, normalize=True):
    """Like spectrum_fn, but the net predicts PCA coefficients; reconstruct (and peak-normalize) to a spectrum.

    `normalize=True` applies per-half max-normalization so the returned spectrum peaks at 1 (matching the
    peak-normalized training targets and the NormalizedSpectrum model used at training time).
    """
    @torch.no_grad()
    def f(seqs):
        H, mask = resid_embed(seqs, dev)
        coef = net(H, mask).cpu().numpy()
        spec = pca.inverse_transform(coef)
        if normalize:
            spec = peak_normalize(spec, spec.shape[-1] // 2)
        return torch.tensor(spec, dtype=torch.float32, device=dev)
    return f
