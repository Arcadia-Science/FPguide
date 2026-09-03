"""Shared FP peak-wavelength surrogate/oracle models + ESM-2 per-residue embedding utilities.

The peak-wavelength counterpart of the retired full-spectrum ``fp_models.py`` (that experiment
is no longer in the repo). Instead of predicting the full 1002-dim excitation/emission
spectrum, every model here regresses just the **two peak wavelengths** ``(ex_max, em_max)``
(nm) taken over the full spectrum — i.e. we condition design on the peaks rather than the
whole curve.

Backbones: a **CNN** (1-D convolutions over residues) and a **Transformer encoder**, each with a
``mean | min | max | std | concat | concatstd`` masked-pooling readout, a learned ``attn`` pool, or a
learned second-order ``cov`` (covariance-probe) pool over the ESM-2 / ProstT5 per-residue embeddings.
Used by ``in-silico-test/1_surrogate_oracle_training/sweep_peak_oracle.py`` and
``in-silico-test/sweep_results.ipynb`` — and by the archived ``train_oracle_prostt5.py`` and
``surrogate_oracle_peak_dual.ipynb`` under ``archive/esm2_design/``, which is gitignored and not
in a clone — so saved checkpoints reconstruct identically wherever they are loaded.
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


# pooling readouts. Stat pools are parameter-free reductions; 'attn' is a learned attention pool;
# 'cov' is a learned second-order (covariance-probe) pool.
STAT_POOLS = {
    "mean": ["mean"], "min": ["min"], "max": ["max"], "std": ["std"],
    "concat": ["mean", "max", "min"],            # default multi-stat readout
    "concatstd": ["mean", "max", "min", "std"],  # + dispersion
}
POOLS = ["mean", "min", "max", "std", "concat", "concatstd", "attn", "cov"]
CONCAT = STAT_POOLS["concat"]                     # back-compat alias
COV_PROBE_DIM = 32                                # default probe width p for covariance pooling


def _pool_kinds(pool):
    return STAT_POOLS.get(pool, [pool])


class Readout(nn.Module):
    """Collapse (B, L, C) over valid residues to a fixed vector.

    Stat pools concatenate parameter-free reductions (out_dim = C * n_stats). 'attn' is a learned
    single-query masked attention pool (out_dim = C). 'cov' is a learned covariance-probe pool:
    a linear probe z_i = Wᵀ x_i (W: C -> p) followed by the (uncentered) second-moment matrix
    C_z = (1/L) Σ_i z_i z_iᵀ over the L valid residues, flattened over its upper triangle
    (out_dim = p(p+1)/2). Stat pools add no parameters, so existing checkpoints (mean/min/max/
    concat) reconstruct with identical state_dict keys.
    """

    def __init__(self, pool, C, probe_dim=COV_PROBE_DIM):
        super().__init__()
        self.pool = pool
        if pool == "attn":
            self.score = nn.Linear(C, 1)
            self.out_dim = C
        elif pool == "cov":
            self.probe_dim = probe_dim
            self.probe = nn.Linear(C, probe_dim, bias=False)     # W: z_i = Wᵀ x_i
            iu, ju = torch.triu_indices(probe_dim, probe_dim)     # upper triangle incl. diagonal
            self.register_buffer("_iu", iu, persistent=False)
            self.register_buffer("_ju", ju, persistent=False)
            self.out_dim = probe_dim * (probe_dim + 1) // 2
        else:
            self.kinds = STAT_POOLS[pool]
            self.out_dim = C * len(self.kinds)

    def forward(self, x, mask):
        if self.pool == "attn":
            s = self.score(x).squeeze(-1).masked_fill(~mask, float("-inf"))
            w = torch.softmax(s, dim=1).unsqueeze(-1)
            return (x * w).sum(1)
        if self.pool == "cov":
            mk = mask.unsqueeze(-1)
            z = self.probe(x) * mk                                # (B, L, p); zero padded residues
            L = mk.sum(1).clamp(min=1).unsqueeze(-1)              # (B, 1, 1) valid-residue count
            Cz = torch.einsum("blp,blq->bpq", z, z) / L          # (B, p, p) = (1/L) Σ z_i z_iᵀ
            return Cz[:, self._iu, self._ju]                     # (B, p(p+1)/2) upper triangle
        return torch.cat([masked_pool(x, mask, k) for k in self.kinds], dim=-1)


def mlp_head(feat, hidden, nl, drop, out):
    """MLP head: (feat) -> hidden x(nl) -> out. Matches the previous inline head structure."""
    layers = [nn.Linear(feat, hidden), nn.ReLU()]
    for _ in range(nl - 1):
        layers += [nn.Dropout(drop), nn.Linear(hidden, hidden), nn.ReLU()]
    layers += [nn.Linear(hidden, out)]
    return nn.Sequential(*layers)


class PeakCNN(nn.Module):
    """Conv1d stack over per-residue embeddings -> readout -> MLP head -> (ex_max, em_max).

    n_conv conv layers (default 2 = the original depth). n_conv=0 is a pooling-only MLP baseline
    (no residue mixing; the readout runs directly over the raw ESM dims).
    """

    def __init__(self, pool, d_in=D_IN, conv_ch=128, k=5, n_conv=2, hidden=256, nl=2, drop=0.2,
                 out=N_PEAKS, probe_dim=COV_PROBE_DIM):
        super().__init__()
        self.pool = pool
        self.n_conv = n_conv
        convs, c = [], d_in
        for _ in range(n_conv):
            convs += [nn.Conv1d(c, conv_ch, k, padding=k // 2), nn.ReLU()]
            c = conv_ch
        self.conv = nn.Sequential(*convs)
        C = conv_ch if n_conv > 0 else d_in
        self.readout = Readout(pool, C, probe_dim=probe_dim)
        self.head = mlp_head(self.readout.out_dim, hidden, nl, drop, out)

    def forward(self, Hb, mask):
        x = self.conv(Hb.transpose(1, 2)).transpose(1, 2) if self.n_conv > 0 else Hb
        return self.head(self.readout(x, mask))


class PeakTransformer(nn.Module):
    """Project embeddings -> Transformer encoder over residues -> masked pool -> MLP head -> (ex_max, em_max).

    Self-attention refines the (already positionally-informed) ESM-2 embeddings; the padding mask keeps
    attention/pooling on valid residues only. pool: 'mean'|'min'|'max'|'concat' (mean+max+min).
    """

    def __init__(self, pool, d_in=D_IN, d_model=128, nhead=4, nlayers=2, ff=256,
                 hidden=256, nl=2, drop=0.2, out=N_PEAKS, probe_dim=COV_PROBE_DIM):
        super().__init__()
        self.pool = pool
        self.proj = nn.Linear(d_in, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=ff, dropout=drop,
                                         batch_first=True, activation="relu")
        # enable_nested_tensor=False keeps the padding-masked path on GPU (the nested-tensor
        # fast-path falls back to CPU on MPS); weights are unchanged either way.
        self.encoder = nn.TransformerEncoder(enc, nlayers, enable_nested_tensor=False)
        self.readout = Readout(pool, d_model, probe_dim=probe_dim)
        self.head = mlp_head(self.readout.out_dim, hidden, nl, drop, out)

    def forward(self, Hb, mask):
        x = self.encoder(self.proj(Hb), src_key_padding_mask=~mask)   # True = ignore (padding)
        return self.head(self.readout(x, mask))


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
    """Construct the trainable base net (predicts standardized peaks) from a spec dict.

    spec['arch'] in {'cnn', 'mlp', 'transformer'}: 'mlp' is a pooling-only baseline (cnn with n_conv=0).
    spec['d_in'] sets the per-residue embedding dim (default D_IN=1280 for ESM-2 650M; use 1024 for
    the ProstT5 encoder). Baked into the checkpoint so load_model reconstructs the right shape.
    """
    arch = spec["arch"]
    d_in = spec.get("d_in", D_IN)
    probe_dim = spec.get("probe_dim", COV_PROBE_DIM)     # only used by the 'cov' readout
    if arch in ("cnn", "mlp"):
        n_conv = 0 if arch == "mlp" else spec.get("n_conv", 2)
        return PeakCNN(spec["pool"], d_in=d_in, conv_ch=spec.get("conv_ch", 128), k=spec.get("k", 5),
                       n_conv=n_conv, hidden=spec.get("hidden", 256), nl=spec.get("nl", 2),
                       drop=drop, out=out, probe_dim=probe_dim).to(dev)
    return PeakTransformer(spec["pool"], d_in=d_in, d_model=spec.get("d_model", 128), nhead=spec.get("nhead", 4),
                           nlayers=spec.get("nlayers", 2), ff=spec.get("ff", 256),
                           hidden=spec.get("hidden", 256), nl=spec.get("nl", 2), drop=drop, out=out,
                           probe_dim=probe_dim).to(dev)


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
