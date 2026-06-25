"""Shared FP-spectrum surrogate/oracle models + ESM-2 per-residue embedding utilities.

Used by surrogate_model_design*.ipynb, finalize_models.ipynb, and guided_design_approach1.ipynb so the
saved checkpoints reconstruct identically wherever they are loaded.
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


class SpectrumNet(nn.Module):
    """MLP (pool raw embeddings) or CNN (conv over residues -> pool) -> spectrum."""
    def __init__(self, pool, use_cnn, d_in=D_IN, conv_ch=128, k=5, hidden=256, nl=2, drop=0.2, out=1002):
        super().__init__()
        self.pool = pool
        self.use_cnn = use_cnn
        if use_cnn:
            self.conv = nn.Sequential(nn.Conv1d(d_in, conv_ch, k, padding=k // 2), nn.ReLU(),
                                      nn.Conv1d(conv_ch, conv_ch, k, padding=k // 2), nn.ReLU())
            feat = conv_ch
        else:
            self.conv = None
            feat = d_in
        layers = [nn.Linear(feat, hidden), nn.ReLU()]
        for _ in range(nl - 1):
            layers += [nn.Dropout(drop), nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, out)]
        self.head = nn.Sequential(*layers)

    def forward(self, Hb, mask):
        x = Hb
        if self.use_cnn:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.head(masked_pool(x, mask, self.pool))


# ---- checkpoint save/load ----
def save_model(path, net, meta):
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, **meta}, path)


def load_model(path, dev, out=1002):
    ck = torch.load(path, map_location=dev)
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
