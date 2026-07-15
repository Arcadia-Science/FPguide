"""ProstT5 encoder per-residue embedding utility — the oracle's structure-aware alternative to ESM-2.

ProstT5 (``Rostlab/ProstT5``, MIT license) is a T5 encoder-decoder fine-tuned from ProtT5-XL-U50 to
translate between amino-acid sequences and Foldseek 3Di structure tokens. Here we use the **encoder
only** as a frozen feature extractor: fed an amino-acid sequence (prefixed with ``<AA2fold>``) it emits
a **1024-dim** per-residue embedding that carries structure-aware information distilled from AlphaFoldDB.

This mirrors ``peak_models.resid_embed`` (ESM-2) so it is a drop-in swap for the *oracle*: same
``(H:(B,Lmax,1024), mask:(B,Lmax))`` return contract. Per the model card, AA input is upper-cased,
the rare residues U/Z/O/B are mapped to X, tokens are whitespace-separated, and the ``<AA2fold>``
directional prefix is prepended. The encoder prepends that prefix token and appends an EOS; we strip
both so row ``i`` holds exactly ``len(seq_i)`` residue vectors, identical alignment to the ESM path.

Weights download on first use (~2.5 GB in half precision). Device auto-detected (CUDA -> MPS -> CPU);
on Apple Silicon set PYTORCH_ENABLE_MPS_FALLBACK=1 for any unsupported op.
"""
from __future__ import annotations

import re

import torch

D_IN_PROSTT5 = 1024          # ProstT5 encoder hidden size
MODEL_NAME = "Rostlab/ProstT5"
_PROSTT5 = None


def get_prostt5(dev):
    """Lazy-load the frozen ProstT5 encoder + tokenizer. Half precision on GPU/MPS, float32 on CPU."""
    global _PROSTT5
    if _PROSTT5 is None:
        from transformers import T5EncoderModel, T5Tokenizer
        tok = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False, legacy=True)
        model = T5EncoderModel.from_pretrained(MODEL_NAME).to(dev)
        # half precision everywhere except CPU (no fp16 matmul on CPU)
        model = model.half() if dev.type != "cpu" else model.float()
        model = model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _PROSTT5 = (model, tok)
    return _PROSTT5


def _prep(seqs):
    """AA sequences -> ProstT5 input strings: U/Z/O/B->X, upper-case, space-separated, <AA2fold> prefix."""
    out = []
    for s in seqs:
        s = re.sub(r"[UZOB]", "X", s.upper())
        out.append("<AA2fold> " + " ".join(list(s)))
    return out


@torch.no_grad()
def resid_embed_prostt5(seqs, dev, bs=8):
    """List[str] (or str) -> (H:(B,Lmax,1024), mask:(B,Lmax) bool) on ``dev``.

    Drop-in analog of ``peak_models.resid_embed`` but from the ProstT5 encoder. Residue ``j`` of
    sequence ``i`` lands at ``H[i, j]``; the ``<AA2fold>`` prefix token and trailing EOS are removed.
    """
    if isinstance(seqs, str):
        seqs = [seqs]
    model, tok = get_prostt5(dev)
    Lmax = max(len(s) for s in seqs)
    H = torch.zeros(len(seqs), Lmax, D_IN_PROSTT5)
    mask = torch.zeros(len(seqs), Lmax, dtype=torch.bool)
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        enc = tok(_prep(chunk), add_special_tokens=True,
                  padding="longest", return_tensors="pt")
        ids = enc["input_ids"].to(dev)
        am = enc["attention_mask"].to(dev)
        reps = model(input_ids=ids, attention_mask=am).last_hidden_state  # (B, 1+L+1(+pad), 1024)
        for j, s in enumerate(chunk):
            n = len(s)
            # token 0 is the <AA2fold> prefix; residues are tokens 1..n; token n+1 is EOS
            H[i + j, :n] = reps[j, 1:1 + n].float().cpu()
            mask[i + j, :n] = True
    return H.to(dev), mask.to(dev)


def prostt5_peaks_fn(net, dev):
    """f(seqs)->(B,2) (ex_max, em_max) nm: embed novel sequences with ProstT5 and run the wrapped oracle."""
    @torch.no_grad()
    def f(seqs):
        H, mask = resid_embed_prostt5(seqs, dev)
        return net(H, mask)
    return f
