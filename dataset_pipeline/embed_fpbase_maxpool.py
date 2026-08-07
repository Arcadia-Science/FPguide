#!/usr/bin/env python3
"""Embed every FPbase protein that has a sequence with ESM-2 and cache the
**max-pooled** representation, for the sequence-space map in
``visualize_curation.ipynb``.

Max pooling takes, for each of the 1280 embedding channels, the largest value over
the sequence's residues. Compared with mean pooling it emphasises the presence of a
distinctive local motif anywhere in the chain rather than the average character of the
whole chain, so short functional signatures survive the pooling step instead of being
diluted by length.

This embeds the **full** FPbase export -- including the cofactor / FAST / opsin proteins
that ``build_dataset.py`` drops -- so the map can show where the excluded families sit
relative to the curated set.

Row order = the order proteins with a sequence appear in ``fpbase_proteins.json``, and a
``seq_md5`` over that same order is stored in the metadata so the notebook can assert the
cache still lines up with the protein table.

Writes into ``data/``:
    fpbase_esm2_650M_max.npy        (N, 1280) float32
    fpbase_esm2_650M_max.meta.json  provenance: model, layer, pooling, n, slugs, seq_md5

Model: esm2_t33_650M_UR50D (weights download on first use). Device auto-detected
(CUDA -> MPS -> CPU). On Apple Silicon set PYTORCH_ENABLE_MPS_FALLBACK=1.

Usage:
    python embed_fpbase_maxpool.py           # skip if a valid cache already exists
    python embed_fpbase_maxpool.py --force   # rebuild
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROT_JSON = os.path.join(HERE, "..", "fpbase-extractor", "fpbase_output", "fpbase_proteins.json")
OUT_NPY = os.path.join(HERE, "data", "fpbase_esm2_650M_max.npy")
OUT_META = os.path.join(HERE, "data", "fpbase_esm2_650M_max.meta.json")
# fpdesign.peak_models holds the shared ESM-2 loader / per-residue embedding helper
sys.path.insert(0, os.path.join(HERE, ".."))

MODEL_NAME = "esm2_t33_650M_UR50D"
CHUNK = 8


def load_records():
    """(slugs, seqs, seq_md5) for every protein with a sequence, in JSON order."""
    proteins = json.load(open(PROT_JSON))
    slugs, raws = [], []
    for p in proteins:
        raw = (p.get("seq") or "").strip().upper()
        if not raw:
            continue
        slugs.append(p.get("slug", ""))
        raws.append(raw)
    md5 = hashlib.md5("\x00".join(raws).encode()).hexdigest()
    return slugs, raws, md5


def valid_cache(md5):
    if not (os.path.exists(OUT_NPY) and os.path.exists(OUT_META)):
        return False
    meta = json.load(open(OUT_META))
    return (meta.get("seq_md5") == md5
            and meta.get("model") == MODEL_NAME
            and meta.get("pooling") == "max")


def get_device():
    import torch
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild an existing cache")
    a = ap.parse_args()

    slugs, raws, md5 = load_records()
    n = len(raws)
    print(f"{n} FPbase proteins with a sequence (Lmax={max(len(s) for s in raws)})")

    if valid_cache(md5) and not a.force:
        print(f"valid cache exists -> {os.path.relpath(OUT_NPY, HERE)} (use --force to rebuild)")
        return

    import torch
    from fpdesign import peak_models as pm

    dev = get_device()
    print(f"device: {dev} | model: {MODEL_NAME} | pooling: max")

    # ESM's alphabet covers X/B/U/Z/O; map anything outside it to X defensively
    alpha = set(pm.get_esm(dev)[1].tok_to_idx)
    esm_seqs = ["".join(c if c in alpha else "X" for c in s) for s in raws]

    emb = np.zeros((n, pm.D_IN), dtype=np.float32)
    for i0 in range(0, n, CHUNK):
        chunk = esm_seqs[i0:i0 + CHUNK]
        H, mask = pm.resid_embed(chunk, dev)               # (b, Lmax, 1280), (b, Lmax)
        # padding must not win the max, so push padded positions to -inf first
        neg_inf = torch.finfo(H.dtype).min
        pooled = H.masked_fill(~mask.unsqueeze(-1), neg_inf).max(dim=1).values
        emb[i0:i0 + len(chunk)] = pooled.cpu().numpy().astype(np.float32)
        print(f"  {min(i0 + CHUNK, n)}/{n}", end="\r", flush=True)

    os.makedirs(os.path.dirname(OUT_NPY), exist_ok=True)
    np.save(OUT_NPY, emb)
    json.dump({
        "model": MODEL_NAME, "layer": pm.ESM_LAYER, "pooling": "max",
        "dim": pm.D_IN, "n": n, "seq_md5": md5, "slugs": slugs,
    }, open(OUT_META, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(OUT_NPY, HERE)} {emb.shape} + meta")


if __name__ == "__main__":
    main()
