#!/usr/bin/env python3
"""Embed *every* FPbase protein that has a sequence with ESM-2 and cache the
mean-pooled representation, for the sequence-space map in
``protein_embedding_map.ipynb``.

Unlike ``dataset_pipeline/embed.py`` (which caches per-residue tensors for the
curated peak/brightness/pKa subsets), this embeds the **full** FPbase export --
including the cofactor / FAST / opsin proteins the training pipeline drops -- so
the map can show where those excluded families sit in embedding space.

Row order = the order proteins with a sequence appear in ``fpbase_proteins.json``
(the same order ``identity_all.npy`` uses), so the cache is row-aligned to the
notebook's protein table. Alignment is verified in the notebook via ``seq_md5``.

Writes into ``fpbase_output/``:
    esm2_650M_mean.npy        (N, 1280) float32   -- mean-pooled ESM-2 embedding
    esm2_650M_mean.meta.json  provenance: model, layer, pooling, n, slug order, seq_md5

Model: esm2_t33_650M_UR50D (weights download on first use). Device auto-detected
(CUDA -> MPS -> CPU). On Apple Silicon set PYTORCH_ENABLE_MPS_FALLBACK=1.

Usage:
    python embed_all.py               # skip if a valid cache already exists
    python embed_all.py --force       # rebuild
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROT_JSON = os.path.join(HERE, "fpbase_output", "fpbase_proteins.json")
OUT_NPY = os.path.join(HERE, "fpbase_output", "esm2_650M_mean.npy")
OUT_META = os.path.join(HERE, "fpbase_output", "esm2_650M_mean.meta.json")
# peak_models (ESM-2 utilities) lives in the sibling peak_design/ folder
sys.path.insert(0, os.path.join(HERE, "..", "peak_design"))

MODEL_NAME = "esm2_t33_650M_UR50D"
CHUNK = 16          # sequences per ESM forward accumulation step


def load_records():
    """(slugs, seqs_for_esm, seq_md5) for every protein with a sequence, in JSON order."""
    proteins = json.load(open(PROT_JSON))
    slugs, seqs, raws = [], [], []
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
    return meta.get("seq_md5") == md5 and meta.get("model") == MODEL_NAME


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
    N = len(raws)
    print(f"{N} FPbase proteins with a sequence (Lmax={max(len(s) for s in raws)})")

    if valid_cache(md5) and not a.force:
        print(f"valid cache exists -> {os.path.relpath(OUT_NPY, HERE)} (use --force to rebuild)")
        return

    import peak_models as pm  # lazy: only import torch/esm when we actually embed
    dev = get_device()
    print(f"device: {dev} | model: {MODEL_NAME}")

    # ESM alphabet handles X/B/U/Z/O; map anything else to X just in case
    alpha = set(pm.get_esm(dev)[1].tok_to_idx)
    esm_seqs = ["".join(c if c in alpha else "X" for c in s) for s in raws]

    emb = np.zeros((N, pm.D_IN), dtype=np.float32)
    for i0 in range(0, N, CHUNK):
        chunk = esm_seqs[i0:i0 + CHUNK]
        H, mask = pm.resid_embed(chunk, dev)              # (b, Lmax, 1280), (b, Lmax)
        m = mask.unsqueeze(-1).float()
        pooled = (H * m).sum(1) / m.sum(1).clamp(min=1)   # masked mean over residues
        emb[i0:i0 + len(chunk)] = pooled.cpu().numpy().astype(np.float32)
        print(f"  {min(i0 + CHUNK, N)}/{N}", end="\r", flush=True)

    np.save(OUT_NPY, emb)
    json.dump({
        "model": MODEL_NAME, "layer": pm.ESM_LAYER, "pooling": "mean",
        "dim": pm.D_IN, "n": N, "seq_md5": md5, "slugs": slugs,
    }, open(OUT_META, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(OUT_NPY, HERE)} {emb.shape} + meta")


if __name__ == "__main__":
    main()
