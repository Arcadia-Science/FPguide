#!/usr/bin/env python3
"""Embed the curated sequences of each dataset with ESM-2 and cache them per trait.

For each trait, writes into its own data/<trait>/curated/ folder a per-residue ESM-2 embedding
cache, row-aligned to that trait's curated data (same index order as <target>.npy / the FASTA /
the assignments CSV):

    esm_residue_fp16.npy   (N, Lmax, 1280) float16   -- H[i, :len_i] are the residue embeddings
    esm_residue_len.npy    (N,) int64                -- sequence lengths

Each trait is embedded independently: a sequence shared across traits is embedded once per trait.
This duplicates a little computation, but keeps every dataset fully self-contained -- no cross-trait
cache sharing, no index bookkeeping. The learning-curve notebooks load these files directly.

ESM-2 (esm2_t33_650M_UR50D) weights download on first use; device is auto-detected (CUDA -> MPS -> CPU).
On Apple Silicon set PYTORCH_ENABLE_MPS_FALLBACK=1 for any unsupported op.

Usage:
    python embed.py                 # all three traits (skips ones already cached)
    python embed.py --trait pka     # one trait
    python embed.py --all --force   # rebuild every cache
    python embed.py --dry-run       # report N/Lmax/size per trait, without loading ESM-2
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# peak_models (ESM-2 embedding utilities) lives in the sibling peak_design/ folder
sys.path.insert(0, os.path.join(HERE, "..", "peak_design"))

TRAITS = {"peak": "peaks_assignments.csv",
          "brightness": "brightness_assignments.csv",
          "pka": "pka_assignments.csv"}
CHUNK = 32   # sequences per accumulation step (resid_embed sub-batches internally)


def load_seqs(trait):
    cur = os.path.join(HERE, "data", trait, "curated")
    rows = list(csv.DictReader(open(os.path.join(cur, TRAITS[trait]))))
    rows.sort(key=lambda r: int(r["index"]))          # enforce row alignment to <target>.npy
    return cur, [r["seq"] for r in rows]


def embed_trait(trait, dev, force=False, dry_run=False):
    cur, seqs = load_seqs(trait)
    N = len(seqs)
    Ls = np.array([len(s) for s in seqs], dtype=np.int64)
    Lmax = int(Ls.max())
    emb = os.path.join(cur, "esm_residue_fp16.npy")
    ln = os.path.join(cur, "esm_residue_len.npy")
    tag = f"[{trait:10}] N={N:4d}  Lmax={Lmax:4d}  (~{N * Lmax * 1280 * 2 / 1e6:.0f} MB)"

    if dry_run:
        print(f"{tag}  dry-run -> {os.path.relpath(emb, HERE)}")
        return
    if os.path.exists(emb) and os.path.exists(ln) and not force:
        print(f"{tag}  cache exists -> skip (use --force to rebuild)")
        return

    import peak_models as pm
    H = np.zeros((N, Lmax, pm.D_IN), dtype=np.float16)
    for i0 in range(0, N, CHUNK):
        chunk = seqs[i0:i0 + CHUNK]
        Hm, _ = pm.resid_embed(chunk, dev)
        for k, s in enumerate(chunk):
            H[i0 + k, :len(s)] = Hm[k, :len(s)].cpu().numpy().astype(np.float16)
        print(f"  {trait}: {min(i0 + CHUNK, N)}/{N}", end="\r", flush=True)
    np.save(emb, H)
    np.save(ln, Ls)
    print(f"\n{tag}  wrote {os.path.relpath(emb, HERE)} + esm_residue_len.npy")


def get_device():
    import torch
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trait", choices=list(TRAITS), help="embed a single trait")
    ap.add_argument("--all", action="store_true", help="embed all three traits (default)")
    ap.add_argument("--force", action="store_true", help="rebuild caches that already exist")
    ap.add_argument("--dry-run", action="store_true", help="report shapes without loading ESM-2")
    a = ap.parse_args()

    traits = [a.trait] if (a.trait and not a.all) else list(TRAITS)
    dev = None if a.dry_run else get_device()
    if not a.dry_run:
        print(f"device: {dev}")
    for t in traits:
        embed_trait(t, dev, force=a.force, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
