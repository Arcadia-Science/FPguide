#!/usr/bin/env python3
"""Cache ProstT5 encoder per-residue embeddings for the curated peak dataset (oracle input).

The ESM-2 counterpart is ``../dataset_pipeline/embed.py``. This writes, into the same curated folder
and row-aligned to ``peaks_assignments.csv`` (same index order as ``peaks.npy`` / the FASTA), a
ProstT5 embedding cache the retrained oracle consumes:

    prostt5_residue_fp16.npy   (N, Lmax, 1024) float16   -- H[i, :len_i] are the residue embeddings
    prostt5_residue_len.npy    (N,) int64                -- sequence lengths

Kept separate from the ESM-2 cache (``esm_residue_fp16.npy``) so the surrogate (ESM-2) and the oracle
(ProstT5) each read their own embeddings. Weights download on first use; device auto-detected.

Usage:
    python embed_prostt5.py                # build if missing
    python embed_prostt5.py --force        # rebuild
    python embed_prostt5.py --bs 4         # smaller batch (long sequences / tight memory)
    python embed_prostt5.py --dry-run      # report shapes without loading ProstT5
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

import prostt5_embed as pe

HERE = os.path.dirname(os.path.abspath(__file__))
CUR = os.path.join(HERE, "..", "dataset_pipeline", "data", "peak", "curated")
ASSIGN = "peaks_assignments.csv"


def load_seqs():
    rows = list(csv.DictReader(open(os.path.join(CUR, ASSIGN))))
    rows.sort(key=lambda r: int(r["index"]))          # enforce alignment to peaks.npy
    return [r["seq"] for r in rows]


def get_device():
    import torch
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild caches that already exist")
    ap.add_argument("--bs", type=int, default=8, help="ProstT5 batch size")
    ap.add_argument("--dry-run", action="store_true", help="report shapes without loading ProstT5")
    a = ap.parse_args()

    seqs = load_seqs()
    N = len(seqs)
    Ls = np.array([len(s) for s in seqs], dtype=np.int64)
    Lmax = int(Ls.max())
    emb = os.path.join(CUR, "prostt5_residue_fp16.npy")
    ln = os.path.join(CUR, "prostt5_residue_len.npy")
    tag = f"[peak] N={N}  Lmax={Lmax}  D={pe.D_IN_PROSTT5}  (~{N * Lmax * pe.D_IN_PROSTT5 * 2 / 1e6:.0f} MB)"

    if a.dry_run:
        print(f"{tag}  dry-run -> {os.path.relpath(emb, HERE)}")
        return
    if os.path.exists(emb) and os.path.exists(ln) and not a.force:
        print(f"{tag}  cache exists -> skip (use --force to rebuild)")
        return

    dev = get_device()
    print(f"device: {dev}\n{tag}")
    H = np.zeros((N, Lmax, pe.D_IN_PROSTT5), dtype=np.float16)
    for i0 in range(0, N, a.bs):
        chunk = seqs[i0:i0 + a.bs]
        Hm, _ = pe.resid_embed_prostt5(chunk, dev, bs=a.bs)
        for k, s in enumerate(chunk):
            H[i0 + k, :len(s)] = Hm[k, :len(s)].cpu().numpy().astype(np.float16)
        print(f"  {min(i0 + a.bs, N)}/{N}", end="\r", flush=True)
    np.save(emb, H)
    np.save(ln, Ls)
    print(f"\nwrote {os.path.relpath(emb, HERE)} + prostt5_residue_len.npy")


if __name__ == "__main__":
    main()
