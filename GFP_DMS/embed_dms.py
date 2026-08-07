#!/usr/bin/env python3
"""Embed the avGFP DMS sequences with ESM-2 650M, matching the dataset_pipeline format.

Row-aligned to avgfp_dms_sequences.csv (same order as that file), this writes into
DMS_data/ a per-residue ESM-2 embedding cache identical in layout to the curated
trait caches produced by dataset_pipeline/embed.py:

    esm_residue_fp16.npy   (N, Lmax, 1280) float16  -- H[i, :len_i] are residue embeddings
    esm_residue_len.npy    (N,) int64               -- sequence lengths

ESM-2 (esm2_t33_650M_UR50D), layer 33, computed in fp32 then stored as fp16, exactly like
the peak/brightness/pka caches. The (51715, 235, 1280) fp16 array is ~31 GB, so it is
written incrementally through a numpy memmap (never fully held in RAM). A sidecar
.embed_progress file records how many rows are done so an interrupted run can --resume.

Stop-codon variants have been removed upstream by transform_dms.py, so every sequence here
is the full-length (235 aa) parent with substitutions.

Usage:
    python embed_dms.py                 # embed (resumes automatically if a partial run exists)
    python embed_dms.py --force         # rebuild from scratch
    python embed_dms.py --dry-run       # report N/Lmax/size without loading ESM-2
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))  # fpdesign.peak_models (ESM-2 utils)

CSV = os.path.join(HERE, "DMS_data", "avgfp_dms_sequences.csv")
EMB = os.path.join(HERE, "DMS_data", "esm_residue_fp16.npy")
LEN = os.path.join(HERE, "DMS_data", "esm_residue_len.npy")
PROG = os.path.join(HERE, "DMS_data", ".embed_progress")
CHUNK = 256          # rows per outer step (resid_embed sub-batches internally at bs)
BS = 16              # ESM forward batch size


def load_seqs():
    rows = list(csv.DictReader(open(CSV)))
    return [r["mutatedSequence"] for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild from scratch, ignore any partial run")
    ap.add_argument("--dry-run", action="store_true", help="report shapes without loading ESM-2")
    a = ap.parse_args()

    seqs = load_seqs()
    N = len(seqs)
    Ls = np.array([len(s) for s in seqs], dtype=np.int64)
    Lmax = int(Ls.max())
    D_IN = 1280
    gb = N * Lmax * D_IN * 2 / 1e9
    print(f"N={N}  Lmax={Lmax}  D_IN={D_IN}  -> esm_residue_fp16.npy ~{gb:.1f} GB")
    print(f"empty(len0) rows: {int((Ls == 0).sum())}")

    if a.dry_run:
        print("dry-run: not loading ESM-2")
        return

    start = 0
    if a.force:
        for p in (EMB, LEN, PROG):
            if os.path.exists(p):
                os.remove(p)

    if os.path.exists(EMB) and os.path.exists(PROG) and not a.force:
        H = np.lib.format.open_memmap(EMB, mode="r+")
        if H.shape != (N, Lmax, D_IN):
            print(f"ERROR: existing {EMB} has shape {H.shape}, expected {(N, Lmax, D_IN)}. Use --force.",
                  file=sys.stderr)
            return
        start = int(open(PROG).read().strip() or 0)
        print(f"resuming from row {start}/{N}")
    else:
        H = np.lib.format.open_memmap(EMB, mode="w+", dtype=np.float16, shape=(N, Lmax, D_IN))

    import torch
    from fpdesign import peak_models as pm
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {dev}")

    for i0 in range(start, N, CHUNK):
        chunk = seqs[i0:i0 + CHUNK]
        # resid_embed handles a mix of lengths (incl. empty strings) within a chunk; it
        # pads to the chunk's own Lmax, so slice each row by its true length on write-out.
        Hm, _ = pm.resid_embed(chunk, dev, bs=BS)
        Hm = Hm.cpu().numpy()
        for k, s in enumerate(chunk):
            n = len(s)
            if n:
                H[i0 + k, :n] = Hm[k, :n].astype(np.float16)
        H.flush()
        done = min(i0 + CHUNK, N)
        with open(PROG, "w") as f:
            f.write(str(done))
        print(f"  {done}/{N}", end="\r", flush=True)

    np.save(LEN, Ls)
    if os.path.exists(PROG):
        os.remove(PROG)
    print(f"\nwrote {EMB}\nwrote {LEN}")


if __name__ == "__main__":
    main()
