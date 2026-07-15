#!/usr/bin/env python
"""ESMFold structure prediction for FP scaffolds (Apple Silicon / MPS, CPU fallback).

Alternative to fetching experimental RCSB structures for defining the
structure-based edit window in ``guided_design_peak_structure_multiscaffold.ipynb``.
Folds the *exact dataset sequence* of any scaffold with ESMFold (``esmfold_v1``),
so the predicted-structure residue numbering matches the dataset sequence 1:1 and
the RCSB fetch + homolog-alignment step is removed (relevant for eqFP578, which has
no exact PDB and currently borrows TagRFP 3M22 at 90 %% id).

Licensing (verified commercial-OK):
  * ESMFold model + code  -> facebookresearch/esm, MIT (weights trained from
    scratch by Meta, not derived from AlphaFold; no DeepMind encumbrance).
  * OpenFold (code dependency only) -> aqlaboratory/openfold, Apache-2.0. We do
    NOT use OpenFold's own trained weights (CC BY 4.0); only its Python modules.

Install notes (see project setup): OpenFold's pinned commit 4b41059 was installed
with its CUDA extension stripped, plus a stub ``attn_core_inplace_cuda`` module, so
the pure-Python forward path imports on macOS. The CUDA kernel is only used by
OpenFold's memory-efficient attention, which ESMFold's default forward never takes.

Usage
-----
    # fold the three scaffolds referenced by the notebook and save PDBs + metrics
    python esmfold_fold.py                       # -> structures/esmfold/{name}.pdb
    python esmfold_fold.py --device cpu          # force CPU
    python esmfold_fold.py --indices 138 52 179  # explicit dataset indices

As a module::

    import esmfold_fold as ef
    model, dev = ef.load_model()                 # esmfold_v1 on MPS (CPU fallback)
    pdb_str, info = ef.fold_sequence(seq, model, dev)
"""
import os
# MPS must be allowed to fall back to CPU for any op ESMFold uses that lacks an
# MPS kernel (e.g. some structure-module frame math). Set BEFORE torch is used.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import json
import time

import numpy as np
import torch

CUR = os.path.join("..", "dataset_pipeline", "data", "peak", "curated")
OUTDIR_DEFAULT = os.path.join("structures", "esmfold")
# scaffolds referenced by guided_design_peak_structure_multiscaffold.ipynb
DEFAULT_INDICES = {"DsRed": 138, "avGFP": 52, "eqFP578": 179}


# ----------------------------------------------------------------------------- data
def load_dataset(cur=CUR):
    """Return (rows, seqs, name2idx) from the curated peak dataset."""
    rows = list(csv.DictReader(open(os.path.join(cur, "peaks_assignments.csv"))))
    n = len(rows)
    seqs = [None] * n
    h = None
    for line in open(os.path.join(cur, "sequences.fasta")):
        line = line.strip()
        if line.startswith(">"):
            h = int(line[1:].split("|")[0])
        elif line:
            seqs[h] = line
    name2idx = {r["name"]: i for i, r in enumerate(rows)}
    return rows, seqs, name2idx


# ----------------------------------------------------------------------------- model
def pick_device(prefer="mps"):
    """Resolve a torch.device, honouring `prefer` then falling back sensibly."""
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer not in ("cpu", "mps", "cuda"):
        raise ValueError(f"unknown device preference {prefer!r}")
    if prefer != "cpu":
        print(f"[esmfold] requested {prefer!r} unavailable -> using CPU")
    return torch.device("cpu")


_MODEL = None  # process-level cache (esmfold_v1 is ~2.7 GB to load)


def load_model(device=None, prefer="mps", chunk_size=128, lm_fp32=True):
    """Load `esm.pretrained.esmfold_v1`, eval mode, on `device` (default MPS).

    chunk_size trades speed for peak memory in the folding trunk's attention
    (smaller = less memory). Weights download on first call (~2.7 GB).

    lm_fp32=True undoes ESMFold's default fp16 cast of the ESM-2 language model
    (`esm/esmfold/v1/esmfold.py` does `self.esm.half()`). fp16 is a CUDA
    memory optimisation; on CPU it raises ("LayerNormKernelImpl not implemented
    for 'Half'") and on MPS it runs but at reduced precision. Keeping the LM in
    fp32 makes CPU/MPS robust and deterministic. (It does NOT rescue GFP-family
    barrels, which ESMFold predicts poorly regardless of precision — see the
    quality note in this module's header discussion / README.)
    Returns (model, device).
    """
    global _MODEL
    import esm  # deferred: importing esm pulls the (heavy) openfold chain

    if device is None:
        device = pick_device(prefer)
    if _MODEL is None:
        t0 = time.time()
        print("[esmfold] loading esmfold_v1 weights (first run downloads ~2.7 GB) ...")
        _MODEL = esm.pretrained.esmfold_v1().eval()
        print(f"[esmfold] weights loaded in {time.time() - t0:.0f}s")
    if lm_fp32:
        _MODEL.esm = _MODEL.esm.float()  # undo the fp16 LM cast (CPU-safe, MPS-precise)
    if chunk_size is not None:
        _MODEL.set_chunk_size(chunk_size)  # lower peak memory in the trunk
    _MODEL = _MODEL.to(device)
    return _MODEL, device


@torch.no_grad()
def fold_sequence(seq, model, device, allow_cpu_fallback=True):
    """Fold a single sequence -> (pdb_string, info dict with mean pLDDT / pTM).

    On a hard MPS runtime error, retries once on CPU (if allow_cpu_fallback).
    """
    seq = seq.strip().upper()
    try:
        out = model.infer([seq])
    except (RuntimeError, NotImplementedError) as e:
        if not (allow_cpu_fallback and device.type != "cpu"):
            raise
        print(f"[esmfold] {device.type} failed ({type(e).__name__}: {str(e)[:120]}) -> retrying on CPU")
        model = model.to("cpu")
        device = torch.device("cpu")
        out = model.infer([seq])
    pdb_str = model.output_to_pdb(out)[0]
    info = {
        "len": len(seq),
        "mean_plddt": float(out["mean_plddt"][0].item()),
        "ptm": float(out["ptm"][0].item()),
        "device": device.type,
    }
    return pdb_str, info


# ----------------------------------------------------------------------------- driver
def fold_scaffolds(indices, outdir=OUTDIR_DEFAULT, prefer="mps", chunk_size=128, cur=CUR):
    """Fold each dataset index, save <outdir>/<name>.pdb, return metrics list.

    `indices` maps name -> dataset row index. Resumable: skips a scaffold whose
    PDB already exists (still records its metrics from esmfold_meta.json if present).
    """
    os.makedirs(outdir, exist_ok=True)
    rows, seqs, _ = load_dataset(cur)
    meta_path = os.path.join(outdir, "esmfold_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    model = device = None
    results = []
    for name, idx in indices.items():
        pdb_path = os.path.join(outdir, f"{name}.pdb")
        seq = seqs[idx]
        assert rows[idx]["name"] == name, f"index {idx} is {rows[idx]['name']!r}, expected {name!r}"
        if os.path.exists(pdb_path) and name in meta:
            print(f"[{name}] cached -> {pdb_path} (pLDDT {meta[name]['mean_plddt']:.1f}, pTM {meta[name]['ptm']:.3f})")
            results.append({"name": name, "idx": idx, "pdb": pdb_path, **meta[name]})
            continue
        if model is None:
            model, device = load_model(prefer=prefer, chunk_size=chunk_size)
            print(f"[esmfold] folding on device={device.type} chunk_size={chunk_size}")
        t0 = time.time()
        pdb_str, info = fold_sequence(seq, model, device)
        dt = time.time() - t0
        with open(pdb_path, "w") as fh:
            fh.write(pdb_str)
        info = {**info, "idx": idx, "seconds": round(dt, 1)}
        meta[name] = info
        json.dump(meta, open(meta_path, "w"), indent=2)
        print(f"[{name}] folded L={info['len']} in {dt:.0f}s | mean pLDDT {info['mean_plddt']:.1f} "
              f"| pTM {info['ptm']:.3f} | device {info['device']} -> {pdb_path}")
        results.append({"name": name, "idx": idx, "pdb": pdb_path, **info})
    return results


def main():
    ap = argparse.ArgumentParser(description="Fold FP scaffolds with ESMFold (MPS, CPU fallback).")
    ap.add_argument("--indices", type=int, nargs="*", default=None,
                    help="explicit dataset row indices (default: DsRed 138, avGFP 52, eqFP578 179)")
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    ap.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    ap.add_argument("--chunk-size", type=int, default=128)
    args = ap.parse_args()

    if args.indices:
        rows, _, _ = load_dataset()
        indices = {rows[i]["name"]: i for i in args.indices}
    else:
        indices = DEFAULT_INDICES
    res = fold_scaffolds(indices, outdir=args.outdir, prefer=args.device, chunk_size=args.chunk_size)
    print("\n=== ESMFold summary ===")
    print(f"{'scaffold':10}{'idx':>5}{'len':>5}{'mean_pLDDT':>12}{'pTM':>7}{'device':>8}")
    for r in res:
        print(f"{r['name']:10}{r['idx']:>5}{r['len']:>5}{r['mean_plddt']:>12.1f}{r['ptm']:>7.3f}{r['device']:>8}")


if __name__ == "__main__":
    main()
