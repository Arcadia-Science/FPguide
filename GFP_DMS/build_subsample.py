#!/usr/bin/env python3
"""Build a stratified 20k subsample cache spanning all four GFP scaffolds.

The bright/dim classification sweep (`sweep_classify_parallel.py`) trains on a compact,
balanced slice of the full ortholog landscape: **5,000 rows from each of the 4 scaffolds**
(avGFP + amacGFP + cgreGFP + ppluGFP) = 20,000 rows. Those scaffolds live in TWO separate,
row-aligned ESM-2 caches with different Lmax and different target-column names:

    avGFP                          -> avgfp_dms_sequences.csv (logMedianBrightness, brightnessClass)
                                      esm_residue_fp16.npy               (N, Lav, 1280) fp16
    amacGFP / cgreGFP / ppluGFP    -> ortho_gfp_dms_sequences.csv (logBrightness, brightnessClass)
                                      ortho_gfp_dms_esm_residue_fp16.npy (N, Lor, 1280) fp16

This script samples 5k rows/scaffold (seeded), copies their per-residue embeddings out of the
two big memmaps into one compact contiguous cache padded to a common Lmax, and writes a
row-aligned CSV with an integer `label` (bright=1, dim=0) plus a per-scaffold stratified
70/15/15 `split`. Everything downstream only touches these small files.

Outputs (into DMS_data/):
    sub20k_esm_residue_fp16.npy   (20000, Lmax, 1280) fp16
    sub20k_esm_residue_len.npy    (20000,) int64
    sub20k_sequences.csv          scaffold, mutatedSequence, brightnessClass, label, split, src, src_row

Usage:
    python build_subsample.py                 # 5k/scaffold, seed 0
    python build_subsample.py --per 5000 --seed 0
    python build_subsample.py --dry-run       # report picks/shapes, write nothing
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DMS = os.path.join(HERE, "DMS_data")

# (scaffold, sequences.csv, embedding.npy, length.npy) -- avGFP is its own cache; the other three
# share the ortho cache. brightnessClass is the shared label column across both CSVs.
SOURCES = [
    ("avGFP",   "avgfp_dms_sequences.csv",      "esm_residue_fp16.npy",            "esm_residue_len.npy"),
    ("amacGFP", "ortho_gfp_dms_sequences.csv",  "ortho_gfp_dms_esm_residue_fp16.npy", "ortho_gfp_dms_esm_residue_len.npy"),
    ("cgreGFP", "ortho_gfp_dms_sequences.csv",  "ortho_gfp_dms_esm_residue_fp16.npy", "ortho_gfp_dms_esm_residue_len.npy"),
    ("ppluGFP", "ortho_gfp_dms_sequences.csv",  "ortho_gfp_dms_esm_residue_fp16.npy", "ortho_gfp_dms_esm_residue_len.npy"),
]
CLASS_COL = "brightnessClass"
SEQ_COL = "mutatedSequence"
LABEL_MAP = {"bright": 1, "dim": 0}
D_IN = 1280


def _read_rows(csv_path):
    with open(csv_path) as fh:
        return list(csv.DictReader(fh))


def _stratified_split(n, rng):
    """Return an (n,) array of 'train'/'val'/'test' with a 70/15/15 shuffled split."""
    perm = rng.permutation(n)
    n_te = int(round(0.15 * n)); n_va = int(round(0.15 * n))
    split = np.empty(n, dtype=object)
    split[perm[:n_te]] = "test"
    split[perm[n_te:n_te + n_va]] = "val"
    split[perm[n_te + n_va:]] = "train"
    return split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=5000, help="rows sampled per scaffold (default 5000)")
    ap.add_argument("--seed", type=int, default=0, help="rng seed for sampling + splits (default 0)")
    ap.add_argument("--stem", default="sub20k", help="output cache stem in DMS_data/ (default sub20k)")
    ap.add_argument("--dry-run", action="store_true", help="report picks/shapes, write nothing")
    a = ap.parse_args()

    out_emb = os.path.join(DMS, f"{a.stem}_esm_residue_fp16.npy")
    out_len = os.path.join(DMS, f"{a.stem}_esm_residue_len.npy")
    out_csv = os.path.join(DMS, f"{a.stem}_sequences.csv")

    rng = np.random.default_rng(a.seed)

    # cache CSV reads (the ortho CSV is shared by three scaffolds)
    csv_cache, len_cache = {}, {}
    picks = []            # list of dicts describing every chosen row (in output order)

    for scaf, csv_name, emb_name, len_name in SOURCES:
        csv_path = os.path.join(DMS, csv_name)
        if csv_name not in csv_cache:
            csv_cache[csv_name] = _read_rows(csv_path)
        rows = csv_cache[csv_name]
        idx = np.array([i for i, r in enumerate(rows) if r["scaffold"] == scaf], dtype=np.int64)
        if len(idx) < a.per:
            raise SystemExit(f"scaffold {scaf}: only {len(idx)} rows < requested {a.per}")
        chosen = np.sort(rng.choice(idx, size=a.per, replace=False))     # sorted -> monotonic memmap reads
        split = _stratified_split(len(chosen), rng)

        emb_path = os.path.join(DMS, emb_name)
        if not os.path.exists(emb_path):
            raise SystemExit(f"missing embedding cache for {scaf}: {emb_path}\n"
                             f"(the ortho embed job must finish first)")
        if len_name not in len_cache:
            len_cache[len_name] = np.load(os.path.join(DMS, len_name)).astype(np.int64)
        Ls = len_cache[len_name]

        n_bright = sum(1 for i in chosen if rows[i][CLASS_COL] == "bright")
        print(f"{scaf:9} from {emb_name:34} picked {len(chosen)}  "
              f"bright={n_bright} ({100*n_bright/len(chosen):.1f}%)")
        for k, i in enumerate(chosen):
            r = rows[i]
            picks.append(dict(scaffold=scaf, seq=r[SEQ_COL], cls=r[CLASS_COL],
                              label=LABEL_MAP[r[CLASS_COL]], split=split[k],
                              emb=emb_name, src_row=int(i), seq_len=int(Ls[i])))

    N = len(picks)
    Lmax = max(p["seq_len"] for p in picks)
    print(f"\ntotal {N} rows | Lmax {Lmax} | output ~{N * Lmax * D_IN * 2 / 1e9:.1f} GB fp16")
    if a.dry_run:
        print("dry-run: nothing written"); return

    # write the compact embedding cache, copying each source cache's chosen rows once
    out = np.lib.format.open_memmap(out_emb, mode="w+", dtype=np.float16, shape=(N, Lmax, D_IN))
    lens = np.zeros(N, dtype=np.int64)
    by_emb = {}
    for dst, p in enumerate(picks):
        by_emb.setdefault(p["emb"], []).append((dst, p["src_row"], p["seq_len"]))
    for emb_name, items in by_emb.items():
        H = np.load(os.path.join(DMS, emb_name), mmap_mode="r")     # (N, Lsrc, 1280) fp16
        Lsrc = H.shape[1]
        print(f"copying {len(items)} rows from {emb_name} (Lsrc={Lsrc}) ...", flush=True)
        for dst, src, sl in items:
            out[dst, :Lsrc] = H[src]                                # pad remainder stays zero
            lens[dst] = sl
        del H
    out.flush(); del out
    np.save(out_len, lens)

    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "scaffold", SEQ_COL, CLASS_COL, "label", "split", "src", "src_row"])
        for i, p in enumerate(picks):
            w.writerow([i, p["scaffold"], p["seq"], p["cls"], p["label"], p["split"], p["emb"], p["src_row"]])

    n_pos = sum(p["label"] for p in picks)
    print(f"\nwrote {out_emb}\nwrote {out_len}\nwrote {out_csv}")
    print(f"labels: bright={n_pos} ({100*n_pos/N:.1f}%)  dim={N - n_pos} ({100*(N-n_pos)/N:.1f}%)")


if __name__ == "__main__":
    main()
