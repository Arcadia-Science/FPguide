#!/usr/bin/env python3
"""Build the in-distribution reference cloud: sub40k's ESM-2 max-pool vectors.

The design campaigns call a sequence in-distribution when its nearest-neighbour L2 distance to this
cloud, in z-scored max-pool space, is inside the cloud's own 99th-percentile self-excluded NN
distance (`design-campaign-EGFP/make_shortlist_case.py`). The cloud is sub40k -- the same 40,000
variants the deployed bright/dim classifier was fitted and selected on -- so "inside the reference
cloud" and "inside the training distribution" are the same statement, and row i of the cloud is
row i of `sub40k_sequences.csv`.

That row alignment is what `nn_distance_accuracy.py` and `visualization.ipynb`'s held-out pass both
rely on to pair a variant's prediction with its own embedding, so it is asserted here rather than
assumed.

    HISTORY. This replaces `esm_maxpool_4scaffold_10k.npz`, an untracked cache built by an ad-hoc
    script that replayed `build_subsample.py`'s sampling -- np.random.default_rng(0), then
    np.sort(rng.choice(idx, 10000, replace=False)) per scaffold -- but omitted the per-scaffold
    `_stratified_split` call. That call consumes the RNG, so the stream diverged after the first
    scaffold: its avGFP block is bit-identical to sub40k's, its three ortho blocks are an
    independent draw sharing only 10,372 of 30,000 rows. It was a valid 10k/scaffold sample of the
    same corpus, so distance-to-cloud was sound, but it was NOT an index of sub40k, and the two
    consumers above were silently reading the wrong row for 3/4 of the data.

Pooling streams the (40000, 237, 1280) fp16 residue cache -- 24 GB, more than this machine's RAM --
in blocks off the memmap and keeps only the (40000, 1280) result, the same approach
`visualization.ipynb` uses for sub20k.

Usage:
    python build_maxpool_cache.py            # skip if a valid cache already exists
    python build_maxpool_cache.py --force    # rebuild
    python build_maxpool_cache.py --verify   # check an existing cache, write nothing
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DMS = os.path.join(HERE, "DMS_data")
OUT = os.path.join(DMS, "sub40k_maxpool.npz")
# (src embedding filename, sequences CSV, continuous-brightness column) for the two source tables
SOURCES = {
    "esm_residue_fp16.npy": ("avgfp_dms_sequences.csv", "logMedianBrightness"),
    "ortho_gfp_dms_esm_residue_fp16.npy": ("ortho_gfp_dms_sequences.csv", "logBrightness"),
}


def brightness(meta):
    """Continuous log brightness per sub40k row, joined from the source table by (src, src_row)."""
    y = np.full(len(meta), np.nan)
    for src, (csv_name, col) in SOURCES.items():
        m = meta["src"].to_numpy() == src
        if not m.any():
            continue
        table = pd.read_csv(os.path.join(DMS, csv_name))
        rows = meta.loc[m, "src_row"].to_numpy()
        y[m] = table[col].to_numpy()[rows]
        # the join is only meaningful if it lands on the same sequence it claims to
        assert (table["mutatedSequence"].to_numpy()[rows]
                == meta.loc[m, "mutatedSequence"].to_numpy()).all(), f"{csv_name} join is misaligned"
    assert np.isfinite(y).all(), "some rows got no brightness"
    return y


def pool(meta, stem="sub40k", block=200):
    """Max-pool each variant's residue embedding over its real length, straight off the memmap."""
    H = np.load(os.path.join(DMS, f"{stem}_esm_residue_fp16.npy"), mmap_mode="r")
    Ls = np.load(os.path.join(DMS, f"{stem}_esm_residue_len.npy")).astype(int)
    assert len(H) == len(meta), f"{stem} embedding cache is not row-aligned with its CSV"
    out = np.empty((len(H), H.shape[2]), np.float32)
    print(f"pooling {H.shape} ({H.nbytes / 1e9:.0f} GB) in blocks of {block}...", flush=True)
    for i in range(0, len(H), block):
        blk = np.asarray(H[i:i + block]).astype(np.float32)
        for j in range(len(blk)):
            out[i + j] = blk[j, :Ls[i + j]].max(0)
        if i % 8000 == 0:
            print(f"  {i:>6,}/{len(H):,}", flush=True)
    return out


def verify(path, meta):
    """Every guarantee downstream code relies on, checked against the CSV and the retired cache."""
    z = np.load(path, allow_pickle=True)
    ok = True
    for key, want in (("scaf", meta["scaffold"].to_numpy()),
                      ("src", meta["src"].to_numpy()),
                      ("src_row", meta["src_row"].to_numpy())):
        good = key in z.files and len(z[key]) == len(meta) and (z[key] == want).all()
        print(f"  {key:8} row-aligned with sub40k_sequences.csv: {good}")
        ok &= good
    print(f"  mp shape {z['mp'].shape}, dtype {z['mp'].dtype}")

    # The retired cache's avGFP block came from the same draw, so it must match bit-for-bit.
    old = os.path.join(DMS, "esm_maxpool_4scaffold_10k.npz")
    if os.path.exists(old):
        same = np.array_equal(np.load(old, allow_pickle=True)["mp"][:10000], z["mp"][:10000])
        print(f"  avGFP block (rows 0-9999) bit-identical to the retired cache: {same}")
        ok &= same
    else:
        print("  retired cache absent -- skipping the avGFP cross-check")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    ap.add_argument("--verify", action="store_true", help="check an existing cache, write nothing")
    a = ap.parse_args()

    meta = pd.read_csv(os.path.join(DMS, "sub40k_sequences.csv"))
    print(f"sub40k: {len(meta)} rows | " +
          " ".join(f"{k} {v}" for k, v in meta["scaffold"].value_counts().items()))

    if a.verify:
        if not os.path.exists(OUT):
            raise SystemExit(f"nothing to verify: {OUT} does not exist")
        print(f"verifying {OUT}")
        raise SystemExit(0 if verify(OUT, meta) else "VERIFY FAILED")

    if os.path.exists(OUT) and not a.force:
        print(f"{OUT} exists -- verifying instead (use --force to rebuild)")
        raise SystemExit(0 if verify(OUT, meta) else "VERIFY FAILED")

    mp = pool(meta)
    np.savez(OUT, mp=mp, scaf=meta["scaffold"].to_numpy().astype("U7"), y=brightness(meta),
             src=meta["src"].to_numpy(), src_row=meta["src_row"].to_numpy())
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.0f} MB)")
    if not verify(OUT, meta):
        raise SystemExit("VERIFY FAILED on the cache just written")


if __name__ == "__main__":
    main()
