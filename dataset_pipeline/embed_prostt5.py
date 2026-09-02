#!/usr/bin/env python3
"""Embed the curated sequences with the ProstT5 encoder and cache them per trait.

The ESM-2 counterpart of this script is `embed.py`; this is the structure-aware cache the
*oracle* consumes. For each trait, writes into its own data/<trait>/curated/ folder a per-residue
ProstT5 embedding cache, row-aligned to that trait's curated data (same index order as
<target>.npy / the FASTA / the assignments CSV):

    prostt5_residue_fp16.npy   (N, Lmax, 1024) float16   -- H[i, :len_i] are the residue embeddings
    prostt5_residue_len.npy    (N,) int64                -- sequence lengths

Row alignment is the whole contract: `in-silico-test/lib/sweep_peak_oracle_base.py` pairs row i of
this cache with row i of `peaks.npy` and with row i of the split CSV. It is enforced the same way
`embed.py` does it -- sort the assignments CSV by `index` before embedding -- and `--verify`
re-checks an existing cache against the CSV without loading ProstT5, so a cache obtained any other
way can be validated before it is trusted.

WHO READS THIS. Only the peak dataset's cache is consumed today: it is the oracle input for
`in-silico-test/` (reached through the per-file symlinks in `in-silico-test/data/`), whose oracle
architecture sweep is the first step that needs it. `--trait brightness` / `--trait pka` work
identically but those curated sets are archived; see dataset_pipeline/README.md.

WHY A SEPARATE SCRIPT. ProstT5 is 1024-dim against ESM-2's 1280 and needs a different tokenizer,
prefix and special-token trim, so the two caches cannot share one loop. The embedding itself comes
from `fpdesign/prostt5_embed.py`, of which `in-silico-test/lib/prostt5_embed.py` is a verbatim
vendored copy -- so this script and the in-silico-test design/scoring path run the same code on the
same weights.

NOT BIT-REPRODUCIBLE, AND THAT IS EXPECTED. The encoder runs in fp16, so accumulation order -- and
therefore the last bit or two of every element -- depends on the batch/padding shape and on the
torch / transformers / GPU build. A single call is deterministic run-to-run, but a rebuild does NOT
byte-match the cache already on disk: measured against it, per-element |diff| is ~2e-4 median and
<=2e-2 worst case against a median element magnitude of ~0.15, i.e. fp16 epsilon, for a per-row
cosine similarity of 0.99999. So rebuilding is numerically equivalent, not identical -- and
`--force` is deliberately opt-in, because every oracle number in `in-silico-test/` was computed
against the cache that is already there. Use `--verify` (structure) and `--spot-check` (numerics)
to validate a cache instead of rebuilding one.

ProstT5 (Rostlab/ProstT5) weights download on first use (~2.5 GB); device is auto-detected
(CUDA -> MPS -> CPU). On Apple Silicon set PYTORCH_ENABLE_MPS_FALLBACK=1 for any unsupported op.
The peak cache (N=758, Lmax=582) is ~0.9 GB on disk and takes ~3 min on an L4.

Usage:
    python embed_prostt5.py                  # peak only (skips a cache that already exists)
    python embed_prostt5.py --trait pka      # one trait
    python embed_prostt5.py --all            # every trait that has a curated set
    python embed_prostt5.py --force          # rebuild even if the cache exists
    python embed_prostt5.py --verify         # check an existing cache, load nothing, write nothing
    python embed_prostt5.py --spot-check 5   # re-embed 5 rows, report agreement with the cache
    python embed_prostt5.py --dry-run        # report N/Lmax/size, without loading ProstT5
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# fpdesign.prostt5_embed holds the shared ProstT5 embedding utilities
sys.path.insert(0, os.path.join(HERE, ".."))

TRAITS = {"peak": "peaks_assignments.csv",
          "brightness": "brightness_assignments.csv",
          "pka": "pka_assignments.csv"}
D_PROSTT5 = 1024   # must match fpdesign.prostt5_embed.D_IN_PROSTT5
CHUNK = 32         # sequences per accumulation step (resid_embed_prostt5 sub-batches internally)


def load_seqs(trait):
    cur = os.path.join(HERE, "data", trait, "curated")
    csv_path = os.path.join(cur, TRAITS[trait])
    if not os.path.exists(csv_path):
        raise SystemExit(f"missing curated set for {trait}: {os.path.relpath(csv_path, HERE)}\n"
                         f"(build it first:  python build_dataset.py --target {trait})")
    rows = list(csv.DictReader(open(csv_path)))
    rows.sort(key=lambda r: int(r["index"]))          # enforce row alignment to <target>.npy
    return cur, [r["seq"] for r in rows]


def paths(cur):
    return (os.path.join(cur, "prostt5_residue_fp16.npy"),
            os.path.join(cur, "prostt5_residue_len.npy"))


def verify_trait(trait):
    """Check an existing cache's shape and per-row lengths against the curated CSV. No model load."""
    cur, seqs = load_seqs(trait)
    emb, ln = paths(cur)
    if not (os.path.exists(emb) and os.path.exists(ln)):
        print(f"[{trait:10}] no cache to verify -> {os.path.relpath(emb, HERE)}")
        return False
    Ls_want = np.array([len(s) for s in seqs], dtype=np.int64)
    H = np.load(emb, mmap_mode="r")
    Ls = np.load(ln)
    ok = True
    for what, got, want in (("rows", H.shape[0], len(seqs)),
                            ("Lmax", H.shape[1], int(Ls_want.max())),
                            ("dim", H.shape[2], D_PROSTT5),
                            ("len rows", Ls.shape[0], len(seqs))):
        if got != want:
            print(f"[{trait:10}] MISMATCH {what}: cache {got} != curated {want}"); ok = False
    if H.dtype != np.float16:
        print(f"[{trait:10}] MISMATCH dtype: cache {H.dtype} != float16"); ok = False
    if ok and not np.array_equal(Ls, Ls_want):
        bad = int((Ls != Ls_want).sum())
        print(f"[{trait:10}] MISMATCH lengths: {bad}/{len(seqs)} rows disagree with the CSV"); ok = False
    print(f"[{trait:10}] {'cache OK' if ok else 'cache INVALID'}  "
          f"{H.shape} {H.dtype} -> {os.path.relpath(emb, HERE)}")
    return ok


# a rebuild agrees to fp16 epsilon, not to the bit; see NOT BIT-REPRODUCIBLE above
SPOT_COS_MIN = 0.9999


def spot_check_trait(trait, dev, n_rows):
    """Re-embed a few evenly spaced rows and report how well they agree with the cached ones."""
    cur, seqs = load_seqs(trait)
    emb, _ = paths(cur)
    if not os.path.exists(emb):
        print(f"[{trait:10}] no cache to spot-check -> {os.path.relpath(emb, HERE)}")
        return False
    from fpdesign.prostt5_embed import resid_embed_prostt5
    H = np.load(emb, mmap_mode="r")
    idx = np.unique(np.linspace(0, len(seqs) - 1, min(n_rows, len(seqs))).round().astype(int))
    ok = True
    for i in idx:
        n = len(seqs[i])
        got = resid_embed_prostt5([seqs[i]], dev)[0][0][:n].cpu().numpy().astype(np.float16)
        want = np.asarray(H[i, :n])
        a, b = got.astype(np.float32).ravel(), want.astype(np.float32).ravel()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        dmax = float(np.abs(a - b).max())
        good = cos >= SPOT_COS_MIN
        ok &= good
        print(f"[{trait:10}] row {i:4d} len {n:3d}  cos={cos:.7f}  max|diff|={dmax:.2e}  "
              f"{'ok' if good else f'BELOW {SPOT_COS_MIN}'}")
    print(f"[{trait:10}] spot-check {'PASSED' if ok else 'FAILED'} on {len(idx)} row(s) "
          f"(bit-identity is not expected -- fp16 accumulation order)")
    return ok


def embed_trait(trait, dev, force=False, dry_run=False):
    cur, seqs = load_seqs(trait)
    N = len(seqs)
    Ls = np.array([len(s) for s in seqs], dtype=np.int64)
    Lmax = int(Ls.max())
    emb, ln = paths(cur)
    tag = f"[{trait:10}] N={N:4d}  Lmax={Lmax:4d}  (~{N * Lmax * D_PROSTT5 * 2 / 1e6:.0f} MB)"

    if dry_run:
        print(f"{tag}  dry-run -> {os.path.relpath(emb, HERE)}")
        return
    if os.path.exists(emb) and os.path.exists(ln) and not force:
        print(f"{tag}  cache exists -> skip (--force to rebuild, --verify to check it)")
        return

    from fpdesign.prostt5_embed import resid_embed_prostt5, D_IN_PROSTT5
    assert D_IN_PROSTT5 == D_PROSTT5, f"ProstT5 dim changed: {D_IN_PROSTT5} != {D_PROSTT5}"
    H = np.zeros((N, Lmax, D_PROSTT5), dtype=np.float16)
    for i0 in range(0, N, CHUNK):
        chunk = seqs[i0:i0 + CHUNK]
        Hm, _ = resid_embed_prostt5(chunk, dev)
        for k, s in enumerate(chunk):
            H[i0 + k, :len(s)] = Hm[k, :len(s)].cpu().numpy().astype(np.float16)
        print(f"  {trait}: {min(i0 + CHUNK, N)}/{N}", end="\r", flush=True)
    np.save(emb, H)
    np.save(ln, Ls)
    print(f"\n{tag}  wrote {os.path.relpath(emb, HERE)} + prostt5_residue_len.npy")


def get_device():
    import torch
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trait", choices=list(TRAITS), help="embed a single trait (default peak)")
    ap.add_argument("--all", action="store_true", help="embed every trait that has a curated set")
    ap.add_argument("--force", action="store_true", help="rebuild caches that already exist")
    ap.add_argument("--verify", action="store_true", help="check existing caches, write nothing")
    ap.add_argument("--spot-check", type=int, metavar="N", default=0,
                    help="re-embed N evenly spaced rows and compare them to the cache (needs ProstT5)")
    ap.add_argument("--dry-run", action="store_true", help="report shapes without loading ProstT5")
    a = ap.parse_args()

    if a.all:      # skip traits whose curated set is archived / not built
        traits = [t for t in TRAITS
                  if os.path.exists(os.path.join(HERE, "data", t, "curated", TRAITS[t]))]
        skipped = [t for t in TRAITS if t not in traits]
        if skipped:
            print(f"no curated set, skipping: {', '.join(skipped)}")
        if not traits:
            raise SystemExit("no curated sets found; run build_dataset.py first")
    else:
        traits = [a.trait or "peak"]

    if a.verify:
        raise SystemExit(0 if all(verify_trait(t) for t in traits) else 1)

    if a.spot_check:
        dev = get_device()
        print(f"device: {dev}")
        raise SystemExit(0 if all(spot_check_trait(t, dev, a.spot_check) for t in traits) else 1)

    dev = None if a.dry_run else get_device()
    if not a.dry_run:
        print(f"device: {dev}")
    for t in traits:
        embed_trait(t, dev, force=a.force, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
