#!/usr/bin/env python3
"""Embed GFP DMS sequences with ESM-2 650M in parallel across multiple GPUs.

This is the multi-GPU counterpart of embed_dms.py. It produces the identical cache layout
(row-aligned to the input CSV, same order as that file):

    <stem>_esm_residue_fp16.npy   (N, Lmax, 1280) float16  -- H[i, :len_i] residue embeddings
    <stem>_esm_residue_len.npy    (N,) int64               -- sequence lengths

How the parallelism works
-------------------------
The N rows are split into K contiguous, disjoint shards (K = number of GPUs). The launcher
process pre-creates the single shared (N, Lmax, 1280) fp16 memmap once, then spawns one
worker per GPU (each pinned via CUDA_VISIBLE_DEVICES so it sees exactly one device as cuda:0).
Every worker opens the SAME .npy in r+ mode and writes only the rows in its shard -- disjoint
byte ranges of one file, which is safe for concurrent writers on Linux. Each shard keeps its
own .embed_progress.shard<i> sidecar, so an interrupted run resumes each GPU where it stopped.

Usage
-----
    python embed_parallel.py                       # embed ortho set on all visible GPUs
    python embed_parallel.py --gpus 0,1,2,3        # choose GPUs (one shard per GPU)
    python embed_parallel.py --input DMS_data/avgfp_dms_sequences.csv
    python embed_parallel.py --dry-run             # report N/Lmax/size/shards, no ESM-2
    python embed_parallel.py --force               # rebuild from scratch
    python embed_parallel.py --bs 24 --chunk 256   # tune per-GPU forward batch / write chunk

(The --worker flag is used internally by the launcher; you don't run it by hand.)
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))  # fpdesign.peak_models (ESM-2 utils)

D_IN = 1280
DEFAULT_INPUT = os.path.join(HERE, "DMS_data", "ortho_gfp_dms_sequences.csv")
LOGDIR = os.path.join(HERE, "logs")


def derive_paths(inp: str):
    """CSV path -> (emb.npy, len.npy) cache paths sharing the input's stem."""
    d, b = os.path.dirname(inp), os.path.basename(inp)
    stem = b[:-len("_sequences.csv")] if b.endswith("_sequences.csv") else os.path.splitext(b)[0]
    return (os.path.join(d, f"{stem}_esm_residue_fp16.npy"),
            os.path.join(d, f"{stem}_esm_residue_len.npy"))


def load_seqs(inp: str, col: str):
    rows = list(csv.DictReader(open(inp)))
    if col not in rows[0]:
        sys.exit(f"ERROR: column {col!r} not in {inp} (have: {list(rows[0])})")
    return [r[col] for r in rows]


def shard_bounds(n: int, k: int):
    """Contiguous split of range(n) into k parts -> list of (start, end)."""
    edges = np.linspace(0, n, k + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(k)]


# --------------------------------------------------------------------------- worker
def run_worker(a):
    emb, _ = derive_paths(a.input) if not a.emb else (a.emb, None)
    seqs = load_seqs(a.input, a.seq_col)
    N = len(seqs)
    prog = f"{emb}.progress.shard{a.shard_id}"

    bounds = shard_bounds(N, a.shards)
    s0, s1 = bounds[a.shard_id]

    start = s0
    if os.path.exists(prog):
        start = max(s0, int(open(prog).read().strip() or s0))
    if start >= s1:
        print(f"[shard {a.shard_id}] already complete ({s0}..{s1})", flush=True)
        return

    H = np.lib.format.open_memmap(emb, mode="r+")  # shared file, header written by launcher

    import torch
    from fpdesign import peak_models as pm
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[shard {a.shard_id}] rows {start}..{s1} of {N} on {os.environ.get('CUDA_VISIBLE_DEVICES','?')} ({dev})",
          flush=True)

    t0 = time.time()
    for i0 in range(start, s1, a.chunk):
        chunk = seqs[i0:min(i0 + a.chunk, s1)]
        Hm, _ = pm.resid_embed(chunk, dev, bs=a.bs)
        Hm = Hm.cpu().numpy()
        for k, s in enumerate(chunk):
            n = len(s)
            if n:
                H[i0 + k, :n] = Hm[k, :n].astype(np.float16)
        H.flush()
        done = min(i0 + a.chunk, s1)
        with open(prog, "w") as f:
            f.write(str(done))
        rate = (done - start) / max(time.time() - t0, 1e-6)
        print(f"[shard {a.shard_id}] {done - s0}/{s1 - s0}  ({rate:.1f} seq/s)", flush=True)
    print(f"[shard {a.shard_id}] done in {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- launcher
def check_gpus(gpus):
    """Fail before spawning workers if a requested GPU id is not actually there.

    Each worker is pinned with CUDA_VISIBLE_DEVICES=<id>; an id past the end of the visible
    devices leaves that worker with no device, and it silently falls back to CPU -- at which
    point the run does not finish rather than failing. Check the ids up front instead.
    """
    import torch
    n = torch.cuda.device_count()
    bad = [g for g in gpus if not g.isdigit() or int(g) >= n]
    if bad:
        raise SystemExit(
            f"{__file__}: --gpus asks for {','.join(gpus)}, but this host has {n} visible CUDA "
            f"device(s){' (none)' if n == 0 else f' (ids 0..{n - 1})'}.\n\n"
            f"Unknown id(s): {','.join(bad)}. A worker pinned to a missing device sees no GPU and\n"
            "falls back to CPU, which will not finish. Pass --gpus with the ids you actually have\n"
            "(e.g. --gpus 0 on a single-GPU host)"
            + ("." if n else ", or run on a GPU host.")
        )


def run_launcher(a):
    emb, lenp = derive_paths(a.input)
    if a.emb:
        emb = a.emb
    seqs = load_seqs(a.input, a.seq_col)
    N = len(seqs)
    Ls = np.array([len(s) for s in seqs], dtype=np.int64)
    Lmax = int(Ls.max())
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip() != ""]
    check_gpus(gpus)
    K = len(gpus)
    gb = N * Lmax * D_IN * 2 / 1e9

    print(f"input : {a.input}")
    print(f"output: {emb}")
    print(f"N={N}  Lmax={Lmax}  D_IN={D_IN}  -> ~{gb:.1f} GB fp16")
    print(f"GPUs  : {gpus}  ({K} shards)")
    for i, (s0, s1) in enumerate(shard_bounds(N, K)):
        print(f"  shard {i} (gpu {gpus[i]}): rows {s0}..{s1}  ({s1 - s0} seqs)")
    if a.dry_run:
        print("dry-run: not launching workers")
        return

    shard_progs = [f"{emb}.progress.shard{i}" for i in range(K)]
    if a.force:
        for p in (emb, lenp, *shard_progs):
            if os.path.exists(p):
                os.remove(p)

    if os.path.exists(emb) and not a.force:
        H = np.lib.format.open_memmap(emb, mode="r")
        if H.shape != (N, Lmax, D_IN):
            sys.exit(f"ERROR: existing {emb} has shape {H.shape}, expected {(N, Lmax, D_IN)}. Use --force.")
        del H
        print("resuming into existing cache")
    else:
        H = np.lib.format.open_memmap(emb, mode="w+", dtype=np.float16, shape=(N, Lmax, D_IN))
        del H  # header written; workers reopen r+
        print("created fresh cache")
    np.save(lenp, Ls)

    os.makedirs(LOGDIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    procs = []
    for i, g in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        log = open(os.path.join(LOGDIR, f"embed_shard{i}_gpu{g}_{stamp}.log"), "w")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               "--shard-id", str(i), "--shards", str(K),
               "--input", a.input, "--emb", emb, "--seq-col", a.seq_col,
               "--bs", str(a.bs), "--chunk", str(a.chunk)]
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, log))
        print(f"launched shard {i} on gpu {g} (pid {p.pid}) -> {log.name}")

    # live aggregate progress from the shard sidecars
    bounds = shard_bounds(N, K)
    t0 = time.time()
    try:
        while any(p.poll() is None for p, _ in procs):
            time.sleep(15)
            done = 0
            for i in range(K):
                s0, s1 = bounds[i]
                if os.path.exists(shard_progs[i]):
                    done += min(int(open(shard_progs[i]).read().strip() or s0), s1) - s0
                # else 0 done for this shard yet
            el = time.time() - t0
            rate = done / max(el, 1e-6)
            eta = (N - done) / rate if rate > 0 else float("inf")
            print(f"  aggregate {done}/{N} ({100*done/N:.1f}%)  {rate:.1f} seq/s  "
                  f"elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)
    except KeyboardInterrupt:
        print("interrupted; terminating workers (progress is saved, rerun to resume)")
        for p, _ in procs:
            p.terminate()

    codes = [p.wait() for p, _ in procs]
    for _, log in procs:
        log.close()
    ok = all(c == 0 for c in codes)
    print(f"worker exit codes: {codes}")
    if ok:
        for p in shard_progs:
            if os.path.exists(p):
                os.remove(p)
        print(f"\nwrote {emb}\nwrote {lenp}")
    else:
        sys.exit("one or more shards failed; see logs/. progress saved, rerun to resume.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT, help="input sequences CSV")
    ap.add_argument("--seq-col", default="mutatedSequence", help="sequence column name")
    ap.add_argument("--emb", default="", help="override embedding .npy path (default: derived from --input)")
    ap.add_argument("--gpus", default="0",
                    help="comma-separated GPU ids, one shard each (default: single GPU; pass "
                         "e.g. 0,1,2,3 on a 4-GPU host)")
    ap.add_argument("--bs", type=int, default=16, help="ESM forward batch size (per GPU)")
    ap.add_argument("--chunk", type=int, default=256, help="rows per outer write step (per GPU)")
    ap.add_argument("--force", action="store_true", help="rebuild from scratch")
    ap.add_argument("--dry-run", action="store_true", help="report shapes/shards without launching")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--shard-id", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--shards", type=int, default=1, help=argparse.SUPPRESS)
    a = ap.parse_args()
    (run_worker if a.worker else run_launcher)(a)


if __name__ == "__main__":
    main()
