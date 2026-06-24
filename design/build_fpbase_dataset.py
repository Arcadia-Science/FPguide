#!/usr/bin/env python3
"""
Build a processed (sequence, excitation+emission spectrum) dataset from an
fpbase-extractor export, for the ESM-2 property-guided design pipeline.

Reads the extractor's `fpbase_proteins.json` (sequences + metadata) and
`fpbase_spectra.json` (full ex/em curves), joins them per (protein, state),
resamples each curve onto a common wavelength grid, peak-normalizes, and writes:

    <outdir>/
      spectra.npy      float32 (N, 2*G)  -- excitation curve then emission curve
      grid_nm.npy      float32 (G,)      -- wavelength grid (nm); indexes BOTH halves
      sequences.fasta                    -- one record per sample (id = slug|state)
      metadata.csv     N rows aligned to spectra.npy / sequences (incl. seq column)
      meta.json        provenance: counts, grid, normalization, ex/em split index

A sample = one (protein, state) that has a sequence AND both an excitation and an
emission curve for that state. Photoswitchable proteins contribute one sample per
state (so a sequence can repeat across states -- see `seq_group` in metadata).

Usage
-----
    python build_fpbase_dataset.py \
        --proteins ../fpbase-extractor/fpbase_output/fpbase_proteins.json \
        --spectra  ../fpbase-extractor/fpbase_output/fpbase_spectra.json \
        --outdir   ../fpbase-extractor/processed_data/ESM-spectrum

Defaults point at the standard fpbase-extractor layout relative to this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import date

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_PROTEINS = os.path.join(HERE, "..", "fpbase-extractor", "fpbase_output", "fpbase_proteins.json")
DEF_SPECTRA = os.path.join(HERE, "..", "fpbase-extractor", "fpbase_output", "fpbase_spectra.json")
DEF_OUTDIR = os.path.join(HERE, "..", "fpbase-extractor", "processed_data", "ESM-spectrum")

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def resample(curve, grid):
    """Linearly resample [[wl, intensity], ...] onto `grid`; 0 outside measured range."""
    arr = np.asarray(curve, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]                 # sort by wavelength
    wl, inten = arr[:, 0], arr[:, 1]
    # collapse any duplicate wavelengths (keep max)
    wl_u, idx = np.unique(wl, return_index=True)
    inten_u = np.maximum.reduceat(inten, np.sort(idx)) if len(idx) != len(wl) else inten[np.argsort(wl)]
    y = np.interp(grid, wl_u, inten_u[: len(wl_u)], left=0.0, right=0.0)
    peak = y.max()
    if peak > 0:
        y = y / peak                                  # peak-normalize to 1.0
    return y.astype(np.float32)


def clean_seq(seq):
    """Uppercase; flag whether sequence is purely standard amino acids."""
    s = (seq or "").strip().upper()
    return s, all(c in STANDARD_AA for c in s) and len(s) > 0


def build(args):
    proteins = json.load(open(args.proteins))
    spectra = json.load(open(args.spectra))
    seq_by_slug = {p["slug"]: p["seq"] for p in proteins if p.get("seq")}
    meta_by_slug = {p["slug"]: p for p in proteins}

    grid = np.arange(args.wl_min, args.wl_max + args.wl_step, args.wl_step, dtype=np.float32)
    G = len(grid)

    rows, spec_rows, seqs = [], [], []
    seq_to_group = {}
    skipped_no_seq = skipped_nonstd = 0

    for entry in spectra:
        slug = entry["slug"]
        # group this protein's spectra by state, then pair ex+em within a state
        by_state = {}
        for s in entry["spectra"]:
            st = s.get("state") or "default"
            by_state.setdefault(st, {})[s["spectrum_type"]] = s
        for state, d in by_state.items():
            if "excitation" not in d or "emission" not in d:
                continue
            if slug not in seq_by_slug:
                skipped_no_seq += 1
                continue
            seq, ok = clean_seq(seq_by_slug[slug])
            if not ok:
                skipped_nonstd += 1
                continue

            ex = resample(d["excitation"]["data"], grid)
            em = resample(d["emission"]["data"], grid)
            ex_max = d["excitation"].get("max") or float(grid[int(np.argmax(ex))])
            em_max = d["emission"].get("max") or float(grid[int(np.argmax(em))])

            p = meta_by_slug.get(slug, {})
            gid = seq_to_group.setdefault(seq, len(seq_to_group))  # identical-seq grouping
            rows.append({
                "index": len(rows),
                "slug": slug,
                "name": p.get("name", ""),
                "state": state,
                "seq_group": gid,
                "parent_organism": p.get("parent_organism") or "",
                "switch_type": p.get("switch_type_label") or "",
                "oligomerization": p.get("oligomerization") or "",
                "is_dark": d["excitation"].get("max") is None,
                "ex_max": round(float(ex_max), 1),
                "em_max": round(float(em_max), 1),
                "ref_year": p.get("ref_year") or "",
                "seq_len": len(seq),
                "aliases": "; ".join(p.get("aliases") or []),
                "seq": seq,
            })
            spec_rows.append(np.concatenate([ex, em]))
            seqs.append((f"{slug}|{state}", seq))

    spectra_arr = np.stack(spec_rows).astype(np.float32)
    os.makedirs(args.outdir, exist_ok=True)

    np.save(os.path.join(args.outdir, "spectra.npy"), spectra_arr)
    np.save(os.path.join(args.outdir, "grid_nm.npy"), grid)

    with open(os.path.join(args.outdir, "sequences.fasta"), "w") as fh:
        for sid, seq in seqs:
            fh.write(f">{sid}\n{seq}\n")

    cols = ["index", "slug", "name", "state", "seq_group", "parent_organism",
            "switch_type", "oligomerization", "is_dark", "ex_max", "em_max",
            "ref_year", "seq_len", "aliases", "seq"]
    with open(os.path.join(args.outdir, "metadata.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    n_unique_seq = len(seq_to_group)
    meta = {
        "created": date.today().isoformat(),
        "source_proteins": os.path.relpath(args.proteins, args.outdir),
        "source_spectra": os.path.relpath(args.spectra, args.outdir),
        "n_samples": len(rows),
        "n_unique_sequences": n_unique_seq,
        "spectrum_dim": int(spectra_arr.shape[1]),
        "grid_nm": {"min": float(grid[0]), "max": float(grid[-1]),
                     "step": float(args.wl_step), "n_points": G},
        "layout": {"excitation": [0, G], "emission": [G, 2 * G],
                    "note": "grid_nm indexes each half; spectra are peak-normalized to 1.0"},
        "skipped_no_sequence": skipped_no_seq,
        "skipped_nonstandard_aa": skipped_nonstd,
        "citation": "FPbase: Lambert TJ, Nat Methods 2019. Data (c) FPbase contributors.",
    }
    json.dump(meta, open(os.path.join(args.outdir, "meta.json"), "w"), indent=2)

    print(f"wrote {len(rows)} samples ({n_unique_seq} unique sequences) -> {os.path.normpath(args.outdir)}")
    print(f"  spectra.npy   {spectra_arr.shape}  (ex[0:{G}] + em[{G}:{2*G}])")
    print(f"  grid          {grid[0]:.0f}-{grid[-1]:.0f} nm @ {args.wl_step:.0f} nm  ({G} pts)")
    print(f"  ex_max range  {min(r['ex_max'] for r in rows):.0f}-{max(r['ex_max'] for r in rows):.0f} nm")
    print(f"  em_max range  {min(r['em_max'] for r in rows):.0f}-{max(r['em_max'] for r in rows):.0f} nm")
    print(f"  seq_len       {min(r['seq_len'] for r in rows)}-{max(r['seq_len'] for r in rows)}")
    dups = len(rows) - n_unique_seq
    print(f"  duplicate-sequence samples (multi-state etc.): {dups}")
    if skipped_no_seq or skipped_nonstd:
        print(f"  skipped: {skipped_no_seq} no-sequence, {skipped_nonstd} non-standard-AA")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proteins", default=DEF_PROTEINS)
    p.add_argument("--spectra", default=DEF_SPECTRA)
    p.add_argument("--outdir", default=DEF_OUTDIR)
    p.add_argument("--wl-min", type=float, default=300.0, help="grid start (nm)")
    p.add_argument("--wl-max", type=float, default=800.0, help="grid end (nm)")
    p.add_argument("--wl-step", type=float, default=1.0, help="grid step (nm)")
    build(p.parse_args())


if __name__ == "__main__":
    main()
