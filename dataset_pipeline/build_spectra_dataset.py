#!/usr/bin/env python3
"""
Full-spectrum FP dataset builder.

Does NOT re-curate from FPbase independently -- it filters the already-curated **peak**
dataset (`data/peak/curated/peaks_assignments.csv`) down to the rows whose peak-resolved
state has a full measured excitation+emission curve pair, then extracts those curves for
exactly that state. Sequence resolution (Stage C) and the NN-4mer filter (Stage E) are
peak's, unchanged -- this script adds no curation logic of its own beyond the curve lookup.

Rationale: an earlier version of this script re-ran Stages A/B2/C/E from scratch against the
455-protein spectra-bearing population. That let a few analyte sensors (mKeima, pHmScarlet)
and a FRET biosensor (GRvT) leak back in: they report several states with wildly different
ex/em, but FPbase only ever measured a full curve pair for one of those states, so a
resolution step that only sees curve-bearing states had nothing to disagree with. Peak's own
Stage C already saw every reported state for these proteins and correctly dropped them as
`analyte_sensor` / `drop_ambiguous`. Filtering peak's output instead of re-deriving it means
that judgment is inherited for free, and can't regress independently.

Source data: data/peak/curated/peaks_assignments.csv (run `build_dataset.py --target peak`
first) plus ../fpbase-extractor/fpbase_output/fpbase_spectra.json for the measured curves.

Writes to <outdir>:
    excitation.npy              (N, L_ex) float32, resampled onto excitation_wavelengths.npy
    emission.npy                (N, L_em) float32, resampled onto emission_wavelengths.npy
    excitation_wavelengths.npy  (L_ex,) float32, nm
    emission_wavelengths.npy    (L_em,) float32, nm
    sequences.fasta             one record per row, header >index|slug|state
    spectra_assignments.csv     the peak assignments row for each kept protein
    curate_meta.json            provenance: which peak rows were kept/dropped and why

Usage:
    python build_spectra_dataset.py --outdir data/spectra/curated
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import date

import numpy as np

from build_dataset import HERE, DEF_PROTEINS

DEF_SPECTRA = os.path.join(HERE, "..", "fpbase-extractor", "fpbase_output", "fpbase_spectra.json")
DEF_PEAK_ASSIGNMENTS = os.path.join(HERE, "data", "peak", "curated", "peaks_assignments.csv")

# Grids chosen from the curated set's own observed range: EX spans it losslessly (230-800nm
# covers every curated excitation curve end to end); EM is trimmed to 250-900nm, which loses
# <0.5% of any curve's intensity mass (a few far-red proteins carry a long near-zero tail out
# to ~1600nm that isn't worth the padding). See dataset_pipeline/README.md for the check.
EX_GRID = np.arange(230.0, 800.0 + 1.0, 1.0)
EM_GRID = np.arange(250.0, 900.0 + 1.0, 1.0)


def _label(protein_name, state_name):
    """Strip a state's 'Protein (Label)' name down to just Label, case-folded.

    Falls back to the full (lowercased) state name if it isn't in that form -- this only
    matters for matching against the spectra file's own short state labels below.
    """
    prefix = (protein_name or "") + " ("
    if state_name and state_name.startswith(prefix) and state_name.endswith(")"):
        return state_name[len(prefix):-1].strip().lower()
    return (state_name or "").strip().lower()


def _match_state_idx(spectrum_state, protein_name, states):
    """Map a fpbase_spectra.json 'state' label (e.g. 'Green', 'default') to an index into
    the protein's own `states` list. Exact label match if unambiguous, else a substring
    fallback; single-state proteins match trivially. Returns None if unresolved."""
    if len(states) == 1:
        return 0
    target = spectrum_state.strip().lower()
    labels = [_label(protein_name, st.get("name")) for st in states]
    if labels.count(target) == 1:
        return labels.index(target)
    candidates = [i for i, l in enumerate(labels) if target in l or l in target]
    return candidates[0] if len(candidates) == 1 else None


def _resample(data, grid):
    """Linear-interpolate a [[wavelength, intensity], ...] curve onto `grid`, zero-filled
    outside the curve's own measured domain."""
    xp = np.asarray([pt[0] for pt in data], dtype=np.float64)
    fp = np.asarray([pt[1] for pt in data], dtype=np.float64)
    return np.interp(grid, xp, fp, left=0.0, right=0.0)


def _curves_by_state_name(protein, spectra_record):
    """slug's protein['states'] name -> {'excitation': data, 'emission': data} for states that
    have both curves measured. Reuses the validated label-matching logic to align FPbase's
    short spectra-file state labels back onto the protein's own full state names."""
    states = protein.get("states") or []
    by_idx = defaultdict(dict)
    for s in spectra_record["spectra"]:
        if s["spectrum_type"] not in ("excitation", "emission"):
            continue
        idx = _match_state_idx(s["state"], protein.get("name"), states)
        if idx is not None:
            by_idx[idx][s["spectrum_type"]] = s["data"]
    out = {}
    for idx, curves in by_idx.items():
        if "excitation" in curves and "emission" in curves:
            out[states[idx].get("name") or "default"] = curves
    return out


def build(peak_assignments_path, proteins_path, spectra_path, outdir):
    if not os.path.exists(peak_assignments_path):
        raise SystemExit(
            f"{peak_assignments_path} not found -- run "
            f"`python build_dataset.py --target peak` first."
        )
    peak_rows = list(csv.DictReader(open(peak_assignments_path)))
    proteins = json.load(open(proteins_path))
    all_spectra = json.load(open(spectra_path))
    pmap = {p["slug"]: p for p in proteins}
    smap = {s["slug"]: s for s in all_spectra}

    stats = defaultdict(int)
    dropped_names = defaultdict(list)
    kept = []

    for r in peak_rows:
        slug = r["slug"]
        sp = smap.get(slug)
        if sp is None:
            stats["skip_no_spectra_measured"] += 1
            dropped_names["no_spectra_measured"].append(r["name"])
            continue
        p = pmap.get(slug)
        curves_by_state = _curves_by_state_name(p, sp)
        curves = curves_by_state.get(r["state"])
        if curves is None:
            stats["skip_no_curve_pair_for_resolved_state"] += 1
            dropped_names["no_curve_pair_for_resolved_state"].append(r["name"])
            continue
        row = dict(r)
        row["_excitation"] = curves["excitation"]
        row["_emission"] = curves["emission"]
        kept.append(row)

    curated = kept  # peak already applied Stage E; no further NN filtering here

    # ---- resample onto shared grids + write outputs ----
    os.makedirs(outdir, exist_ok=True)
    ex_arr = np.stack([_resample(r["_excitation"], EX_GRID) for r in curated]).astype(np.float32)
    em_arr = np.stack([_resample(r["_emission"], EM_GRID) for r in curated]).astype(np.float32)
    np.save(os.path.join(outdir, "excitation.npy"), ex_arr)
    np.save(os.path.join(outdir, "emission.npy"), em_arr)
    np.save(os.path.join(outdir, "excitation_wavelengths.npy"), EX_GRID.astype(np.float32))
    np.save(os.path.join(outdir, "emission_wavelengths.npy"), EM_GRID.astype(np.float32))

    with open(os.path.join(outdir, "sequences.fasta"), "w") as fh:
        for i, r in enumerate(curated):
            sid = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in f"{r['slug']}|{r['state']}")
            fh.write(f">{i}|{sid}\n{r['seq']}\n")

    cols = list(peak_rows[0].keys())  # peaks_assignments.csv's own columns, "index" already first
    with open(os.path.join(outdir, "spectra_assignments.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(curated):
            r = dict(r); r["index"] = i
            w.writerow(r)

    n_unique = len({r["seq"] for r in curated})
    meta = {
        "created": date.today().isoformat(),
        "target": "spectra",
        "source_peak_assignments": os.path.relpath(peak_assignments_path, outdir),
        "source_proteins": os.path.relpath(proteins_path, outdir),
        "source_spectra": os.path.relpath(spectra_path, outdir),
        "pipeline": ("filter data/peak/curated (peak's own A|B1|B2|C|D|E, unchanged) down to "
                     "rows whose peak-resolved state has a measured excitation+emission curve pair"),
        "params": {
            "ex_grid_nm": [float(EX_GRID[0]), float(EX_GRID[-1])],
            "em_grid_nm": [float(EM_GRID[0]), float(EM_GRID[-1])],
        },
        "counts": {
            "peak_rows": len(peak_rows),
            "curated": len(curated), "unique_sequences": n_unique,
            "skipped_no_spectra_measured": stats["skip_no_spectra_measured"],
            "skipped_no_curve_pair_for_resolved_state": stats["skip_no_curve_pair_for_resolved_state"],
        },
        "dropped_names": {k: sorted(set(v)) for k, v in dropped_names.items() if v},
    }
    json.dump(meta, open(os.path.join(outdir, "curate_meta.json"), "w"), indent=2)

    print(f"[spectra] {len(curated)}/{len(peak_rows)} peak rows have a full curve pair for their "
          f"resolved state -> {os.path.normpath(outdir)}")
    print(f"   dropped: no measured spectra at all {stats['skip_no_spectra_measured']}, "
          f"measured but not for the resolved state {stats['skip_no_curve_pair_for_resolved_state']}")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peak-assignments", default=DEF_PEAK_ASSIGNMENTS)
    ap.add_argument("--proteins", default=DEF_PROTEINS)
    ap.add_argument("--spectra", default=DEF_SPECTRA)
    ap.add_argument("--outdir", default=os.path.join(HERE, "data", "spectra", "curated"))
    a = ap.parse_args()
    build(a.peak_assignments, a.proteins, a.spectra, a.outdir)


if __name__ == "__main__":
    main()
