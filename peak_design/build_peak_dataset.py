#!/usr/bin/env python3
"""
Build the peak-conditioned training inputs for the *peak_design* workflow.

Unlike the full-spectrum pipeline (../design), peak conditioning only needs each protein's
**reported peak wavelengths** (ex_max, em_max) plus its sequence -- NOT the resampled ex/em
curves. So we read the fpbase-extractor protein export directly (`fpbase_proteins.json`),
which yields ~2x more usable samples than the spectra-gated `ESM-spectrum` dataset
(every (protein, state) that has a sequence and FPbase-reported ex_max & em_max, rather
than only those with full excitation AND emission curves).

A sample = one (protein, state) with a standard-AA sequence and reported ex_max & em_max.
A protein with several spectral states contributes one (protein, state) row per state, so a
sequence can repeat across states (tracked via `seq_group`).

Multi-state collapse (on by default; disable with --no-collapse-multistate)
---------------------------------------------------------------------------
A sequence->peak regressor needs one peak per sequence, and a sequence that appears under
multiple states both (a) gives the model inconsistent targets and (b) leaks across the
train/test split (the split is per-row, so the same sequence can sit in both). We therefore
collapse each `seq_group` that has >1 state to a single row. The rule applies *only* to
multi-state seq_groups; single-state samples are never touched. Per group:
    1. drop_sensor    -- pH / Ca2+ sensor states (names ~ pH|acidic|alkaline|basic|ecliptic|
                         calcium): the peak is set by the environment/analyte, not the folded
                         sequence -- same rationale as the existing frFAST/nirFAST drop.
    2. dedup_identical -- all states share one (ex_max, em_max): keep any one.
    3. drop_ambiguous -- emission spread < EM_EPS nm (e.g. excitation-only variants, or a
                         dual-bright photoswitch): emission cannot identify a native precursor.
    4. keep_min_emission -- photoconvertible / multistate / timer: keep the shortest-emission
                         state = the native, as-folded precursor. Green->red conversion is an
                         irreversible, conjugation-extending (always red-shifting) reaction, so
                         the bluest state is necessarily the un-converted one.

Writes (default outdir: ./training_data):
    peaks.npy            float32 (N, 2)  -- [ex_max, em_max] in nm, row-aligned to `index`
    sequences.fasta                      -- one record per sample, header `>index|slug|state`
    peak_assignments.csv  N rows aligned to peaks.npy / sequences
    peak_meta.json        provenance: source, counts, peak ranges

Per-residue ESM-2 embeddings are NOT built here; the notebooks compute (and cache) them on
first run.

Usage
-----
    python build_peak_dataset.py
    python build_peak_dataset.py --proteins ../fpbase-extractor/fpbase_output/fpbase_proteins.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import date

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_PROTEINS = os.path.join(HERE, "..", "fpbase-extractor", "fpbase_output", "fpbase_proteins.json")
DEF_OUTDIR = os.path.join(HERE, "training_data")

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# State whose peak is set by the environment/analyte (pH, Ca2+) rather than the folded
# sequence -- detected from the FPbase state name. Same logic that drops frFAST/nirFAST.
SENSOR_RE = re.compile(r"\bpH\b|pH\s*\d|acidic|alkaline|\bbasic\b|ecliptic|calcium", re.I)
# Min-emission only identifies a native precursor when the states' emissions actually
# differ. Below this spread (nm) the states are excitation-only variants or a dual-bright
# switch, and we cannot tell which is fundamental -> drop. Real green->red conversions
# separate emission by tens of nm, well above this threshold.
EM_EPS = 10.0


def clean_seq(seq):
    """Uppercase; flag whether sequence is purely standard amino acids."""
    s = (seq or "").strip().upper()
    return s, (len(s) > 0 and all(c in STANDARD_AA for c in s))


def _decide_multistate(grows):
    """Pick the surviving row index (into `grows`) for a >1-state seq_group, or None to drop.

    Returns (decision_label, local_index_or_None).
    """
    states = [r["state"] for r in grows]
    pairs = {(r["ex_max"], r["em_max"]) for r in grows}
    ems = [r["em_max"] for r in grows]
    if any(SENSOR_RE.search(s or "") for s in states):
        return "drop_sensor", None
    if len(pairs) == 1:
        return "dedup_identical", 0
    if max(ems) - min(ems) < EM_EPS:
        return "drop_ambiguous", None
    # photoconvertible / multistate / timer: native precursor = shortest emission
    return "keep_min_emission", min(range(len(grows)), key=lambda k: grows[k]["em_max"])


def collapse_seq_groups(rows):
    """Collapse every >1-state seq_group to one row. Single-state groups pass through.

    Returns (kept_local_indices_sorted, stats_dict).
    """
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["seq_group"]].append(i)
    keep, stats, dropped = [], Counter(), []
    for idxs in groups.values():
        if len(idxs) == 1:
            keep.append(idxs[0])
            stats["single_state"] += 1
            continue
        decision, local = _decide_multistate([rows[i] for i in idxs])
        stats[decision] += 1
        if local is None:
            dropped.append(rows[idxs[0]]["name"])
        else:
            keep.append(idxs[local])
    keep.sort()
    return keep, {"counts": dict(stats), "dropped_multistate_names": dropped}


def build(args):
    proteins = json.load(open(args.proteins))

    rows, peak_rows, fasta = [], [], []
    seq_to_group = {}
    skipped_no_seq = skipped_nonstd = skipped_no_peak = 0

    for p in proteins:
        seq, ok = clean_seq(p.get("seq"))
        if not seq:
            skipped_no_seq += 1
            continue
        if not ok:
            skipped_nonstd += 1
            continue
        for st in (p.get("states") or []):
            ex, em = st.get("ex_max"), st.get("em_max")
            if ex is None or em is None:
                skipped_no_peak += 1
                continue
            idx = len(rows)
            gid = seq_to_group.setdefault(seq, len(seq_to_group))   # identical-seq grouping (samples only)
            rows.append({
                "index": idx,
                "slug": p.get("slug", ""),
                "name": p.get("name", ""),
                "state": st.get("name") or "default",
                "seq_group": gid,
                "parent_organism": p.get("parent_organism") or "",
                "switch_type": p.get("switch_type_label") or "",
                "oligomerization": p.get("oligomerization") or "",
                "is_dark": bool(st.get("is_dark")),
                "ex_max": round(float(ex), 1),
                "em_max": round(float(em), 1),
                "ref_year": p.get("ref_year") or "",
                "seq_len": len(seq),
                "aliases": "; ".join(p.get("aliases") or []),
                "seq": seq,
            })
            peak_rows.append([float(ex), float(em)])
            fasta.append((idx, p.get("slug", ""), st.get("name") or "default"))

    if not rows:
        raise SystemExit(f"no usable (protein, state) samples found in {args.proteins}")

    n_states = len(rows)
    collapse_stats = None
    if args.collapse_multistate:
        keep, collapse_stats = collapse_seq_groups(rows)
        rows = [rows[i] for i in keep]
        peak_rows = [peak_rows[i] for i in keep]
        # re-index rows, renumber seq_groups contiguously, and rebuild the fasta records
        regroup = {}
        for new_idx, r in enumerate(rows):
            r["index"] = new_idx
            r["seq_group"] = regroup.setdefault(r["seq_group"], len(regroup))
        fasta = [(r["index"], r["slug"], r["state"]) for r in rows]

    peaks = np.asarray(peak_rows, dtype=np.float32)
    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "peaks.npy"), peaks)

    with open(os.path.join(args.outdir, "sequences.fasta"), "w") as fh:
        for idx, slug, state in fasta:
            sid = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in f"{slug}|{state}")
            fh.write(f">{idx}|{sid}\n{rows[idx]['seq']}\n")

    cols = ["index", "slug", "name", "state", "seq_group", "parent_organism", "switch_type",
            "oligomerization", "is_dark", "ex_max", "em_max", "ref_year", "seq_len", "aliases", "seq"]
    with open(os.path.join(args.outdir, "peak_assignments.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    n_unique_seq = len({r["seq"] for r in rows})
    meta = {
        "created": date.today().isoformat(),
        "source_proteins": os.path.relpath(args.proteins, args.outdir),
        "n_samples": len(rows),
        "n_unique_sequences": n_unique_seq,
        "targets": ["ex_max", "em_max"],
        "units": "nm",
        "ex_max_range": [float(peaks[:, 0].min()), float(peaks[:, 0].max())],
        "em_max_range": [float(peaks[:, 1].min()), float(peaks[:, 1].max())],
        "skipped_no_sequence": skipped_no_seq,
        "skipped_nonstandard_aa": skipped_nonstd,
        "skipped_state_missing_peak": skipped_no_peak,
        "collapse_multistate": bool(args.collapse_multistate),
        "n_states_before_collapse": n_states,
        "collapse": collapse_stats,
        "note": "Peaks are FPbase-reported ex_max/em_max per (protein, state); no spectral curve required.",
        "citation": "FPbase: Lambert TJ, Nat Methods 2019. Data (c) FPbase contributors.",
    }
    json.dump(meta, open(os.path.join(args.outdir, "peak_meta.json"), "w"), indent=2)

    dups = len(rows) - n_unique_seq
    print(f"wrote {len(rows)} samples ({n_unique_seq} unique sequences) -> {os.path.normpath(args.outdir)}")
    print(f"  peaks.npy     {peaks.shape}  (col0 ex_max, col1 em_max)")
    print(f"  ex_max range  {peaks[:, 0].min():.0f}-{peaks[:, 0].max():.0f} nm")
    print(f"  em_max range  {peaks[:, 1].min():.0f}-{peaks[:, 1].max():.0f} nm")
    print(f"  seq_len       {min(r['seq_len'] for r in rows)}-{max(r['seq_len'] for r in rows)}")
    if args.collapse_multistate:
        c = collapse_stats["counts"]
        print(f"  collapsed multi-state seq_groups: {n_states} states -> {len(rows)} samples")
        print(f"    kept min-emission {c.get('keep_min_emission', 0)}, dedup-identical {c.get('dedup_identical', 0)}; "
              f"dropped sensor {c.get('drop_sensor', 0)}, ambiguous {c.get('drop_ambiguous', 0)}")
        print(f"    dropped: {collapse_stats['dropped_multistate_names']}")
    else:
        print(f"  duplicate-sequence samples (multi-state etc.): {dups}")
    print(f"  skipped: {skipped_no_seq} no-seq, {skipped_nonstd} non-standard-AA, {skipped_no_peak} state-without-peak")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proteins", default=DEF_PROTEINS, help="fpbase-extractor fpbase_proteins.json")
    p.add_argument("--outdir", default=DEF_OUTDIR, help="where to write peaks.npy / sequences.fasta / peak_assignments.csv")
    p.add_argument("--collapse-multistate", default=True, action=argparse.BooleanOptionalAction,
                   help="collapse each >1-state seq_group to one row (native precursor / drop sensors); "
                        "use --no-collapse-multistate to keep every (protein, state) as a separate sample")
    build(p.parse_args())


if __name__ == "__main__":
    main()
