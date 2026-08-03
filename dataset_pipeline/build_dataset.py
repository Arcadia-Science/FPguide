#!/usr/bin/env python3
"""
Unified FP dataset builder + curator (refactor of build_peak_dataset.py + the three
build_curate_split notebooks), targeting peak / brightness / pKa from one code path.

Pipeline (the split + visualization are intentionally NOT done here):

  Stage A  std-AA intake ............... drop sequences with non-standard residues (X/B/Z/U/O)
  Stage B  genuine-FP filters .......... (B1) emission gate: require ex_max AND em_max
                                          (B2) exogenous-signal exclusion: cofactor tag
                                               OR fluorogen-activating FAST family
                                               OR retinal opsins / channelrhodopsins
           -> yields the (uncollapsed) genuine-FP superset, shared across all traits
  Stage C  per-trait sequence resolution  collapse each identical-sequence group to ONE row,
                                          target-aware, runs BEFORE the target gate so an
                                          analyte state's value can never be mislabeled onto
                                          the sequence. Drops analyte sensors when the target
                                          is not consistent across the sequence's states
                                          (peaks/brightness); keeps them for pKa (state-invariant).
                                          Photoconversions resolve to the min-EMISSION native precursor.
  Stage D  per-trait target gate ....... keep the resolved row only if it carries the target
  Stage E  per-trait NN-4mer filter .... drop sequence-isolated rows (NN char-4-gram cosine < 0.10)

Writes, per trait, to <outdir>:
    <target>.npy            (N,2) for peak / (N,1) for brightness|pka
    sequences.fasta
    <target>_assignments.csv
    curate_meta.json        full provenance incl. per-stage drop reasons

Usage:
    python build_dataset.py --target peak       --outdir data/peak/curated
    python build_dataset.py --target brightness --outdir data/brightness/curated
    python build_dataset.py --target pka        --outdir data/pka/curated
    python build_dataset.py --all               # build peak (the maintained target) under ./data/peak/curated

brightness and pKa are no longer part of the actively maintained pipeline -- their previously
generated data now lives under archive/data_brightness/ and archive/data_pka/ (see README). The
rules above still build them on request via --target; --all does not, so it doesn't regenerate
data the project has moved away from.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from datetime import date

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_PROTEINS = os.path.join(HERE, "..", "fpbase-extractor", "fpbase_output", "fpbase_proteins.json")

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ---- Stage B2: exogenous-signal detection (name + aliases only; NEVER organism) ----
# Fluorogen-activating tags are always written all-caps FAST (FAST/pFAST/frFAST/nirFAST/...),
# so match them CASE-SENSITIVELY -- avoids false positives on 'Fast-FT' (a timer), 'sf:fast.3'
# (a designed variant), and aliases containing lowercase 'fast' (e.g. ffDronpa).
FAST_RE = re.compile(r"\b(?:fr|nir|p)?FAST\b")
# Retinal opsins / channelrhodopsins (case-insensitive).
OPSIN_RE = re.compile(r"channelrhod|rhodopsin|opsin|\bChR\d?\b", re.I)
# Biliverdin IR-FPs missed by the cofactor='BV' tag. iFP2.0 (Deinococcus) is the only untagged
# bilin protein with spectra that would otherwise leak in -- the other untagged phytochromes
# (hfriFP, RpBphP2/6) report no ex/em and are already removed by the emission gate.
MANUAL_EXCLUDE = {"iFP2.0"}

# ---- Stage C: analyte-condition tag (backstop for single-reported-value sensors) ----
# NOTE: '\bbasic\b' is deliberately omitted -- it only false-positives ordinary FPs.
ANALYTE_RE = re.compile(r"calcium|\bpH\b|pH\s*\d|acidic|alkaline|ecliptic", re.I)

EM_EPS = 10.0            # nm; emission spread above which a multi-state group is a real (red-shifting) photoconversion
NN_THRESHOLD = 0.10      # Stage E: nearest-neighbor 4-mer cosine similarity floor
# Light-controlled FPs: for a same-color on/off bimodal target, keep the ON state instead of
# dropping it as ambiguous. (Photoconvertibles are green->red separated, so they resolve to the
# native green precursor via keep_min_emission first -- that IS their default on-state.)
ONSTATE_SWITCHES = {"photoactivatable", "photoswitchable", "photoconvertible"}

TRAITS = {
    #  field      : per-state key in fpbase_proteins.json (None => (ex_max, em_max))
    #  drop_sensors: apply analyte-sensor drop (True for peak/brightness, False for pKa)
    #  agree_tol  : scalar spread within which multiple state values count as "consistent"
    "peak":       {"field": None,         "drop_sensors": True,  "agree_tol": None},
    "brightness": {"field": "brightness", "drop_sensors": True,  "agree_tol": 0.15},   # relative
    "pka":        {"field": "pka",        "drop_sensors": False, "agree_tol": 0.15},    # pH units (absolute)
}


def clean_seq(seq):
    s = (seq or "").strip().upper()
    return s, (len(s) > 0 and all(c in STANDARD_AA for c in s))


def is_exogenous(p):
    """Reason string if the signal is set by a bound molecule (cofactor / fluorogen / retinal /
    biliverdin) rather than the folded sequence, else None."""
    if (p.get("cofactor") or "").strip().lower() not in ("", "none"):
        return "cofactor"
    if p.get("name") in MANUAL_EXCLUDE:
        return "manual_irfp"
    text = (p.get("name") or "") + " " + " ".join(p.get("aliases") or [])
    if FAST_RE.search(text):
        return "fluorogen"
    if OPSIN_RE.search(text):
        return "opsin"
    return None


def _target_value(st, field):
    """Return the trait target for a state: (ex,em) tuple for peak, else the scalar (or None)."""
    if field is None:
        ex, em = st.get("ex_max"), st.get("em_max")
        return None if (ex is None or em is None) else (round(float(ex), 1), round(float(em), 1))
    v = st.get(field)
    return None if v is None else round(float(v), 3)


def _scalar_spread_ok(vals, tol, relative):
    if len(vals) <= 1:
        return True
    lo, hi = min(vals), max(vals)
    if relative:
        return (hi - lo) <= tol * max(abs(hi), 1e-9)
    return (hi - lo) <= tol


def _is_analyte_sensor(states, cfg):
    """A tagged (pH/Ca2+/...) multi-condition construct whose target is NOT consistent across states.

    Tagged + consistent target (e.g. ecliptic pHluorin's fixed 395/509 peak) is NOT a sensor here --
    the value is well-defined for the sequence, so it is kept.
    """
    if not any(ANALYTE_RE.search(s.get("name") or "") for s in states):
        return False  # not tagged -> let the data-driven ambiguity branch decide
    field = cfg["field"]
    tvals = [v for s in states if (v := _target_value(s, field)) is not None]
    n_genuine = len(states)
    if field is None:  # peak: consistent iff a single unique (ex,em) across all states
        consistent = len({v for v in tvals}) == 1 and len(tvals) == n_genuine
    else:              # scalar: every state reports it AND the values agree
        consistent = (len(tvals) == n_genuine) and _scalar_spread_ok(
            tvals, cfg["agree_tol"], relative=(cfg["field"] == "brightness"))
    return not consistent


def _resolve_group(states, cfg, switch_types=frozenset()):
    """Collapse one identical-sequence group to a single surviving state index, or None to drop.

    Returns (local_index_or_None, reason).
    """
    field = cfg["field"]
    # states carrying the target, with their emissions (for the min-emission precursor rule)
    tstates = [(i, s) for i, s in enumerate(states) if _target_value(s, field) is not None]
    if not tstates:
        return None, "no_target"

    if cfg["drop_sensors"] and _is_analyte_sensor(states, cfg):
        return None, "analyte_sensor"

    if field is None:                                    # ---- peak target ----
        pairs = {_target_value(s, None) for _, s in tstates}
        if len(pairs) == 1:
            return tstates[0][0], "dedup_identical"
        ems = [s["em_max"] for _, s in tstates]
        if max(ems) - min(ems) >= EM_EPS:                # real green->red conversion
            k = min(range(len(tstates)), key=lambda j: tstates[j][1]["em_max"])
            return tstates[k][0], "keep_min_emission"
        return None, "drop_ambiguous"                    # excitation-only / ratiometric, untagged

    # ---- scalar target (brightness / pka) ----
    if len(tstates) == 1:
        return tstates[0][0], "single_state"
    tvals = [_target_value(s, field) for _, s in tstates]
    if _scalar_spread_ok(tvals, cfg["agree_tol"], relative=(field == "brightness")):
        return tstates[0][0], "dedup_consistent"
    ems = [s.get("em_max") for _, s in tstates]
    if None not in ems and max(ems) - min(ems) >= EM_EPS:
        k = min(range(len(tstates)), key=lambda j: tstates[j][1]["em_max"])
        return tstates[k][0], "keep_min_emission"
    if switch_types & ONSTATE_SWITCHES:      # same-color on/off switch -> keep the ON state
        cand = [(i, s) for i, s in tstates if not s.get("is_dark")] or tstates
        k = max(range(len(cand)), key=lambda j: _target_value(cand[j][1], field))
        return cand[k][0], "keep_on_state"
    return None, "drop_ambiguous"


def build(target, proteins_path, outdir):
    cfg = TRAITS[target]
    field = cfg["field"]
    proteins = json.load(open(proteins_path))

    stats = defaultdict(int)
    dropped_names = defaultdict(list)          # reason -> [names]
    groups = defaultdict(list)                 # seq -> list of (protein, state)

    # ---- Stage A + B: build the genuine-FP superset ----
    for p in proteins:
        seq, ok = clean_seq(p.get("seq"))
        if not seq:
            stats["skip_no_sequence"] += 1
            continue
        if not ok:
            stats["skip_nonstandard_aa"] += 1
            dropped_names["nonstandard_aa"].append(p.get("name", ""))
            continue
        exo = is_exogenous(p)
        if exo:
            stats["skip_exogenous"] += 1
            stats[f"exogenous_{exo}"] += 1
            dropped_names[f"exogenous_{exo}"].append(p.get("name", ""))
            continue
        kept_states = [st for st in (p.get("states") or [])
                       if st.get("ex_max") is not None and st.get("em_max") is not None]
        if not kept_states:
            stats["skip_no_emission"] += 1
            continue
        for st in kept_states:
            groups[seq].append((p, st))

    # ---- Stage C + D: per-trait sequence resolution + target gate ----
    kept = []   # dicts (one per surviving sequence)
    for seq, ps in groups.items():
        states = [st for _, st in ps]
        switch_types = {p.get("switch_type_label") or "" for p, _ in ps}
        local, reason = _resolve_group(states, cfg, switch_types)
        stats[f"resolve_{reason}"] += 1
        if local is None:
            dropped_names[reason].append(ps[0][0].get("name", ""))
            continue
        p, st = ps[local]
        tv = _target_value(st, field)
        row = {
            "slug": p.get("slug", ""), "name": p.get("name", ""),
            "state": st.get("name") or "default", "parent_organism": p.get("parent_organism") or "",
            "switch_type": p.get("switch_type_label") or "", "oligomerization": p.get("oligomerization") or "",
            "is_dark": bool(st.get("is_dark")),
            "ex_max": round(float(st["ex_max"]), 1), "em_max": round(float(st["em_max"]), 1),
            "ref_year": p.get("ref_year") or "", "seq_len": len(seq),
            "aliases": "; ".join(p.get("aliases") or []), "seq": seq,
            "n_states_in_group": len(states), "resolve_reason": reason,
        }
        if field == "peak" or field is None:
            row["_target"] = [row["ex_max"], row["em_max"]]
        else:
            row[field] = tv
            row["_target"] = [tv]
        kept.append(row)

    n_after_resolve = len(kept)

    # ---- Stage E: NN 4-mer similarity filter (dataset-relative) ----
    seqs = [r["seq"] for r in kept]
    Xk = CountVectorizer(analyzer="char", ngram_range=(4, 4)).fit_transform(seqs)
    Sk = cosine_similarity(Xk)
    np.fill_diagonal(Sk, -1.0)
    nn = Sk.max(1)
    nn_drop = nn < NN_THRESHOLD
    stats["nn_dropped"] = int(nn_drop.sum())
    dropped_names["nn_isolated"] = [kept[i]["name"] for i in np.where(nn_drop)[0]]
    curated = [r for i, r in enumerate(kept) if not nn_drop[i]]

    # ---- write outputs ----
    os.makedirs(outdir, exist_ok=True)
    tname = "peaks" if target == "peak" else target
    tgt = np.asarray([r["_target"] for r in curated], dtype=np.float32)
    np.save(os.path.join(outdir, f"{tname}.npy"), tgt)

    with open(os.path.join(outdir, "sequences.fasta"), "w") as fh:
        for i, r in enumerate(curated):
            sid = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in f"{r['slug']}|{r['state']}")
            fh.write(f">{i}|{sid}\n{r['seq']}\n")

    base_cols = ["index", "slug", "name", "state", "parent_organism", "switch_type",
                 "oligomerization", "is_dark", "ex_max", "em_max"]
    tail_cols = ["ref_year", "seq_len", "aliases", "n_states_in_group", "resolve_reason", "seq"]
    tcol = [] if field is None else [field]
    cols = base_cols + tcol + tail_cols
    with open(os.path.join(outdir, f"{tname}_assignments.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(curated):
            r = dict(r); r["index"] = i
            w.writerow(r)

    n_unique = len({r["seq"] for r in curated})
    meta = {
        "created": date.today().isoformat(),
        "target": target,
        "source_proteins": os.path.relpath(proteins_path, outdir),
        "pipeline": "A std-AA | B1 emission-gate | B2 exogenous | C resolve(target-aware) | D target-gate | E NN4mer<0.10",
        "params": {"EM_EPS": EM_EPS, "NN_THRESHOLD": NN_THRESHOLD,
                   "drop_analyte_sensors": cfg["drop_sensors"], "agree_tol": cfg["agree_tol"]},
        "counts": {
            "curated": len(curated), "unique_sequences": n_unique,
            "after_resolve": n_after_resolve, "nn_dropped": int(nn_drop.sum()),
            "skipped_no_sequence": stats["skip_no_sequence"],
            "skipped_nonstandard_aa": stats["skip_nonstandard_aa"],
            "skipped_exogenous": stats["skip_exogenous"],
            "skipped_no_emission_states": stats["skip_no_emission"],
            "resolve": {k[len("resolve_"):]: v for k, v in sorted(stats.items()) if k.startswith("resolve_")},
        },
        "dropped_names": {k: sorted(set(v)) for k, v in dropped_names.items() if v},
    }
    if len(tgt):
        if field is None:
            meta["ex_max_range"] = [float(tgt[:, 0].min()), float(tgt[:, 0].max())]
            meta["em_max_range"] = [float(tgt[:, 1].min()), float(tgt[:, 1].max())]
        else:
            meta[f"{field}_range"] = [float(tgt.min()), float(tgt.max())]
    json.dump(meta, open(os.path.join(outdir, "curate_meta.json"), "w"), indent=2)

    print(f"[{target}] curated {len(curated)} ({n_unique} unique seq) -> {os.path.normpath(outdir)}")
    print(f"   Stage A/B skips: no-seq {stats['skip_no_sequence']}, non-std-AA {stats['skip_nonstandard_aa']}, "
          f"exogenous {stats['skip_exogenous']}, no-emission {stats['skip_no_emission']}")
    print(f"   Stage C resolve: " + ", ".join(
        f"{k[len('resolve_'):]} {v}" for k, v in sorted(stats.items()) if k.startswith("resolve_")))
    print(f"   Stage E NN<{NN_THRESHOLD}: dropped {int(nn_drop.sum())} -> {[kept[i]['name'] for i in np.where(nn_drop)[0]]}")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=list(TRAITS), help="which trait to build")
    ap.add_argument("--proteins", default=DEF_PROTEINS)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--all", action="store_true",
                     help="build peak (the maintained target) under ./data/peak/curated")
    a = ap.parse_args()
    if a.all:
        build("peak", a.proteins, os.path.join(HERE, "data", "peak", "curated"))
    else:
        if not a.target:
            ap.error("give --target or --all")
        outdir = a.outdir or os.path.join(HERE, "data", a.target, "curated")
        build(a.target, a.proteins, outdir)


if __name__ == "__main__":
    main()
