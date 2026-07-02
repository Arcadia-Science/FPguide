#!/usr/bin/env python3
"""Compare the refactored curated datasets to the existing peak_design/training_data*/curated sets."""
import csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.join(HERE, "..", "peak_design")

TRAITS = {
    "peak":       ("data/peak/curated/peaks_assignments.csv",
                   f"{PD}/training_data/curated/peak_assignments.csv"),
    "brightness": ("data/brightness/curated/brightness_assignments.csv",
                   f"{PD}/training_data_brightness/curated/brightness_assignments.csv"),
    "pka":        ("data/pka/curated/pka_assignments.csv",
                   f"{PD}/training_data_pka/curated/pka_assignments.csv"),
}


def load(path):
    rows = list(csv.DictReader(open(path)))
    by_name = {r["name"]: r for r in rows}
    return rows, by_name, {r["seq"] for r in rows}


for trait, (new_p, old_p) in TRAITS.items():
    new_rows, new_by, new_seqs = load(os.path.join(HERE, new_p))
    old_rows, old_by, old_seqs = load(old_p)
    added = sorted(set(new_by) - set(old_by))
    removed = sorted(set(old_by) - set(new_by))
    meta = json.load(open(os.path.join(HERE, os.path.dirname(new_p), "curate_meta.json")))
    print("=" * 78)
    print(f"### {trait.upper()}   refactored={len(new_rows)}  existing={len(old_rows)}  "
          f"(Δ={len(new_rows)-len(old_rows):+d})")
    print(f"  sequences: refactored∩existing={len(new_seqs & old_seqs)}, "
          f"only-refactored={len(new_seqs - old_seqs)}, only-existing={len(old_seqs - new_seqs)}")
    print(f"\n  + NEWLY KEPT (in refactor, not existing): {len(added)}")
    for n in added:
        r = new_by[n]
        print(f"      {n:32} state={r['state'][:26]:26} reason={r.get('resolve_reason','')}")
    print(f"\n  - NEWLY DROPPED (in existing, not refactor): {len(removed)}")
    # attribute each removal to a refactor stage
    dn = meta["dropped_names"]
    def why(name):
        for reason, names in dn.items():
            if name in names:
                return reason
        return "?"
    for n in removed:
        print(f"      {n:32} -> {why(n)}")
    print()
