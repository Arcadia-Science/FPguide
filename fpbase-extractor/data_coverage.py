#!/usr/bin/env python3
"""Report how much FPbase data records each photophysical trait (or combinations).

Reads the `fpbase_proteins.json` produced by `fpbase-extract` and counts, over
all (protein, state) records, how many carry:

  * spectra      -- both ex_max AND em_max present
  * brightness   -- brightness present
  * maturation   -- maturation time present

It prints (1) a per-trait coverage table and (2) a breakdown over every
combination of the three traits (the full 2^3 presence/absence grid).

Counts are at the (protein, state) level by default, since one protein may have
several photophysical states. Use `--by protein` to instead count a protein as
covered if ANY of its states carries the trait.

Dependency-free (standard library only).

Examples:
    python3 data_coverage.py fpbase_output/fpbase_proteins.json
    python3 data_coverage.py data.json --by protein
    python3 data_coverage.py data.json --out coverage.md --format md
"""

import argparse
import json
import os
import sys

# Trait name -> predicate over a single state dict.
TRAITS = [
    ("spectra", lambda s: s.get("ex_max") is not None and s.get("em_max") is not None),
    ("brightness", lambda s: s.get("brightness") is not None),
    ("maturation", lambda s: s.get("maturation") is not None),
]
TRAIT_NAMES = [name for name, _ in TRAITS]


def find_default_json():
    """Look for the export in common locations."""
    for path in ("fpbase_proteins.json", os.path.join("fpbase_output", "fpbase_proteins.json")):
        if os.path.isfile(path):
            return path
    return None


def state_flags(state):
    """Return a tuple of booleans (one per trait) for a single state."""
    if not state:
        return tuple(False for _ in TRAITS)
    return tuple(pred(state) for _, pred in TRAITS)


def collect_records(proteins, by_state):
    """Return a list of flag-tuples, one per counted record.

    by_state=True  -> one record per (protein, state).
    by_state=False -> one record per protein, OR-ing flags across its states.
    """
    records = []
    for p in proteins:
        states = p.get("states") or []
        if by_state:
            if not states:
                records.append(state_flags(None))
            for state in states:
                records.append(state_flags(state))
        else:
            merged = [False] * len(TRAITS)
            for state in states:
                for i, flag in enumerate(state_flags(state)):
                    merged[i] = merged[i] or flag
            records.append(tuple(merged))
    return records


def per_trait_rows(records):
    """Count present/absent for each individual trait."""
    total = len(records)
    rows = []
    for i, name in enumerate(TRAIT_NAMES):
        present = sum(1 for r in records if r[i])
        pct = (100.0 * present / total) if total else 0.0
        rows.append([name, present, total - present, f"{pct:.1f}%"])
    return rows


def combination_rows(records):
    """Count records for every present/absent combination of all traits."""
    total = len(records)
    counts = {}
    for r in records:
        counts[r] = counts.get(r, 0) + 1

    rows = []
    # Iterate combos from all-present to none, descending by count.
    all_combos = sorted(counts.keys(), key=lambda c: (-counts[c], c))
    for combo in all_combos:
        present = [TRAIT_NAMES[i] for i, flag in enumerate(combo) if flag]
        label = " + ".join(present) if present else "(none)"
        n = counts[combo]
        pct = (100.0 * n / total) if total else 0.0
        rows.append([label] + ["Y" if f else "-" for f in combo] + [n, f"{pct:.1f}%"])
    return rows


def render(headers, rows, fmt):
    cells = [[str(c) for c in row] for row in rows]
    if fmt == "md":
        out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        for row in cells:
            out.append("| " + " | ".join(row) + " |")
        return "\n".join(out)
    if fmt == "tsv":
        return "\n".join(["\t".join(headers)] + ["\t".join(row) for row in cells])
    # text
    widths = [len(h) for h in headers]
    for row in cells:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    out.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in cells:
        out.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json_path", nargs="?", help="Path to fpbase_proteins.json (auto-detected if omitted)")
    parser.add_argument("--by", choices=["state", "protein"], default="state",
                        help="Count unit: per (protein, state) record [default] or per protein.")
    parser.add_argument("--format", choices=["text", "md", "tsv"], default="text", help="Output format (default: text)")
    parser.add_argument("--out", help="Write to this file instead of stdout")
    args = parser.parse_args(argv)

    path = args.json_path or find_default_json()
    if not path:
        parser.error("no JSON file given and none found. Run `fpbase-extract` first, or pass the path explicitly.")
    if not os.path.isfile(path):
        parser.error(f"file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        proteins = json.load(fh)

    by_state = args.by == "state"
    records = collect_records(proteins, by_state)
    unit = "(protein, state) records" if by_state else "proteins"

    per_trait = render(["trait", "with_data", "missing", "coverage"], per_trait_rows(records), args.format)
    combos = render(["combination"] + TRAIT_NAMES + ["count", "share"], combination_rows(records), args.format)

    sep = "\n\n"
    blocks = [
        f"Data coverage over {len(records)} {unit} (from {len(proteins)} proteins)",
        "Per-trait coverage:",
        per_trait,
        "Coverage by combination (spectra = ex_max & em_max):",
        combos,
    ]
    text = sep.join(blocks)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"Wrote coverage report -> {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
