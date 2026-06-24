#!/usr/bin/env python3
"""Read an FPbase JSON export and render a table of protein phenotypes.

Reads the `fpbase_proteins.json` produced by `fpbase-extract` and emits a table
with one row per (protein, state) — since a single protein may have several
photophysical states. Prints an aligned text table to stdout by default, or
writes Markdown / TSV to a file with `--out`.

Dependency-free (standard library only).

Examples:
    python3 phenotype_table.py fpbase_output/fpbase_proteins.json
    python3 phenotype_table.py data.json --sort em_max --limit 20
    python3 phenotype_table.py data.json --out phenotypes.md --format md
"""

import argparse
import json
import os
import sys

# (column header, protein-key OR ("state", state-key)) in display order.
COLUMNS = [
    ("name", ("protein", "name")),
    ("organism", ("protein", "parent_organism")),
    ("oligomerization", ("protein", "oligomerization")),
    ("switch_type", ("protein", "switch_type_label")),
    ("state", ("state", "name")),
    ("ex_max", ("state", "ex_max")),
    ("em_max", ("state", "em_max")),
    ("ext_coeff", ("state", "ext_coeff")),
    ("qy", ("state", "qy")),
    ("brightness", ("state", "brightness")),
    ("pka", ("state", "pka")),
    ("maturation", ("state", "maturation")),
    ("lifetime", ("state", "lifetime")),
]

HEADERS = [c[0] for c in COLUMNS]


def find_default_json():
    """Look for the export in common locations."""
    candidates = [
        "fpbase_proteins.json",
        os.path.join("fpbase_output", "fpbase_proteins.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def build_rows(proteins):
    """Flatten proteins into one row per (protein, state)."""
    rows = []
    for p in proteins:
        states = p.get("states") or [None]
        for state in states:
            row = []
            for _, source in COLUMNS:
                kind, key = source
                if kind == "protein":
                    value = p.get(key)
                else:  # state
                    value = state.get(key) if state else None
                row.append(value)
            rows.append(row)
    return rows


def sort_rows(rows, sort_key):
    """Sort rows by a column name; None values sink to the bottom."""
    if sort_key not in HEADERS:
        raise SystemExit(
            f"error: --sort '{sort_key}' is not a column. Choose from: {', '.join(HEADERS)}"
        )
    idx = HEADERS.index(sort_key)

    def key(row):
        v = row[idx]
        # (is_none, normalized_value) keeps None last for both num and str.
        if v is None:
            return (1, "")
        if isinstance(v, (int, float)):
            return (0, v)
        return (0, str(v).lower())

    return sorted(rows, key=key)


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def render_text(rows):
    cells = [[_fmt(v) for v in row] for row in rows]
    widths = [len(h) for h in HEADERS]
    for row in cells:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    lines = []
    lines.append("  ".join(h.ljust(widths[i]) for i, h in enumerate(HEADERS)))
    lines.append("  ".join("-" * widths[i] for i in range(len(HEADERS))))
    for row in cells:
        lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


def render_markdown(rows):
    lines = ["| " + " | ".join(HEADERS) + " |", "| " + " | ".join("---" for _ in HEADERS) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def render_tsv(rows):
    lines = ["\t".join(HEADERS)]
    for row in rows:
        lines.append("\t".join(_fmt(v) for v in row))
    return "\n".join(lines)


RENDERERS = {"text": render_text, "md": render_markdown, "tsv": render_tsv}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render a phenotype table from an FPbase JSON export.")
    parser.add_argument("json_path", nargs="?", help="Path to fpbase_proteins.json (auto-detected if omitted)")
    parser.add_argument("--sort", help=f"Sort by column: {', '.join(HEADERS)}")
    parser.add_argument("--limit", type=int, help="Show only the first N rows")
    parser.add_argument("--format", choices=list(RENDERERS), default="text", help="Output format (default: text)")
    parser.add_argument("--out", help="Write to this file instead of stdout")
    args = parser.parse_args(argv)

    path = args.json_path or find_default_json()
    if not path:
        parser.error(
            "no JSON file given and none found. Run `fpbase-extract` first, or pass the path explicitly."
        )
    if not os.path.isfile(path):
        parser.error(f"file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        proteins = json.load(fh)

    rows = build_rows(proteins)
    if args.sort:
        rows = sort_rows(rows, args.sort)
    if args.limit is not None:
        rows = rows[: args.limit]

    table = RENDERERS[args.format](rows)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(table + "\n")
        print(f"Wrote {len(rows)} rows -> {args.out}", file=sys.stderr)
    else:
        print(table)
        print(f"\n{len(rows)} rows from {len(proteins)} proteins.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
