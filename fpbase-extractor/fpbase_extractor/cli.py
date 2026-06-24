"""Command-line interface for the FPbase extractor."""

import argparse
import sys

from . import __version__
from .client import FPbaseError, fetch_proteins, fetch_spectra
from .extract import normalize_all, write_outputs
from .spectra import normalize_spectra, write_spectra_outputs


def _parse_formats(value):
    formats = [f.strip().lower() for f in value.split(",") if f.strip()]
    allowed = {"csv", "json", "fasta"}
    bad = set(formats) - allowed
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown format(s): {', '.join(sorted(bad))}. Allowed: csv, json, fasta"
        )
    return formats


def build_parser():
    parser = argparse.ArgumentParser(
        prog="fpbase-extract",
        description="Extract protein sequences and phenotype data from FPbase "
        "(https://www.fpbase.org).",
    )
    parser.add_argument(
        "-o", "--outdir", default="fpbase_output", help="Output directory (default: fpbase_output)"
    )
    parser.add_argument(
        "-f",
        "--formats",
        type=_parse_formats,
        default=["csv", "json", "fasta"],
        help="Comma-separated output formats: csv,json,fasta (default: all)",
    )
    parser.add_argument(
        "--source",
        choices=["graphql", "rest", "auto"],
        default="auto",
        help="Data source. graphql=richest (default via auto), rest=fallback, "
        "auto=graphql then rest on failure.",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Case-insensitive substring; keep only proteins whose name or "
        "aliases match.",
    )
    parser.add_argument(
        "--spectra",
        action="store_true",
        help="Also fetch full excitation/emission spectra (curves) and write "
        "fpbase_spectra.json + fpbase_spectra_long.csv.",
    )
    parser.add_argument(
        "--basename", default="fpbase_proteins", help="Output file basename (default: fpbase_proteins)"
    )
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds (default: 60)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _matches(protein, needle):
    needle = needle.lower()
    if needle in (protein.get("name") or "").lower():
        return True
    return any(needle in str(a).lower() for a in protein.get("aliases", []))


def main(argv=None):
    args = build_parser().parse_args(argv)

    print(f"Fetching proteins from FPbase (source={args.source})...", file=sys.stderr)
    try:
        raw, used = fetch_proteins(source=args.source, timeout=args.timeout)
    except FPbaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    proteins = normalize_all(raw)
    print(f"Fetched {len(proteins)} proteins via {used}.", file=sys.stderr)

    if args.filter:
        proteins = [p for p in proteins if _matches(p, args.filter)]
        print(f"{len(proteins)} proteins match filter {args.filter!r}.", file=sys.stderr)

    if not proteins:
        print("No proteins to write.", file=sys.stderr)
        return 0

    results = write_outputs(proteins, args.outdir, args.formats, basename=args.basename)
    for fmt, (path, count) in results.items():
        print(f"  {fmt:13} -> {path} ({count} records)", file=sys.stderr)

    if args.spectra:
        print("Fetching full excitation/emission spectra...", file=sys.stderr)
        try:
            raw_spectra = fetch_spectra(timeout=max(args.timeout, 120))
        except FPbaseError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        spectra = normalize_spectra(raw_spectra)
        # Only keep spectra for proteins that passed the name/alias filter.
        if args.filter:
            keep = {p["slug"] for p in proteins}
            spectra = [s for s in spectra if s["slug"] in keep]
        print(f"{len(spectra)} proteins have spectra.", file=sys.stderr)
        # Spectra are tabular/structured only: honor json/csv from --formats,
        # default to both when neither was requested (e.g. fasta-only run).
        spectra_formats = [f for f in args.formats if f in ("json", "csv")] or ["json", "csv"]
        sresults = write_spectra_outputs(spectra, args.outdir, spectra_formats)
        for fmt, (path, count) in sresults.items():
            print(f"  {fmt:13} -> {path} ({count} records)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
