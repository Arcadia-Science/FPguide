#!/usr/bin/env python3
"""Transform the Sarkisyan et al. (2016) avGFP DMS genotype table into a CSV
of full mutated sequences with brightness values.

Source dataset
--------------
Sarkisyan, K. et al. "Local fitness landscape of the green fluorescent
protein." Nature 533, 397-401 (2016).
https://figshare.com/articles/dataset/Local_fitness_landscape_of_the_green_fluorescent_protein/3102154

Input file: amino_acid_genotypes_to_brightness.tsv
    Columns: aaMutations  uniqueBarcodes  medianBrightness  std
    - aaMutations: colon-separated mutations, each token formatted as
      S<wt><pos><mut>, e.g. "SA108D:SN144D". The leading "S" is a constant
      prefix, <wt> is the parent residue, <pos> is the (1-based) position in
      the dataset's own numbering, <mut> is the mutant residue ("*" = stop).
      An empty aaMutations field is the parent (wild-type) genotype.

Variants containing any premature stop codon ("*") are REMOVED: the truncated
protein cannot fold a chromophore, and their brightness calls in this dataset are
dominated by single-barcode artifacts, so every emitted sequence is full length.
    - medianBrightness: median *log10* fluorescence of the variant. This column
      is ALREADY on a log10 scale (dark variants pile up at log10(20) ~= 1.301),
      so it is emitted unchanged as `logMedianBrightness`.

Output CSV columns (default):
    scaffold             - GFP scaffold name (always "avGFP" here; present so this table
                           shares a schema with the multi-scaffold orthologue dataset)
    mutatedSequence      - full-length mutated protein sequence
    logMedianBrightness  - the source medianBrightness value, i.e. log10(fluorescence)
    linearBrightness     - fluorescence on a linear scale, = 10 ** logMedianBrightness
    brightnessClass      - "bright" / "dim", split at a data-driven per-scaffold threshold
                           (KDE antimode of the log-brightness distribution; see
                           brightness_threshold.py)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from brightness_threshold import classify, log_brightness_threshold

SCAFFOLD = "avGFP"

# Canonical avGFP amino-acid sequence (UniProt P42212), 238 residues.
# Used ONLY to fill positions that never appear as a mutation in the data.
P42212 = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDF"
    "FKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFK"
    "IRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

# The dataset numbers residues 1..235, corresponding to canonical positions 3..237.
SEQ_LEN = 235
PARENT_OFFSET = 2  # dataset position p == canonical position p + PARENT_OFFSET

MUT_RE = re.compile(r"^S([A-Z])(\d+)([A-Z*])$")


def build_parent(rows: list[list[str]]) -> str:
    """Reconstruct the parent sequence from the data.

    For every position, the wild-type residue is taken from the mutation tokens
    (which are internally consistent). Positions that never get mutated are
    filled from the canonical P42212 sequence.
    """
    canonical_slice = P42212[PARENT_OFFSET : PARENT_OFFSET + SEQ_LEN]
    parent = list(canonical_slice)

    seen_wt: dict[int, str] = {}
    for row in rows:
        muts = row[0].strip()
        if not muts:
            continue
        for token in muts.split(":"):
            m = MUT_RE.match(token)
            if not m:
                raise ValueError(f"Unparseable mutation token: {token!r}")
            wt, pos, _mut = m.group(1), int(m.group(2)), m.group(3)
            prev = seen_wt.get(pos)
            if prev is not None and prev != wt:
                raise ValueError(
                    f"Inconsistent WT residue at position {pos}: {prev} vs {wt}"
                )
            seen_wt[pos] = wt

    for pos, wt in seen_wt.items():
        parent[pos - 1] = wt

    # Framing guard: positions never seen in the data are filled from the
    # canonical P42212 slice, which is only valid if the dataset numbering is
    # aligned to P42212 by a constant offset. Confirm that alignment using the
    # data-covered positions before trusting any filled residue.
    #
    # A residue is allowed to differ from P42212 only when the data explicitly
    # reports a different WT there (i.e. a genuine parent mutation such as F64L),
    # never at a filled (never-mutated) position.
    for pos in range(1, SEQ_LEN + 1):
        if pos in seen_wt:
            continue
        canonical = P42212[pos - 1 + PARENT_OFFSET]
        # require at least one immediate neighbour to be data-covered and to
        # agree with P42212 at the same offset, proving there is no frameshift.
        neighbours = [n for n in (pos - 1, pos + 1) if n in seen_wt]
        if not neighbours:
            continue
        if not any(seen_wt[n] == P42212[n - 1 + PARENT_OFFSET] for n in neighbours):
            raise ValueError(
                f"Cannot safely fill un-mutated position {pos}: dataset numbering "
                f"does not align to P42212 near this position (possible frameshift)."
            )
        parent[pos - 1] = canonical

    return "".join(parent)


def apply_mutations(parent: str, mut_field: str):
    """Return the mutated sequence, or None if the variant contains a stop codon.

    Variants carrying any premature stop ("*") are dropped: a truncated protein
    cannot fold the chromophore, and the dataset's stop-variant brightness calls
    are dominated by single-barcode artifacts (see project notes), so they carry
    no reliable sequence->brightness signal.
    """
    seq = list(parent)
    for token in mut_field.split(":"):
        m = MUT_RE.match(token)
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        if parent[pos - 1] != wt:
            raise ValueError(
                f"WT mismatch in {token!r}: parent has {parent[pos - 1]} at {pos}"
            )
        if mut == "*":
            return None
        seq[pos - 1] = mut
    return "".join(seq)


def main() -> int:
    default_in = Path(__file__).resolve().parent / "DMS_data" / "amino_acid_genotypes_to_brightness.tsv"
    default_out = Path(__file__).resolve().parent / "DMS_data" / "avgfp_dms_sequences.csv"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=default_in, help="input TSV (default: %(default)s)")
    ap.add_argument("-o", "--output", type=Path, default=default_out, help="output CSV (default: %(default)s)")
    ap.add_argument("--keep-extra", action="store_true",
                    help="also include aaMutations, uniqueBarcodes and std columns")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    with args.input.open(newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        rows = [r for r in reader if r]

    expected = ["aaMutations", "uniqueBarcodes", "medianBrightness", "std"]
    if header[: len(expected)] != expected:
        print(f"WARNING: unexpected header: {header}", file=sys.stderr)

    parent = build_parent(rows)
    print(f"Parent sequence ({len(parent)} aa): {parent}")

    out_cols = ["scaffold", "mutatedSequence", "logMedianBrightness", "linearBrightness",
                "brightnessClass"]
    if args.keep_extra:
        out_cols += ["aaMutations", "uniqueBarcodes", "std"]

    n_dropped_stop = n_skipped_brightness = 0
    records: list[dict] = []

    for row in rows:
        mut_field = row[0].strip()
        brightness_raw = row[2].strip() if len(row) > 2 else ""

        # medianBrightness is already log10(fluorescence); keep it as-is.
        try:
            log_brightness = float(brightness_raw)
        except ValueError:
            n_skipped_brightness += 1
            continue

        seq = apply_mutations(parent, mut_field) if mut_field else parent  # empty -> wild type

        if seq is None:  # variant contains a stop codon -> removed
            n_dropped_stop += 1
            continue

        records.append({
            "scaffold": SCAFFOLD,
            "mutatedSequence": seq,
            "logMedianBrightness": log_brightness,
            "linearBrightness": 10 ** log_brightness,
            "aaMutations": mut_field,
            "uniqueBarcodes": row[1] if len(row) > 1 else "",
            "std": row[3] if len(row) > 3 else "",
        })

    # data-driven bright/dim split from this scaffold's log-brightness distribution
    logs = [r["logMedianBrightness"] for r in records]
    thr, info = log_brightness_threshold(logs)
    labels = classify(logs, thr)
    n_bright = 0
    for r, lab in zip(records, labels):
        r["brightnessClass"] = lab
        n_bright += lab == "bright"

    with args.output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(out_cols)
        for r in records:
            writer.writerow([r[c] for c in out_cols])

    n_written = len(records)
    print(f"Wrote {n_written} variants to {args.output}")
    print(f"[{SCAFFOLD}] bright/dim threshold (log10) = {thr:.4f} via {info['method']} "
          f"(modes {info.get('modes')}) -> bright {n_bright} ({100*n_bright/n_written:.1f}%), "
          f"dim {n_written - n_bright} ({100*(n_written - n_bright)/n_written:.1f}%)")
    if n_dropped_stop:
        print(f"Removed {n_dropped_stop} variants containing a stop codon")
    if n_skipped_brightness:
        print(f"Skipped {n_skipped_brightness} rows with missing/invalid medianBrightness")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
