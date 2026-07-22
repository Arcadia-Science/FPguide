#!/usr/bin/env python3
"""Transform the orthologous-GFP DMS table (amacGFP / cgreGFP / ppluGFP) into a CSV of
full mutated sequences with brightness values and a per-scaffold bright/dim label.

Source dataset
--------------
Gonzalez Somermeyer, L. et al. "Heterogeneity of the GFP fitness landscape and
data-driven protein design." eLife 11:e75842 (2022). https://doi.org/10.7554/eLife.75842
Data: github.com/aequorea238/Orthologous_GFP_Fitness_Peaks (final_datasets).

Input file: amacGFP_cgreGFP_ppluGFP2__final_aminoacid_genotypes_to_brightness.csv
    Columns: aa_genotype_pseudo, aa_genotype_native, gene, total_cell_count,
             n_replicates, replicates_mean_brightness
    - gene: one of amacGFP / cgreGFP / ppluGFP (each row's mutations are relative to
      that scaffold's wild type).
    - aa_genotype_native: colon-separated mutations in the scaffold's OWN residue
      numbering, each token <wt><pos><mut>. "wt" (or empty) is the wild-type genotype.
      The alphabet includes the 20 amino acids plus "*" (stop) and "." (rare
      stop/ambiguous call); the terminal stop occupies the last position.
    - replicates_mean_brightness: mean fluorescence on a LINEAR scale (unlike the
      Sarkisyan avGFP table, which is already log10). We emit log10 of it.

Wild-type reconstruction: every protein position is observed as the wild-type residue in
some mutation token, so each scaffold's parent sequence is reconstructed directly from the
data (no external reference needed) and cross-checked for internal consistency.

Variants that are not clean full-length point-substitution proteins are REMOVED: any token
whose wild-type or mutant residue is non-standard ("*"/"."), or that touches the terminal
stop position (C-terminal read-through). A truncated/extended protein cannot fold the
chromophore, mirroring the stop-drop policy of transform_dms.py.

Output CSV columns (default), all scaffolds concatenated:
    scaffold             - amacGFP / cgreGFP / ppluGFP
    mutatedSequence      - full-length mutated protein sequence
    logBrightness        - log10(replicates_mean_brightness)
    linearBrightness     - replicates_mean_brightness (linear)
    brightnessClass      - "bright" / "dim", split at a data-driven PER-SCAFFOLD threshold
                           (KDE antimode of that scaffold's log-brightness; see
                           brightness_threshold.py)
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from brightness_threshold import classify, log_brightness_threshold

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
TOKEN_RE = re.compile(r"^(.)(\d+)(.)$")


def parse_tokens(genotype: str):
    """Yield (wt, pos, mut) for a native genotype string ('wt'/'' -> no tokens)."""
    g = genotype.strip()
    if g in ("", "wt"):
        return
    for tok in g.split(":"):
        m = TOKEN_RE.match(tok)
        if not m:
            raise ValueError(f"Unparseable mutation token: {tok!r}")
        yield m.group(1), int(m.group(2)), m.group(3)


def build_parent(rows: list[dict]) -> tuple[str, int]:
    """Reconstruct one scaffold's wild-type sequence from its mutation tokens.

    The wild-type residue at a position is read only from CLEAN substitution tokens (both
    WT and mutant are standard amino acids). Tokens involving "*"/"." carry unreliable WT
    letters -- the "." ambiguous call produces spurious WT disagreements (e.g. a stray
    "E30." against the true "G30x"), so they are ignored here. The protein length is the
    largest clean-sub position; the terminal stop (WT '*') is excluded. Every position
    1..L must be observed among clean substitutions.
    """
    wt_by_pos: dict[int, str] = {}
    for r in rows:
        for wt, pos, mut in parse_tokens(r["aa_genotype_native"]):
            if wt not in STD_AA or mut not in STD_AA:  # skip '*'/'.' (stop / ambiguous)
                continue
            prev = wt_by_pos.get(pos)
            if prev is not None and prev != wt:
                raise ValueError(f"Inconsistent WT at position {pos}: {prev} vs {wt}")
            wt_by_pos[pos] = wt
    if not wt_by_pos:
        raise ValueError("no parseable mutations; cannot reconstruct parent")
    length = max(wt_by_pos)
    missing = [p for p in range(1, length + 1) if p not in wt_by_pos]
    if missing:
        raise ValueError(f"cannot reconstruct parent: positions never observed as WT: {missing}")
    return "".join(wt_by_pos[p] for p in range(1, length + 1)), length


def apply_mutations(parent: str, genotype: str, length: int):
    """Return the mutated sequence, or None if the variant is not a clean full-length
    point-substitution protein (premature stop, '.', or terminal read-through)."""
    seq = list(parent)
    for wt, pos, mut in parse_tokens(genotype):
        if pos > length or wt not in STD_AA or mut not in STD_AA:
            return None                                # stop / '.' / read-through -> drop
        if parent[pos - 1] != wt:
            raise ValueError(f"WT mismatch at {pos}: parent has {parent[pos - 1]}, token says {wt}")
        seq[pos - 1] = mut
    return "".join(seq)


def main() -> int:
    here = Path(__file__).resolve().parent
    default_in = here / "DMS_data" / "amacGFP_cgreGFP_ppluGFP2__final_aminoacid_genotypes_to_brightness.csv"
    default_out = here / "DMS_data" / "ortho_gfp_dms_sequences.csv"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=default_in, help="input CSV (default: %(default)s)")
    ap.add_argument("-o", "--output", type=Path, default=default_out, help="output CSV (default: %(default)s)")
    ap.add_argument("--keep-extra", action="store_true",
                    help="also include aa_genotype_native, n_replicates, total_cell_count")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    with args.input.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    by_gene: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_gene[r["gene"]].append(r)

    out_cols = ["scaffold", "mutatedSequence", "logBrightness", "linearBrightness", "brightnessClass"]
    if args.keep_extra:
        out_cols += ["aaMutationsNative", "nReplicates", "totalCellCount"]

    records: list[dict] = []
    total_stop = total_bad_brightness = 0

    for gene in sorted(by_gene):
        grows = by_gene[gene]
        parent, length = build_parent(grows)
        print(f"[{gene}] parent ({length} aa): {parent}")

        gene_recs: list[dict] = []
        n_stop = n_bad = 0
        for r in grows:
            raw = r["replicates_mean_brightness"].strip()
            try:
                linear = float(raw)
            except ValueError:
                n_bad += 1
                continue
            if not (linear > 0) or not math.isfinite(linear):
                n_bad += 1
                continue

            seq = apply_mutations(parent, r["aa_genotype_native"], length)
            if seq is None:                            # truncated / extended / ambiguous
                n_stop += 1
                continue

            gene_recs.append({
                "scaffold": gene,
                "mutatedSequence": seq,
                "logBrightness": math.log10(linear),
                "linearBrightness": linear,
                "aaMutationsNative": r["aa_genotype_native"].strip(),
                "nReplicates": r.get("n_replicates", ""),
                "totalCellCount": r.get("total_cell_count", ""),
            })

        # per-scaffold bright/dim split from this scaffold's log-brightness distribution
        logs = [rec["logBrightness"] for rec in gene_recs]
        thr, info = log_brightness_threshold(logs)
        labels = classify(logs, thr)
        n_bright = 0
        for rec, lab in zip(gene_recs, labels):
            rec["brightnessClass"] = lab
            n_bright += lab == "bright"
        n = len(gene_recs)
        print(f"[{gene}] kept {n} | bright/dim threshold (log10) = {thr:.4f} via {info['method']} "
              f"(modes {info.get('modes')}) -> bright {n_bright} ({100*n_bright/n:.1f}%), "
              f"dim {n - n_bright} ({100*(n - n_bright)/n:.1f}%)")
        if n_stop:
            print(f"[{gene}] removed {n_stop} truncated/extended/ambiguous variants")
        if n_bad:
            print(f"[{gene}] skipped {n_bad} rows with missing/invalid brightness")
        records.extend(gene_recs)
        total_stop += n_stop
        total_bad_brightness += n_bad

    with args.output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(out_cols)
        for rec in records:
            writer.writerow([rec[c] for c in out_cols])

    print(f"\nWrote {len(records)} variants across {len(by_gene)} scaffolds to {args.output}")
    if total_stop:
        print(f"Removed {total_stop} truncated/extended/ambiguous variants total")
    if total_bad_brightness:
        print(f"Skipped {total_bad_brightness} rows with invalid brightness total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
