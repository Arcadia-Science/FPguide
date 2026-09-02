#!/usr/bin/env python
"""Assemble the MSA input: every unique sequence in the curated FP dataset.

Two variants, because one of the three inputs is archived (see BUILDABILITY below):

  ``--union`` (default)  the published input: the union of all three curated trait sets
                         (peak 758, brightness 533, pKa 368 -> **763** unique sequences)
                         -> ``data/fp_all.fasta`` + ``data/fp_all_meta.csv``
  ``--peak-only``        the peak curated set alone (**758** sequences)
                         -> ``data/fp_peak_only.fasta`` + ``data/fp_peak_only_meta.csv``

The union, not the raw FPbase export, is the right universe for a family alignment:
``dataset_pipeline`` has already dropped the bilin/phytochrome near-infrared class, which does not
share the GFP fold and would only inject junk columns (see dataset_pipeline/README.md, "Note on the
biliverdin / phytochrome class").

The 5 sequences the union adds over peak alone are proteins the *peak* curation deliberately
dropped -- the analyte sensors CAR-GECO1, mKeima, pHluorin4 and pHmScarlet (a single (ex, em) label
is ill-defined when the peak moves with analyte) and the unresolvable multi-state PSLSSmKate -- but
which carry a brightness or pKa measurement. A family alignment wants fold coverage, not label
coverage, so the union readmits them as sequences even though they have no peak label.

BUILDABILITY. brightness and pKa were archived after this alignment was built
(dataset_pipeline/README.md, "Archived: brightness & pKa"), so on a fresh clone
``data/<trait>/curated/`` exists for peak only and the 763-sequence union CANNOT be rebuilt without
regenerating them first. This script therefore refuses to guess: ``--union`` fails loudly, naming
the exact ``build_dataset.py`` commands, rather than silently emitting 758 sequences under the
763-sequence filenames. Note that you may not need to run this at all --
``data/fp_all.fasta``, ``data/fp_all_meta.csv`` and ``data/fp_all.aln.fasta`` are all tracked in git.

The two variants write DIFFERENT filenames on purpose. Everything downstream (``conservation.py``,
``esm_profiles.py``, ``esm_vs_family.py``, ``compare_design_windows.py``, and
``design-campaign-EGFP/msa-guided/build_msa_pssm.py``) reads ``fp_all*`` by name and pairs
alignment rows to metadata rows by ``msa_id``, so a 758-row file under those names would silently
invalidate every published number. ``--peak-only`` cannot overwrite them.
"""
import argparse
import sys
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
CUR = HERE.parent / "dataset_pipeline" / "data"
TRAITS = {"peak": "peaks_assignments.csv",
          "brightness": "brightness_assignments.csv",
          "pka": "pka_assignments.csv"}
META_COLS = ["slug", "name", "parent_organism", "switch_type", "oligomerization",
             "ex_max", "em_max", "ref_year"]

# variant -> (traits consumed, output stem, expected sequence count)
VARIANTS = {"union": (("peak", "brightness", "pka"), "fp_all", 763),
            "peak-only": (("peak",), "fp_peak_only", 758)}


def trait_csv(trait):
    return CUR / trait / "curated" / TRAITS[trait]


def require(traits, variant):
    """Hard-exit with the rebuild path if any trait set the variant needs is absent."""
    missing = [t for t in traits if not trait_csv(t).exists()]
    if not missing:
        return
    lines = [f"missing curated trait set(s) needed for --{variant}: {', '.join(missing)}", ""]
    for t in missing:
        lines.append(f"    expected  {trait_csv(t)}")
    lines += [
        "",
        'These sets were archived after this alignment was built (dataset_pipeline/README.md,',
        '"Archived: brightness & pKa"). Three ways forward:',
        "",
        "  1. You probably do not need to run this script. The 763-sequence input and its",
        "     alignment are tracked in git:",
        "         data/fp_all.fasta  data/fp_all_meta.csv  data/fp_all.aln.fasta",
        "",
        "  2. Rebuild the archived sets from the tracked FPbase export, then re-run this:",
        "         cd ../dataset_pipeline",
    ]
    lines += [f"         python build_dataset.py --target {t}" for t in missing]
    lines += [
        "         cd ../msa_conservation && python build_msa_input.py",
        "",
        "  3. Build the 758-sequence peak-only variant instead -- NOT the published input,",
        "     and it writes its own filenames so it cannot clobber fp_all.*:",
        "         python build_msa_input.py --peak-only",
    ]
    raise SystemExit("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--union", action="store_true",
                   help="all three curated trait sets, 763 seqs -> fp_all.* (default; published)")
    g.add_argument("--peak-only", action="store_true",
                   help="peak curated set alone, 758 seqs -> fp_peak_only.*")
    a = ap.parse_args()

    variant = "peak-only" if a.peak_only else "union"
    traits, stem, n_expect = VARIANTS[variant]
    require(traits, variant)

    frames = {t: pd.read_csv(trait_csv(t)) for t in traits}

    # peak is the largest set, so take it as the base and append what the other
    # two traits add (2 from brightness, 3 from pKa).
    rows, seen = [], set()
    for trait in traits:
        df = frames[trait]
        for r in df.to_dict("records"):
            if r["seq"] in seen:
                continue
            seen.add(r["seq"])
            rows.append({**{c: r.get(c) for c in META_COLS},
                         "seq": r["seq"], "seq_len": len(r["seq"]), "source_trait": trait})

    meta = pd.DataFrame(rows).sort_values("slug").reset_index(drop=True)
    meta.insert(0, "msa_id", meta.index)
    for trait, df in frames.items():
        meta[f"in_{trait}"] = meta.seq.isin(set(df.seq))

    (HERE / "data").mkdir(exist_ok=True)
    fasta, meta_csv = HERE / "data" / f"{stem}.fasta", HERE / "data" / f"{stem}_meta.csv"
    with open(fasta, "w") as fh:
        for r in meta.itertuples():
            fh.write(f">{r.msa_id}|{r.slug}\n{r.seq}\n")
    meta.to_csv(meta_csv, index=False)

    counts = "  ".join(f"{t} {int(meta[f'in_{t}'].sum())}" for t in traits)
    print(f"[{variant}] {len(meta)} unique sequences  ({counts})")
    print(f"length: min {meta.seq_len.min()}  median {meta.seq_len.median():.0f}  max {meta.seq_len.max()}")
    print(f"organisms: {meta.parent_organism.nunique()}")
    print(f"-> data/{stem}.fasta, data/{stem}_meta.csv")

    if len(meta) != n_expect:
        print(f"\nWARNING: expected {n_expect} sequences for --{variant}, got {len(meta)}. "
              f"The curated inputs have changed since this analysis was published; "
              f"downstream counts will not match the write-up.", file=sys.stderr)

    if variant == "peak-only":
        print("\nNOTE: this is NOT the published input. It is missing the 5 sequences only",
              "\nbrightness/pKa readmit (CAR-GECO1, mKeima, pHluorin4, pHmScarlet, PSLSSmKate),",
              "\nso every conservation count will differ from the write-up. run_msa.sh aligns",
              f"\nfp_all.fasta, not this file; to align this one:",
              f"\n    mafft --maxiterate 1000 --thread 1 --randomseed 0 \\",
              f"\n          data/{stem}.fasta > data/{stem}.aln.fasta",
              "\nand point the downstream scripts at it yourself -- they default to fp_all.*.")


if __name__ == "__main__":
    main()
