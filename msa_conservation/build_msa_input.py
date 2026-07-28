#!/usr/bin/env python
"""Assemble the MSA input: every unique sequence in the curated FP dataset.

The input is the union of the three curated trait sets (peak 758, brightness 533,
pKa 368 -> 763 unique sequences). That union, not the raw FPbase export, is the
right universe for a family alignment: ``dataset_pipeline`` has already dropped
the bilin/phytochrome near-infrared class, which does not share the GFP fold and
would only inject junk columns (see dataset_pipeline/README.md, "Note on the
biliverdin / phytochrome class").

Writes ``data/fp_all.fasta`` (headers ``>msa_id|slug``) plus ``data/fp_all_meta.csv``
carrying the per-sequence metadata the conservation analysis reports on.
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
CUR = HERE.parent / "dataset_pipeline" / "data"
TRAITS = {"peak": "peaks_assignments.csv",
          "brightness": "brightness_assignments.csv",
          "pka": "pka_assignments.csv"}
META_COLS = ["slug", "name", "parent_organism", "switch_type", "oligomerization",
             "ex_max", "em_max", "ref_year"]


def main():
    frames = {t: pd.read_csv(CUR / t / "curated" / f) for t, f in TRAITS.items()}

    # peak is the largest set, so take it as the base and append what the other
    # two traits add (2 from brightness, 5 from pKa).
    rows, seen = [], set()
    for trait in ("peak", "brightness", "pka"):
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
    with open(HERE / "data" / "fp_all.fasta", "w") as fh:
        for r in meta.itertuples():
            fh.write(f">{r.msa_id}|{r.slug}\n{r.seq}\n")
    meta.to_csv(HERE / "data" / "fp_all_meta.csv", index=False)

    print(f"{len(meta)} unique sequences  "
          f"(peak {meta.in_peak.sum()}, brightness {meta.in_brightness.sum()}, pka {meta.in_pka.sum()})")
    print(f"length: min {meta.seq_len.min()}  median {meta.seq_len.median():.0f}  max {meta.seq_len.max()}")
    print(f"organisms: {meta.parent_organism.nunique()}")
    print("-> data/fp_all.fasta, data/fp_all_meta.csv")


if __name__ == "__main__":
    main()
