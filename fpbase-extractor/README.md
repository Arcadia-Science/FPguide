# fpbase-extractor

Extract **protein sequences** and **all phenotypical information** for every
fluorescent protein in [FPbase](https://www.fpbase.org) into ready-to-use
FASTA, CSV, and JSON files.

The entire database (~1,000 proteins) is pulled in a **single GraphQL request**,
so a full extraction takes about a second. No third-party dependencies — just
the Python standard library.

## What it extracts

For each protein:

| Category | Fields |
|----------|--------|
| Identity | name, slug, aliases, parent organism |
| Sequence | amino-acid sequence (`seq`) |
| Cross-refs | GenBank, UniProt, IPG ID, PDB structures |
| Oligomerization | `agg` code + readable label (monomer, dimer, …) |
| Switching | `switch_type` code + label (basic, photoswitchable, …) |
| Cofactor | required cofactor, if any |
| Reference | primary publication DOI, year, journal, title |

Photophysical phenotype is stored **per state** (a protein can have several —
e.g. the on/off forms of a photoswitchable protein):

> excitation max, emission max, extinction coefficient, quantum yield,
> brightness, pKa, maturation time, fluorescence lifetime, two-photon
> excitation max / peak GM / QY, dark-state flag, and display colors (ex/em hex).

With `--spectra`, it also pulls the **full excitation/emission spectral curves**
(the complete wavelength-vs-intensity traces, ~227k data points across 455
proteins), not just the peak maxima.

## Install

This package is installed (editable) as part of the project-wide conda env. From the
**project root** (`esm2_fp_design/`):

```bash
conda env create -f environment.yml   # installs `-e ./fpbase-extractor`
conda activate esm2-fp-design
```

The extractor itself is pure standard library, so you can also run it without installing:

```bash
python3 -m fpbase_extractor.cli --outdir fpbase_output
```

## Usage

```bash
# Everything, all three formats, into ./fpbase_output/
fpbase-extract

# Only sequences for the mScarlet family, as FASTA
fpbase-extract --filter mScarlet --formats fasta --outdir scarlet/

# CSV only, into a custom directory
fpbase-extract --formats csv --outdir data/

# Include the full excitation/emission spectral curves
fpbase-extract --spectra --outdir data/
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --outdir` | `fpbase_output` | Output directory |
| `-f, --formats` | `csv,json,fasta` | Comma-separated subset of `csv,json,fasta` |
| `--source` | `auto` | `graphql` (richest), `rest` (fallback), or `auto` |
| `--spectra` | off | Also fetch full excitation/emission spectral curves |
| `--filter` | – | Case-insensitive substring matched on name/aliases |
| `--basename` | `fpbase_proteins` | Output file basename |
| `--timeout` | `60` | HTTP timeout (seconds) |

## Output files

- **`fpbase_proteins.fasta`** — one record per protein with a sequence.
  Headers carry a phenotype summary:
  `>egfp | EGFP | ex=488.0 | em=507.0 | olig=weak dimer | org=Aequorea victoria`
- **`fpbase_proteins.csv`** — flat table, **one row per (protein, state)**.
  Protein-level columns repeat across a protein's state rows; stateless
  proteins get one row with empty phenotype columns. List fields (aliases,
  PDB) are `; `-joined.
- **`fpbase_proteins.json`** — full normalized records with nested `states`,
  best for programmatic use.

With `--spectra` (only for proteins that have measured curves):

- **`fpbase_spectra_long.csv`** — tidy long format, **one row per
  (protein, spectrum, wavelength)**: `slug, name, state, spectrum_type,
  wavelength, intensity`. Loads straight into pandas/R/ggplot for plotting.
- **`fpbase_spectra.json`** — nested per protein: each spectrum has its
  `spectrum_type` (excitation/emission), `max`, and `data` array of
  `[wavelength, intensity]` pairs.

The spectra exports are ~20 MB and nothing downstream reads them, so they are no
longer kept in `fpbase_output/`. Rerun with `--spectra` if you need them.

## Downstream

`fpbase_output/fpbase_proteins.json` is the input to
[`dataset_pipeline/build_dataset.py`](../dataset_pipeline), which curates it into the
peak / brightness / pKa training sets. That is the only consumer of this folder.

## Archived material

Exploratory work that is not part of the dataset-building path lives in
[`archive/`](archive) (untracked): the mutual-information study, the ESM-2 embedding map
and its caches, the older `(sequence, spectrum)` dataset lineage, and the phenotype
coverage scripts. See `archive/README.md` for what each group was and how to regenerate it.

## Use as a library

```python
from fpbase_extractor import fetch_proteins, normalize_all, write_outputs

raw, source = fetch_proteins(source="graphql")
proteins = normalize_all(raw)

# proteins[0] -> {"name", "seq", "parent_organism", "oligomerization",
#                 "states": [{"ex_max", "em_max", "qy", ...}], ...}

write_outputs(proteins, "out/", formats=["csv", "json", "fasta"])
```

## Data sources

- **GraphQL** (`https://www.fpbase.org/graphql/`) — default; one request returns
  sequences, all states, organism, and reference details.
- **REST** (`https://www.fpbase.org/api/proteins/?format=json`) — fallback;
  a subset of fields (no aliases, organism, or reference metadata).

## Tests

`pytest` ships with the `esm2-fp-design` env:

```bash
pytest
```

Tests cover normalization of both API shapes and all output writers offline
(no network required).

## Notes

- Data is © FPbase contributors and released under the database's terms; please
  cite FPbase (Lambert TJ, *Nat Methods* 2019) when using the data.
- Field coverage reflects what FPbase has curated — many entries have `null`
  values for properties that were never measured.
```
