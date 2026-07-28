# esm2_fp_design

ESM-2–guided design of **fluorescent proteins (FPs)** conditioned on their photophysical
properties. The project has three parts:

- **[`fpbase-extractor/`](fpbase-extractor)** — pulls FP sequences, phenotypes, and full
  excitation/emission spectral curves from [FPbase](https://www.fpbase.org).
- **[`dataset_pipeline/`](dataset_pipeline)** — curates the raw export into the peak,
  brightness, and pKa training sets, and builds leakage-safe train/val/test splits.
- **[`esm2_design/`](esm2_design)** — designs sequences toward target properties with ESM-2
  and evaluates them against an independent oracle.

## Environment (one combined conda env)

From this directory (the project root, so the editable `fpbase-extractor` install resolves):

```bash
conda env create -f environment.yml
conda activate esm2-fp-design
```

Apple-Silicon friendly — uses the Mac GPU via **MPS** (no CUDA on macOS); set
`PYTORCH_ENABLE_MPS_FALLBACK=1` for any op MPS doesn't support. The ESM-2 650M weights (~650 MB)
download to the torch cache on first use.

## Pipeline

```
fpbase-extractor (fpbase-extract)
        └─ fpbase_output/fpbase_proteins.json    raw sequences + phenotypes
dataset_pipeline/build_dataset.py
        └─ data/<trait>/curated/                 curated peak / brightness / pKa sets
dataset_pipeline/make_dual_split.py
        └─ dual_splits.csv                       coordinated surrogate & oracle splits
esm2_design/
        └─ property-guided design  +  independent-oracle evaluation
```

See each subfolder's `README.md` for details.

The earlier `(sequence, spectrum)` lineage — `design/build_fpbase_dataset.py` feeding
`fpbase-extractor/processed_data/ESM-spectrum/` — has been archived to
`fpbase-extractor/archive/esm_spectrum/`.
