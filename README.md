# esm2_fp_design

ESM-2–guided design of **fluorescent proteins (FPs)** conditioned on their excitation/emission
spectra. The project has two parts:

- **[`fpbase-extractor/`](fpbase-extractor)** — pulls FP sequences, phenotypes, and full
  excitation/emission spectral curves from [FPbase](https://www.fpbase.org).
- **[`design/`](design)** — builds the `(sequence, spectrum)` dataset, clusters
  sequences by identity for leakage-safe train/test splits, and designs sequences toward a target
  spectrum with ESM-2.

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
fpbase-extractor (fpbase-extract --spectra)
        └─ fpbase_output/                       raw sequences + ex/em curves
design/build_fpbase_dataset.py
        └─ fpbase-extractor/processed_data/ESM-spectrum/   (sequence, spectrum) dataset
design/fpbase_cluster.ipynb
        └─ identity clusters  +  training_data/  (surrogate & oracle train/val/test splits)
design/guided_design_approach1.ipynb
        └─ surrogate-guided design  +  independent-oracle evaluation
```

See each subfolder's `README.md` for details.
