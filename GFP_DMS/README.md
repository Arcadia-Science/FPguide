# GFP_DMS — deep mutational scanning of GFP brightness

Processed sequence→brightness datasets and modeling code for four green-fluorescent-protein
scaffolds from two deep-mutational-scanning (DMS) studies. Each variant's full-length protein
sequence is paired with its measured (log) brightness and a data-driven **bright / dim** label,
ready for ESM-2 embedding and brightness regression / classification.

## Scaffolds & sources

| scaffold | source study | raw file |
|----------|--------------|----------|
| avGFP    | Sarkisyan et al., *Nature* **533**, 397–401 (2016). [doi:10.1038/nature17995](https://doi.org/10.1038/nature17995) · [figshare](https://figshare.com/articles/dataset/3102154) | `DMS_data/amino_acid_genotypes_to_brightness.tsv` |
| amacGFP, cgreGFP, ppluGFP | Gonzalez Somermeyer et al., *eLife* **11**:e75842 (2022). [doi:10.7554/eLife.75842](https://doi.org/10.7554/eLife.75842) · [github](https://github.com/aequorea238/Orthologous_GFP_Fitness_Peaks) | `DMS_data/amacGFP_cgreGFP_ppluGFP2__final_aminoacid_genotypes_to_brightness.csv` |

## Processed datasets

Sequences that are not clean, full-length point-substitution proteins (premature stop `*`,
ambiguous `.` calls, or C-terminal read-through) are dropped, so every row is a full-length
sequence. Brightness is on a log10 scale; each scaffold gets its own bright/dim threshold (below).

| dataset (CSV) | scaffold(s) | brightness col | sequences | bright/dim threshold (log10) | % bright |
|---------------|-------------|----------------|-----------|------------------------------|----------|
| `DMS_data/avgfp_dms_sequences.csv`  | avGFP    | `logMedianBrightness` (already log10) | 51,715 | 2.41 | 58.2% |
| `DMS_data/ortho_gfp_dms_sequences.csv` | amacGFP | `logBrightness` (= log10 of mean fluorescence) | 33,511 | 3.02 | 83.5% |
| ″ | cgreGFP | ″ | 24,516 | 3.64 | 52.8% |
| ″ | ppluGFP | ″ | 31,402 | 3.07 | 83.7% |
| **total** | | | **141,144** | | |

Shared columns: `scaffold`, `mutatedSequence`, `<log brightness>`, `linearBrightness`,
`brightnessClass` ∈ {`bright`, `dim`}. (The avGFP table names its log column `logMedianBrightness`
because the source is a median; the orthologue table uses `logBrightness` from a replicate mean.)

## Bright/dim threshold

`brightness_threshold.py` sets the split **per scaffold** at the **KDE antimode** — the
lowest-density valley between the non-functional/dead pile (censored at the assay floor) and the
functional mode near wild type. A single fixed cutoff is wrong because the floor and functional
mode sit at scaffold-specific brightness values (e.g. avGFP's dead floor is at log10 ≈ 1.3 vs.
≈ 2.7–2.8 for the orthologue assay). `visualize_thresholds.ipynb` shows the distributions, the
chosen cut, two alternatives (2-component GMM crossover, mode midpoint), and threshold sensitivity.

## Layout

```
GFP_DMS/
├── DMS_data/                         raw tables + processed CSVs (+ ESM-2 caches, git-ignored)
├── brightness_threshold.py           per-scaffold KDE-antimode bright/dim threshold
├── transform_avgfp_dms.py            raw avGFP TSV      -> avgfp_dms_sequences.csv
├── transform_ortho_dms.py            raw orthologue CSV -> ortho_gfp_dms_sequences.csv
├── embed_dms.py                      ESM-2 650M per-residue embeddings (memmap cache)
├── sweep_brightness.py               architecture sweep + full-data training for log-brightness regression
├── visualize_sweep.ipynb             sweep leaderboard, predicted-vs-true, post-hoc classifier
├── visualize_thresholds.ipynb        brightness distributions & threshold analysis
└── figures/                          exported plots
```

## Reproduce

```bash
# 1. build the processed sequence CSVs (fast, CPU)
python transform_avgfp_dms.py            # -> DMS_data/avgfp_dms_sequences.csv
python transform_ortho_dms.py            # -> DMS_data/ortho_gfp_dms_sequences.csv

# 2. ESM-2 650M embeddings (GPU; ~20 seq/s on an L4). avGFP cache already built.
python embed_dms.py                      # -> DMS_data/esm_residue_fp16.npy (+ _len)

# 3. brightness regression: architecture search, then full-data training of the winners
python sweep_brightness.py --subsample 10000       # architecture sweep
python sweep_brightness.py --top-k 3 --max-epochs 30   # train top-3 on all rows (val+test at endpoint)
```

## Not in git (see `.gitignore`)

ESM-2 embedding caches (`DMS_data/*.npy`, tens of GB), `trained_models/` (checkpoints +
prediction caches), and run logs — all regenerable from the scripts above. The raw tables and the
processed `*_sequences.csv` datasets **are** tracked.

> Note: `embed_dms.py` currently targets the avGFP CSV/paths; the orthologue embeddings are not yet
> built. Embedding all four scaffolds' 141,144 sequences takes ~1.9 h on an L4 (~54 GB for the
> orthologue cache alone).
