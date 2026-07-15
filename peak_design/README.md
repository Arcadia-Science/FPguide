# peak_design

Peak-conditioned counterpart of [`../design`](../design). Same ESM-2
([`esm2_t33_650M_UR50D`](https://github.com/facebookresearch/esm)) fluorescent-protein design workflow —
build a dataset, split it without val leakage, train a surrogate + oracle, and run surrogate-guided
sequence design judged by the oracle — but **conditioned on the spectral peaks `(ex_max, em_max)`** (in nm,
taken over the full spectrum) instead of the full 1002-dim excitation/emission curve. Part of the
**esm2_fp_design** project; the conda environment and FPbase data live one level up.

## Environment

Project-wide env `esm2-fp-design` (see [`../environment.yml`](../environment.yml)):

```bash
conda env create -f ../environment.yml   # from this folder
conda activate esm2-fp-design
```

Device auto-detected (CUDA → MPS → CPU). On Apple Silicon set `PYTORCH_ENABLE_MPS_FALLBACK=1` for any
unsupported op. ESM-2 weights download on first use.

## What changed vs `../design`

| | `../design` (full spectrum) | `peak_design` (this folder) |
|---|---|---|
| **Data source** | `processed_data/ESM-spectrum` (gated on full ex+em curves → 453 samples) | `fpbase_proteins.json` directly (gated only on seq + reported peaks → **894 samples**, 832 unique seqs) |
| **Target** | 1002-dim ex/em curve (PCA-reduced to 32 coeffs) | 2 scalars `(ex_max, em_max)` in nm |
| **Backbones** | MLP + CNN over ESM-2 embeddings | **CNN + Transformer** over ESM-2 embeddings |
| **Pooling readouts** | mean/min/max/std/concat | **mean/min/max/concat** |
| **Training loss** | per-half cosine similarity | standardized-peak MSE |
| **Selection metric** | val MSE (reconstructed spectrum) | val **peak MAE (nm)** |
| **Guidance score** | `z(log p_ESM) + λ·z(cosine to target spectrum)` | `z(log p_ESM) + λ·z(−peak error to target)` |
| **Surrogate / oracle** | both cnn-max | **surrogate = best CNN, oracle = best Transformer** (independent families) |

## Contents

| File | Role |
|------|------|
| [`build_peak_dataset.py`](build_peak_dataset.py) | Build the **raw** dataset from `../fpbase-extractor/fpbase_output/fpbase_proteins.json` → `training_data/` (`peaks.npy` (N,2), `sequences.fasta`, `peak_assignments.csv`, `peak_meta.json`). One sample per (protein, state) with a standard-AA sequence and reported `ex_max`/`em_max`. Raw-build only — no curation or splitting. |
| [`peak_models.py`](peak_models.py) | Shared models: `PeakCNN`, `PeakTransformer` (mean/min/max/concat pooling), a `StandardizedPeaks` wrapper (returns nm), checkpoint save/load, and ESM-2 per-residue embedding utilities. `build_base` takes `d_in` so the same backbones run on ESM-2 (1280-d) or ProstT5 (1024-d) embeddings. |
| [`prostt5_embed.py`](prostt5_embed.py) | ProstT5 (`Rostlab/ProstT5`, MIT) encoder-only per-residue embeddings (**1024-d**, structure-aware), `<AA2fold>`-prefixed, drop-in analog of `peak_models.resid_embed`. Backs the **ProstT5 oracle** — a genuinely independent evaluator (different pLM *and* architecture family from the ESM-2 surrogate). |
| [`embed_prostt5.py`](embed_prostt5.py) | Cache ProstT5 embeddings for the curated peak set → `data/peak/curated/prostt5_residue_fp16.npy` (N,Lmax,1024) + `prostt5_residue_len.npy`, row-aligned to `peaks_assignments.csv`. |
| [`train_oracle_prostt5.py`](train_oracle_prostt5.py) | Retrain the oracle (`cnn-concatstd-d1`) on ProstT5 embeddings (or `--emb esm` for the ESM-2 baseline, identical protocol) on the dual **oracle** split → `trained_models/dual_oracle_{prostt5,esm}_net.pt` + `_scaler.npz` + `_results.json`. |
| [`curate_split_visualize.ipynb`](curate_split_visualize.ipynb) | Reads the raw `training_data/`, flags outliers (cofactor ∪ NN-4mer<0.10 ∪ frFAST/nirFAST), writes the **curated** dataset → `training_data/curated/`, does the coordinated surrogate/oracle split, and visualizes the splits (embedding, ex_max, em_max, length). |
| [`surrogate_oracle_peak_dual.ipynb`](surrogate_oracle_peak_dual.ipynb) | Two coordinated train/val/test splits; train CNN×4 + Transformer×4; pick **surrogate** (best CNN) and **oracle** (best Transformer); save `trained_models/dual_*` + per-sample roles to `training_data/dual_splits.csv`. |
| [`guided_design_peak.ipynb`](guided_design_peak.ipynb) | Surrogate-guided masked refinement toward a target `(ex_max, em_max)`, scored independently by the oracle; sequence-similar/spectrally-distinct and ~80%-identity peak-shift examples, λ ramp 20→47 over 10 rounds. |
| [`training_data/`](training_data) | Peak targets + splits + the per-residue embedding cache (`esm_residue_fp16.npy`, ~1.7 GB, built on first run). `training_data/curated/` holds the outlier-filtered dataset + its split. |
| [`archive/`](archive) | EDA notebooks kept for inspection: `outlier_visualization*.ipynb`. See [`archive/README.md`](archive/README.md). |

## Typical order

1. `python build_peak_dataset.py` → `training_data/peaks.npy`, `sequences.fasta`, `peak_assignments.csv` (reads `../fpbase-extractor/fpbase_output/fpbase_proteins.json`).
2. `curate_split_visualize.ipynb` → flag/drop outliers → `training_data/curated/` + the coordinated split.
3. `surrogate_oracle_peak_dual.ipynb` → trained surrogate/oracle + `training_data/dual_splits.csv` (builds the ESM-2 embedding cache on first run).
4. `guided_design_peak.ipynb` → peak-guided design + oracle evaluation → `peak_designs/`.

## Notes

- Peak targets are standardized per dataset on its train split; the `StandardizedPeaks` wrapper bakes the
  inverse-standardization in, so checkpoints + saved scalers (`dual_*_scaler.npz`) reconstruct exactly.
- The oracle is a *different architecture family* than the surrogate (Transformer vs CNN), so design success
  is judged by a genuinely independent model — never by the model that guided generation.
- Generation logic (windowed ESM-2 masked-LM refinement, λ ramp, top-k naturalness leash) is unchanged from
  `../design`; only the conditioning target (peaks vs full curve) and the scoring metric differ.

### ProstT5 oracle (structure-aware evaluator)

The oracle can be run on **ProstT5** encoder embeddings instead of ESM-2, making it independent of the
surrogate in *both* the pLM and the architecture family. The surrogate that guides generation is
untouched (still ESM-2). To build and train:

```bash
python embed_prostt5.py            # cache ProstT5 embeddings (~900 MB, downloads weights on first run)
python train_oracle_prostt5.py     # retrain the oracle -> trained_models/dual_oracle_prostt5_net.pt
```

To evaluate designs with the ProstT5 oracle in `guided_design_peak.ipynb`, swap the oracle load/scorer:

```python
import prostt5_embed as pe
os_ = np.load("trained_models/dual_oracle_prostt5_scaler.npz")
oracle_net, o_meta = pm.load_wrapped("trained_models/dual_oracle_prostt5_net.pt", os_["mean"], os_["std"], dev)
oracle_peaks = pe.prostt5_peaks_fn(oracle_net, dev)   # ProstT5 embed + oracle, replaces pm.peaks_fn
```

Head-to-head on the dual oracle split (`cnn-concatstd-d1`, identical protocol, seeds 0/1/2), peak MAE
(nm): **ProstT5** val 13.99 ± 0.14 / test 12.28 ± 0.50; **ESM-2** val 13.98 ± 0.39 / test 10.80 ± 0.14.
The two agree on validation; ESM-2 is a bit sharper on the held-out test here.
