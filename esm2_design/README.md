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
| **Surrogate / oracle** | both cnn-max | **surrogate = ESM-2 CNN, oracle = ProstT5 CNN** (independent in pLM *and* architecture) |

## Contents

| File | Role |
|------|------|
| [`peak_models.py`](peak_models.py) | Shared models: `PeakCNN`, `PeakTransformer` with `mean/min/max/std/concat/concatstd` masked pooling plus a learned `attn` pool and a learned second-order `cov` (covariance-probe) pool, a `StandardizedPeaks` wrapper (returns nm), checkpoint save/load, and ESM-2 per-residue embedding utilities. `build_base` takes `d_in` so the same backbones run on ESM-2 (1280-d) or ProstT5 (1024-d) embeddings. |
| [`pockets.py`](pockets.py) | Structure-based edit-window module (imported, not run directly). Given an experimental RCSB structure in `structures/`, defines the editable window as the chromophore tripeptide + all residues with a heavy atom within 5 Å, mapped onto the dataset sequence by alignment. Provides the original 3-scaffold `struct_pocket_experimental()` plus the generalized `experimental_window(name, seq, pdb_id)` (with an alignment-quality gate) used by [`guided_design_peak_structure_multiscaffold.ipynb`](guided_design_peak_structure_multiscaffold.ipynb) and [`parallel_pipeline/`](parallel_pipeline). |
| [`prostt5_embed.py`](prostt5_embed.py) | ProstT5 (`Rostlab/ProstT5`, MIT) encoder-only per-residue embeddings (**1024-d**, structure-aware), `<AA2fold>`-prefixed, drop-in analog of `peak_models.resid_embed`. Backs the **ProstT5 oracle** — a genuinely independent evaluator (different pLM *and* architecture family from the ESM-2 surrogate). |
| [`embed_prostt5.py`](embed_prostt5.py) | Cache ProstT5 embeddings for the curated peak set → `data/peak/curated/prostt5_residue_fp16.npy` (N,Lmax,1024) + `prostt5_residue_len.npy`, row-aligned to `peaks_assignments.csv`. |
| [`train_oracle_prostt5.py`](train_oracle_prostt5.py) | Retrain the oracle (`cnn-concatstd-d1`) on ProstT5 embeddings (or `--emb esm` for the ESM-2 baseline, identical protocol) on the dual **oracle** split → `trained_models/dual_oracle_{prostt5,esm}_net.pt` + `_scaler.npz` + `_results.json`. |
| [`sweep_peak_oracle.py`](sweep_peak_oracle.py) | Role-specific architecture sweep — **surrogate on ESM-2** (`surrogate_role` split) and **oracle on ProstT5** (`oracle_role` split), one sweep each. Sweeps backbone (mlp/cnn/transformer) × pooling (`mean/max/concat/concatstd/attn` + a `cov` covariance-probe pool on CNN d1–3) × depth × seeds → `trained_models/{surrogate_sweep,oracle_sweep}/*.pt` + per-role leaderboards. |
| [`add_seeds.py`](add_seeds.py) | Patch a role's sweep (`--role {surrogate,oracle}`) with a fixed extra list of 5 CNN configs × 3 seeds, skipping checkpoints that already exist, then prints a mean±std leaderboard — extends a sweep without rerunning it wholesale. |
| [`confirm_top3.py`](confirm_top3.py) | Seed-robustness check (`--role`): retrains the sweep's 3 top configs at 3 seeds (reusing existing seed-0 checkpoints) and writes `results_top3.csv` alongside the role's sweep, without touching the main `results.csv`. |
| [`sweep_results.ipynb`](sweep_results.ipynb) | Reads each sweep straight from its checkpoints (surrogate=ESM-2, oracle=ProstT5) and reports val/test MAE bars, val-vs-test scatter, and top-config pred-vs-truth, plus an embedding-NN null baseline. Also covers the brightness/pKa scalar sweeps. |
| [`guided_design_peak_structure_multiscaffold.ipynb`](guided_design_peak_structure_multiscaffold.ipynb) | Guided design against real scaffolds (DsRed, avGFP, eqFP578, + LSS/other targets) with the edit window defined by an experimental structure's 5 Å chromophore pocket (`pockets.py`) rather than sequence position. Produces per-task refinement trajectories, edited-residue maps, best-design-vs-target bars, and a diagnostic on why large-Stokes-shift designs plateau short of target (the LSS phenotype sits outside the editable pocket). |
| [`learning_curve.ipynb`](learning_curve.ipynb) | Two analyses on the sweep-winning surrogate (ESM-2 cnn-max-d1) and oracle (ProstT5 cnn-concatstd/max-d2): a data-scaling curve (test peak MAE vs. training-set fraction) and a sequence→peak landscape-ruggedness check (Δpeak vs. pairwise sequence identity, for ground truth and for surrogate predictions). Writes its outputs into `trained_models/` (`learning_curve_dual.png`, `ruggedness.png`, `pairsim_by_split_100pool.png`, ...). |
| [`visualization.ipynb`](visualization.ipynb) | Write-up figures over the 758 curated FPs: a t-SNE re-embedding raised into 3D "terrain" surfaces (static + interactive Plotly) and a 1D KDE-smoothed companion, a true-colour t-SNE where each point is painted its own visible-light color, the dual surrogate/oracle split's contingency heatmap, and the surrogate/oracle architecture sweeps. Writes to the repo-level [`../figures/`](../figures) (`landscape_ruggedness_*`, `landscape_truecolour_*`, `dual_split_overview.*`, `peak_{surrogate,oracle}_sweep_val_mae.*`), matching `dataset_pipeline/visualize_curation.ipynb`'s convention. |
| [`cluster_split/`](cluster_split) | Split-robustness experiments, independent of the main dual split: `run_oracle_cv.py` (populates `oracle_cv_cache/`) plus notebooks that replace the random split with 70%/85%-sequence-identity-clustered splits (`seqid70/85_cluster_split.ipynb`), retrain the fixed surrogate/oracle configs on them and compare to random (`surrogate_oracle_peak_seqid70/85.ipynb`), and run 5-fold grouped CV at both thresholds (`oracle_cross_cluster_cv.ipynb`). |
| [`parallel_pipeline/`](parallel_pipeline) | Standalone batched (CUDA fp16) known-structure design campaign: `curate_knownstruct.py` (find scaffolds with an experimental PDB + pair to train-split targets at ~80% identity), `select_knownstruct.py` (split into train/test-scaffold cohorts), `design_knownstruct.py` (guided design via `pockets.experimental_window`, resumable, `--smoke N` for wiring tests), `summarize_knownstruct.py` (per-cohort + aggregate summary tables), `visualize_knownstruct.ipynb`. Outputs land in `peak_designs/structure/parallel_pipeline/`. |
| Data & splits | Built and curated in [`../dataset_pipeline/`](../dataset_pipeline) (this replaces an earlier standalone `build_peak_dataset.py`, now merged into `../dataset_pipeline/build_dataset.py`); the curated peak set + coordinated dual split + per-residue embedding caches (ESM-2 `esm_residue_fp16.npy`, ProstT5 `prostt5_residue_fp16.npy`) live in `../dataset_pipeline/data/peak/curated/`. |
| [`archive/`](archive) | Superseded notebooks and closed-out efforts kept for reference (the ESM-2-both `surrogate_oracle_peak_dual.ipynb` trainer and its `guided_design_peak.ipynb` / `guided_design_peak_chromophore_multiscaffold.ipynb` usage, plus the large-Stokes-shift branch in [`archive/lss/`](archive/lss)). See [`archive/README.md`](archive/README.md). |

## Outputs & caches

Generated data, not checked-in logic — regenerate by rerunning the script/notebook noted.

| Directory | Contents |
|---|---|
| [`trained_models/`](trained_models) | Checkpoint store. `surrogate_sweep/` + `oracle_sweep/` (per-config `.pt` + `results.csv/json`, from `sweep_peak_oracle.py`, extended by `add_seeds.py`/`confirm_top3.py`); `lc_models/` (checkpoints at 5 training-fraction points × 3 seeds per role, from `learning_curve.ipynb`); loose learning-curve/ruggedness outputs (`learning_curve_dual.png`, `ruggedness.png`, `pairsim_by_split_100pool.png`, ...); `dual_oracle_*_net.pt` + `_scaler.npz` from `train_oracle_prostt5.py`. |
| [`peak_designs/`](peak_designs) | Guided-design CSV/PNG outputs. `chromophore/` (sequence-position-anchored window runs), `structure/` (structure-anchored runs mirroring `guided_design_peak_structure_multiscaffold.ipynb`, incl. `structure/parallel_pipeline/` = the batched known-structure campaign), `backbone/dsred/` (backbone-only ablation), `moderate/`/`temp5/` (parameter-variant reruns), plus loose early single-pair CSVs. |
| [`structures/`](structures) | Cached RCSB `.pdbx` files. Top level: the 3 hand-picked scaffolds (`1G7K` DsRed, `1GFL` avGFP, `3M22` eqFP578); `structures/experimental/` holds the ~85 additional structures for the known-structure cohort. Populated by `pockets.py`'s RCSB fetch and `parallel_pipeline/curate_knownstruct.py`. |
| [`figures/`](figures) | Plots for reporting: sweep/training-curve PNGs and design-quality diagnostics (`design_error_*`). Mixed provenance across notebooks/scripts. `visualization.ipynb`'s figures live in the repo-level [`../figures/`](../figures) instead, alongside `dataset_pipeline/visualize_curation.ipynb`'s. |
| [`oracle_cv_cache/`](oracle_cv_cache) | Per-fold `.npz` (`fold_{group70,group85,random}_N.npz`) and pooled out-of-fold predictions (`oof_{group70,group85,random}.npz`) from `cluster_split/run_oracle_cv.py`, consumed by `cluster_split/oracle_cross_cluster_cv.ipynb`. |

## Typical order

Dataset build, curation, the coordinated dual split, and the ESM-2 embedding cache are produced in
[`../dataset_pipeline/`](../dataset_pipeline) (`python ../dataset_pipeline/embed.py --trait peak`); the
curated set lives in `../dataset_pipeline/data/peak/curated/`. From this folder:

1. `python embed_prostt5.py` → cache the ProstT5 (structure-aware) per-residue embeddings for the oracle.
2. `python sweep_peak_oracle.py --role both` → train the surrogate (ESM-2) and oracle (ProstT5) sweeps → `trained_models/{surrogate_sweep,oracle_sweep}/`. Optionally patch/confirm with `add_seeds.py` / `confirm_top3.py`.
3. `sweep_results.ipynb` → inspect the leaderboards / pred-vs-truth and pick the surrogate + oracle configs.
4. `python train_oracle_prostt5.py` → finalize the chosen oracle on ProstT5 → `trained_models/dual_oracle_prostt5_net.pt` (+ scaler).

From there, three optional branches build on the finalized surrogate/oracle:

- **Structure-anchored design** — `guided_design_peak_structure_multiscaffold.ipynb` for a handful of scaffolds, or `parallel_pipeline/` for the batched known-structure campaign across ~85 structures (`curate_knownstruct.py` → `select_knownstruct.py` → `design_knownstruct.py` → `summarize_knownstruct.py`).
- **Split robustness** — `cluster_split/` reruns the surrogate/oracle protocol on sequence-identity-clustered splits (70%/85%) instead of the random dual split, to check the reported MAE isn't inflated by near-duplicate train/test leakage.
- **Reporting** — `learning_curve.ipynb` (data-scaling + landscape ruggedness) and `visualization.ipynb` (peak-landscape figures) for write-up figures.

> The original single-notebook flow that trained **both** roles on ESM-2
> (`surrogate_oracle_peak_dual.ipynb`) and its ESM-2-oracle design notebooks (`guided_design_peak.ipynb`,
> `guided_design_peak_chromophore_multiscaffold.ipynb`) now live in [`archive/`](archive).

## Notes

- Peak targets are standardized per dataset on its train split; the `StandardizedPeaks` wrapper bakes the
  inverse-standardization in, so checkpoints + saved scalers (`dual_*_scaler.npz`) reconstruct exactly.
- The oracle uses a *different pLM* than the surrogate (**ProstT5** vs the surrogate's **ESM-2**) and a
  disjoint training split, so design success is judged by a genuinely independent model — never by the
  model that guided generation.
- Generation logic (windowed ESM-2 masked-LM refinement, λ ramp, top-k naturalness leash) is unchanged from
  `../design`; only the conditioning target (peaks vs full curve) and the scoring metric differ.

### ProstT5 oracle (structure-aware evaluator)

The oracle runs on **ProstT5** encoder embeddings (structure-aware, 1024-d), making it independent of the
surrogate in *both* the pLM and the embedding modality. The surrogate that guides generation stays on
**ESM-2**. To build and train:

```bash
python embed_prostt5.py            # cache ProstT5 embeddings (~900 MB, downloads weights on first run)
python train_oracle_prostt5.py     # retrain the oracle -> trained_models/dual_oracle_prostt5_net.pt
```

To score designs with the ProstT5 oracle (e.g. adapting the archived `archive/guided_design_peak.ipynb`),
swap the oracle load/scorer:

```python
import prostt5_embed as pe
os_ = np.load("trained_models/dual_oracle_prostt5_scaler.npz")
oracle_net, o_meta = pm.load_wrapped("trained_models/dual_oracle_prostt5_net.pt", os_["mean"], os_["std"], dev)
oracle_peaks = pe.prostt5_peaks_fn(oracle_net, dev)   # ProstT5 embed + oracle, replaces pm.peaks_fn
```

Head-to-head on the dual oracle split (`cnn-concatstd-d1`, identical protocol, seeds 0/1/2), peak MAE
(nm): **ProstT5** val 13.99 ± 0.14 / test 12.28 ± 0.50; **ESM-2** val 13.98 ± 0.39 / test 10.80 ± 0.14.
The two agree on validation; ESM-2 is a bit sharper on the held-out test here.
