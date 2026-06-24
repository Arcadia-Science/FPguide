# design

ESM-2 ([`esm2_t33_650M_UR50D`](https://github.com/facebookresearch/esm), ~650 MB) tooling for
fluorescent-protein design: build a sequence↔spectrum dataset, split it without family leakage, and
design sequences toward a target excitation/emission spectrum. Part of the **esm2_fp_design** project —
the conda environment and the FPbase data live one level up (see [`../README.md`](../README.md) and
[`../fpbase-extractor`](../fpbase-extractor)).

## Environment

Uses the project-wide env `esm2-fp-design` (defined in [`../environment.yml`](../environment.yml)).
From the project root:

```bash
conda env create -f environment.yml
conda activate esm2-fp-design
```

Device is auto-detected (CUDA → MPS → CPU). On Apple Silicon it runs on the Mac GPU via **MPS**; for
any unsupported op use `PYTORCH_ENABLE_MPS_FALLBACK=1`. ESM-2 weights download on first use.

## Contents

| File | Role |
|------|------|
| [`build_fpbase_dataset.py`](build_fpbase_dataset.py) | Build the processed `(sequence, ex/em spectrum)` dataset from the fpbase-extractor export → `../fpbase-extractor/processed_data/ESM-spectrum/` (`spectra.npy`, `sequences.fasta`, `metadata.csv`). |
| [`fpbase_cluster.ipynb`](fpbase_cluster.ipynb) | Sequence-identity alignment (Biopython `PairwiseAligner`), clustering exploration (dendrogram, between-cluster heatmaps, membership lists), and the **coordinated surrogate/oracle train/val/test splits** → `training_data/`. The identity matrix is cached (`identity_matrix.npy` + fingerprint) so restarts skip re-alignment. |
| [`guided_design_approach1.ipynb`](guided_design_approach1.ipynb) | **Property-guided design** — a small surrogate `g(seq)→spectrum` (on frozen ESM-2 embeddings) steers ESM-2 masked refinement toward a target spectrum; designs are scored by an **independent oracle**, never by `g`. Currently a synthetic-data demo, to be wired to `training_data/`. |
| [`design.ipynb`](design.ipynb) | General masked-sequence design with ESM-2: naive single-pass fill, iterative/autoregressive decoding, and pseudo-perplexity ranking. |
| [`training_data/`](training_data) | Saved cluster-wise splits (see its README). |
| [`archive/`](archive) | Frozen references: the original `design.py` CLI, the synthetic Approach-1 baseline, and the k-mer clustering explorer. |

## Typical order

1. In `../fpbase-extractor`: `fpbase-extract --spectra` → `fpbase_output/`.
2. `python build_fpbase_dataset.py` → `processed_data/ESM-spectrum/`.
3. `fpbase_cluster.ipynb` → identity clusters + `training_data/` splits.
4. `guided_design_approach1.ipynb` → surrogate-guided design + oracle evaluation.

## Design approaches

- **Approach 1 (implemented, synthetic demo):** inference-time guidance. Freeze ESM-2, train a small
  surrogate `g`, and steer masked refinement by `score = z(log p_ESM) − λ·z(spectral error)` within
  ESM's top-k (the naturalness leash). Evaluation uses an independent oracle (different architecture),
  not the surrogate — see the notebook for the surrogate-vs-oracle and on/off-manifold caveats.
- **Approach 2 (planned):** a FiLM adaptor + conditional masked-LM that learns spectrum conditioning
  into a frozen ESM-2 via a small trainable adaptor.
