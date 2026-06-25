# archive/

Frozen, read-only reference notebooks. Do not edit — re-run to reproduce.

| File | Notes |
|------|-------|
| `guided_design_approach1_synthetic.ipynb` | Synthetic (`true_spectrum` toy) baseline of Approach 1; superseded by the real-data `../guided_design_approach1.ipynb`. |
| `fpbase_kmer_clustering_explorer.ipynb` | Early k-mer-proxy clustering explorer; superseded by `fpbase_cluster.ipynb` (exact alignment). |
| `fpbase_cluster.ipynb` | Sequence-identity alignment + single-linkage clustering and the cluster-wise (group-based) train/val/test splits. Archived: not using group-based splits for now (the dual notebook uses random splits). |
| `surrogate_model.ipynb` | Early surrogate exploration (mean-pool MLP + RF); superseded by `../surrogate_model_design_dual.ipynb`. |
| `surrogate_model_design.ipynb` | Pooling×architecture sweep predicting the raw 1002-dim curve (train/val only); superseded by `../surrogate_model_design_dual.ipynb` (coordinated dual datasets + PCA-coeff targets). |
| `finalize_models.ipynb` | Picked surrogate+oracle from top-2 models on separate random splits; **superseded by `../surrogate_model_design_dual.ipynb`** (coordinated dual datasets + dual labels + param-matched sweep). |
| `surrogate_model_design_random.ipynb` | Single random-split pooling×arch sweep; **superseded by `../surrogate_model_design_dual.ipynb`**. |

Frozen reference only. Note: `guided_design_approach1.ipynb` still loads `trained_models/{surrogate,oracle}_net.pt`
(produced by the archived `finalize_models.ipynb`) until it is repointed to the dual models.
