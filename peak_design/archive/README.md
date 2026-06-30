# archive — dataset-inspection notebooks

Exploratory EDA kept for reference while inspecting the FP dataset. Neither is on the live path
(`../build_peak_dataset.py` → `../curate_split_visualize.ipynb` → `../surrogate_oracle_peak_dual.ipynb` →
`../guided_design_peak.ipynb`).

| File | What it is |
|------|------------|
| [`outlier_visualization.ipynb`](outlier_visualization.ipynb) | EDA of the **full 894** samples along five outlier criteria (peak-label, sequence-isolation, embedding anomaly, length, cofactor); spectral map, t-SNE, consensus heatmap. |
| [`outlier_visualization_filtered.ipynb`](outlier_visualization_filtered.ipynb) | The same EDA **after** dropping cofactor-bound FPs (835 canonical), with the four remaining criteria recomputed within the subset — surfaces the residual outliers (incl. the cofactor proteins FPbase left untagged). |

Both are stored **with their executed outputs** (figures/tables embedded), so they can be inspected as-is.
Their data paths are relative to the parent `peak_design/` folder — to *re-run* them, launch Jupyter with
`peak_design/` as the working directory and open `archive/<notebook>.ipynb` (not from inside `archive/`).

The outlier filter these notebooks explored is now applied for real in
[`../curate_split_visualize.ipynb`](../curate_split_visualize.ipynb) (cofactor ∪ NN-4mer<0.10 ∪ frFAST/nirFAST),
which writes the curated dataset + split to `../training_data/curated/`.
