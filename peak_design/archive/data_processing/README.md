# Archived data-processing (superseded by `dataset_pipeline/`)

These files built and curated the FP datasets before the pipeline was consolidated into
[`../../../dataset_pipeline/`](../../../dataset_pipeline). They're kept for provenance and are no longer
the source of truth — the rules here differ from the current pipeline (see the `dataset_pipeline` README
for the reasoning behind every current rule and how the outputs changed).

## What's here

| Item | Was | Replaced by |
|------|-----|-------------|
| `build_peak_dataset.py` | raw peak dataset builder | `dataset_pipeline/build_dataset.py --target peak` |
| `curate_split_visualize.ipynb` | peak curation + dual split + viz | `dataset_pipeline/build_dataset.py` + `make_dual_split.py` |
| `build_curate_split_pka.ipynb` | pKa build + curate + split + learning curve + off-target scoring | build/curate/split → `dataset_pipeline`; learning curve → `../../learning_curve_pka.ipynb` |
| `build_curate_split_brightness.ipynb` | brightness build + curate + split + learning curve | build/curate/split → `dataset_pipeline`; learning curve → `../../learning_curve_brightness.ipynb` |
| `training_data/` | raw + curated peak data, dual splits, and the ESM-2 residue embedding cache | `dataset_pipeline/data/peak/curated/` |
| `training_data_brightness/` | raw + curated brightness data | `dataset_pipeline/data/brightness/curated/` |
| `training_data_pka/` | raw + curated pKa data | `dataset_pipeline/data/pka/curated/` |

## Why the datasets changed

The new pipeline isn't a drop-in reproduction — it fixes several issues these notebooks had (name-based
sensor drops that discarded valid FPs, an untagged biliverdin IR-FP left in, silent dedup on peak while
brightness disagreed, etc.). `dataset_pipeline/compare.py` diffs the two and every delta is documented.
If you need the exact old datasets, they're preserved here unchanged.

## Note on off-target scoring

`build_curate_split_pka.ipynb` also had a section 7 that scored the guided designs against the full-pool
pKa model. That wasn't extracted (only the learning curve was kept live). It's preserved in the archived
notebook if you want to revive it against the new data.
