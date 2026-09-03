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
≈ 2.7–2.8 for the orthologue assay). Section 1 of `visualization.ipynb` shows the distributions,
the chosen cut, and two alternatives (2-component GMM crossover, mode midpoint).

## Layout

```
GFP_DMS/
├── DMS_data/                         processed CSVs + small caches tracked; raw tables and ESM-2 caches git-ignored
├── brightness_threshold.py           per-scaffold KDE-antimode bright/dim threshold
├── transform_avgfp_dms.py            raw avGFP TSV      -> avgfp_dms_sequences.csv
├── transform_ortho_dms.py            raw orthologue CSV -> ortho_gfp_dms_sequences.csv
├── embed_dms.py                      ESM-2 650M per-residue embeddings (memmap cache)
├── embed_parallel.py                 the same embedding pass sharded across GPUs
├── build_subsample.py                stratified 4-scaffold sub20k / sub40k caches + 70/15/15 split
├── sweep_classify_parallel.py        multi-GPU bright/dim CLASSIFIER sweep (24 configs) -> sweep_class4/
├── build_maxpool_cache.py            the 40k in-distribution reference cloud the campaigns gate on
├── nn_distance_accuracy.py           held-out accuracy stratified by the campaigns' in-distribution NN distance
├── visualization.ipynb               write-up figures: the bright/dim label, the sweep, and the campaign's ROC
└── figures/                          exported plots
```

## Reproduce

There are two ways in, and they differ by ~50 GB and two hours of GPU time. **Lane A** re-runs this
folder's analysis against the published reference cloud; **lane B** rebuilds everything from the two
source studies. Pick A unless you are specifically reproducing the classifier training.

**Lane A — re-run the analysis (~325 MB download, no GPU). Runs the notebook end to end.**

```bash
# both reference clouds + the row-aligned sequence tables they are verified against
gh release download reference-cloud-v1 -p 'sub20k_*' -p 'sub40k_*' -D DMS_data/
python build_maxpool_cache.py --verify   # confirm row i of the cloud is row i of the CSV
python nn_distance_accuracy.py --probs   # -> figures/nn_distance_accuracy.{png,csv} (+ _meta)
jupyter nbconvert --to notebook --execute --inplace visualization.ipynb
```

`gh` is pinned in `environment.yml` alongside `mafft` because it pulls all four assets in one
command, but it is a convenience rather than a requirement — the same release assets are
reachable over plain HTTPS with no credentials:

```bash
BASE=https://github.com/Arcadia-Science/FPguide/releases/download/reference-cloud-v1
for f in sub20k_maxpool.npy sub20k_sequences.csv sub40k_maxpool.npz sub40k_sequences.csv; do
  curl -fL -o "DMS_data/$f" "$BASE/$f"
done
```

Verify the transfer (`build_maxpool_cache.py --verify` checks row alignment, not bytes):

```
96854e2125a9150bd9ec1dada06ce345ea9c79fd9c748327ce3ede903cb14484  sub20k_maxpool.npy
05d96861c7d8db35758e62fceffb3ae4f4d4b2079e9e6bac426df2ba8eab815d  sub20k_sequences.csv
dc19bbb0a1b0f79bded6b7ed6dce185f6d6c0bcda54cdc86ebb13083e1024bbe  sub40k_maxpool.npz
8260f8b9aad8a48399189be796c65bcaa39a6de01ac66349a906a62ece1b56fe  sub40k_sequences.csv
```

```bash
cd DMS_data && sha256sum -c <<'EOF'
96854e2125a9150bd9ec1dada06ce345ea9c79fd9c748327ce3ede903cb14484  sub20k_maxpool.npy
05d96861c7d8db35758e62fceffb3ae4f4d4b2079e9e6bac426df2ba8eab815d  sub20k_sequences.csv
dc19bbb0a1b0f79bded6b7ed6dce185f6d6c0bcda54cdc86ebb13083e1024bbe  sub40k_maxpool.npz
8260f8b9aad8a48399189be796c65bcaa39a6de01ac66349a906a62ece1b56fe  sub40k_sequences.csv
EOF
```

The two clouds play different roles, which is what their sizes encode: **sub20k chose the
architecture** (the 24-configuration sweep and its kNN baseline ran on it) and **sub40k trained
the deployed classifier** and *is* the in-distribution reference cloud the design campaigns gate
on. They are independent draws, not nested — see step 3 of lane B.

The brightness head is tracked in git
([`fpdesign/models/brightness_cnn-max-d2_40k.pt`](../fpdesign/models/brightness_cnn-max-d2_40k.pt)),
and so is everything else the notebook reads that would otherwise require a pass over the
multi-GB embedding caches — ~41 MB in total, and the reason it runs without them:

| tracked artifact | size | what it replaces |
|---|---|---|
| `DMS_data/avgfp_dms_sequences.csv` | 14 MB | re-deriving the avGFP table from the raw study download |
| `DMS_data/ortho_gfp_dms_sequences.csv` | 25 MB | the same for the three orthologue scaffolds |
| `DMS_data/heldout_scored.npz` | 918 KB | scoring 97,499 never-fitted variants over the 31 GB + 54 GB caches |
| `trained_models/sweep_class4/results.csv` | 5 KB | the 24-configuration sweep leaderboard |
| `trained_models/sweep_class4/val_logits_top5.npz` | 61 KB | rescoring the top-5 configs over the 12 GB sub20k cache |

The two processed sequence tables are tracked rather than summarised because Section 1 recomputes
the bright/dim cut from the *continuous* log-brightness distribution — a KDE antimode per scaffold
— which the categorical `brightnessClass` column in the published subsample tables cannot stand in
for. Checking that the rule reproduces the labels the pipelines actually wrote is the point of that
section.

`heldout_scored.npz` is keyed by a hash of the checkpoint's bytes plus the size of the exclusion
set, so it is reused whenever the deployed weights and the two subsample splits are unchanged, and
correctly discarded if either moves. It does not key on the checkpoint's *filename*, which would
invalidate it on a rename.

**Why `nn_distance_accuracy.py` takes `--probs` here.** Run bare, it opens
`sub40k_esm_residue_fp16.npy` — the 24 GB per-residue cache, a lane B product — purely to run the
classifier forward over all 40,000 rows. That forward pass is the *only* thing it needs the cache
for: the NN distances come from the published max-pool cloud, and the labels and splits from the
published sequence table. `--probs` replays that one column from a previous run's
`figures/nn_distance_accuracy_per_row.csv`, which is tracked, and recomputes everything else —
the 40k × 40k self-excluded nearest-neighbour distances, the val-F1-optimal threshold, every
stratified accuracy, the Wilson intervals, the AUROCs and both figure panels.

The replayed run reproduces the committed `nn_distance_accuracy.csv` and `.png` **byte for byte**.
It is a replay of the forward pass, not an independent re-derivation of it, so `_meta.json`
records `prob_mode: replayed` and leaves `ckpt` null; re-verifying that the checkpoint produces
those probabilities is lane B, which is the lane split working as intended. Row alignment against
the sequence table is asserted rather than assumed — a truncated, shuffled or wrong CSV is
rejected with an explanation instead of yielding plausible bins from mismatched rows.

**Lane B — rebuild from the two studies.** Download the raw tables first (see
[Scaffolds & sources](#scaffolds--sources)).

> **Hardware this lane assumes.** The `--gpus 0,1,2,3` below is written for a **4-GPU host**, one
> worker per GPU. The ~1.9 h embedding figure quoted in this README is *single*-GPU wall time
> (141,144 sequences at ~20 seq/s on an L4); sharding it over 4 GPUs brings it to ~30 min. Both
> launchers default to `--gpus 0` and now **refuse to start** if you ask for an id the host does
> not have — a worker pinned to a missing device sees no GPU, falls back to CPU and never
> finishes. Pass only ids you actually have. Budget **~85 GB of free disk** for the two
> per-residue caches (31 GB avGFP + 54 GB orthologue), before the subsample caches.
>
> **Host RAM matters as much as GPU count.** Each sweep worker holds the *whole* subsample cache
> in RAM — a full copy per process, not a shared mmap — so step 4's `--gpus 0,1,2,3` needs
> **~48 GB of free RAM** on sub20k (4 × 12.1 GB fp16), and the sub40k refit **~24 GB** for its
> single worker. `sweep_classify_parallel.py` prints the budget and refuses to start if it will
> not fit (`--no-ram-check` overrides); on a smaller host, drop to fewer `--gpus` — the sweep
> still completes, each worker just takes more configs.

```bash
# 1. build the processed sequence CSVs (fast, CPU)
python transform_avgfp_dms.py            # -> DMS_data/avgfp_dms_sequences.csv
python transform_ortho_dms.py            # -> DMS_data/ortho_gfp_dms_sequences.csv

# 2. ESM-2 650M embeddings (GPU; ~20 seq/s on an L4). BOTH are required by step 3 --
#    embed_dms.py covers avGFP only, and the other three scaffolds come from embed_parallel.py.
python embed_dms.py                      # -> DMS_data/esm_residue_fp16.npy (+ _len)
python embed_parallel.py --input DMS_data/ortho_gfp_dms_sequences.csv --gpus 0,1,2,3
                                         # 4-GPU host; --gpus 0 on one GPU (the ~1.9 h figure)
                                         # -> DMS_data/ortho_gfp_dms_esm_residue_fp16.npy (+ _len)

# 3. the stratified 4-scaffold subsample caches. The default is sub20k (5k rows x 4 scaffolds);
#    sub40k -- 10k x 4, what the DEPLOYED classifier and the reference cloud both use -- is a
#    second, explicit invocation. Both draw at seed 0, but a 5k and a 10k draw from the same pool
#    are independent samples: sub20k is NOT a subset of sub40k.
python build_subsample.py                                  # -> DMS_data/sub20k_*
python build_subsample.py --per 10000 --stem sub40k        # -> DMS_data/sub40k_*

# 4. bright/dim classifier sweep: 24 configs, one worker per GPU, ranked by val AUROC.
#    First on sub20k (the exploratory sweep), then the winning family refit on sub40k. The refit
#    lands wherever --out says; nn_distance_accuracy.py does NOT default to it -- its --ckpt
#    default is the tracked ../fpdesign/models/brightness_cnn-max-d2_40k.pt, so that lane A runs
#    on a clone with no trained_models/. Step 5 passes --ckpt to score what you just trained.
python sweep_classify_parallel.py --dry-run                # list configs + shard assignment first
python sweep_classify_parallel.py --gpus 0,1,2,3           # -> trained_models/sweep_class4/results.csv
                                                           # again: 4 GPUs; --gpus 0 works, serially
python sweep_classify_parallel.py --gpus 0 --configs cnn-max-d2 \
       --data-stem sub40k --out trained_models/cnn_max_d2_40k
                                         # -> trained_models/cnn_max_d2_40k/cnn-max-d2_s0.pt

# 5. the in-distribution reference cloud the design campaigns gate on, then the figures
python build_maxpool_cache.py            # -> DMS_data/sub40k_maxpool.npz (~10 min, I/O-bound)
python nn_distance_accuracy.py --ckpt trained_models/cnn_max_d2_40k/cnn-max-d2_s0.pt
                                         # -> figures/nn_distance_accuracy.{png,csv} (+ _per_row, _meta)
                                         # drop --ckpt to score the tracked copy instead: byte-
                                         # identical weights, so the numbers are the same
jupyter nbconvert --to notebook --execute --inplace visualization.ipynb
```

The reference cloud is `sub40k` itself, so row *i* of the cloud is row *i* of
`sub40k_sequences.csv`. Everything that pairs a variant's prediction with its own embedding —
`nn_distance_accuracy.py` and `visualization.ipynb`'s held-out pass — depends on that, and both
assert it on `(src, src_row)` rather than trusting it.

The sweep's winner, `cnn-max-d2` trained on the 40k cache, is the brightness head the design
campaigns load — it is the one checkpoint from this folder tracked in git, as
`fpdesign/models/brightness_cnn-max-d2_40k.pt`.

## Not in git (see `.gitignore`)

`DMS_data/` and `trained_models/` are ignored **by default, with named exceptions** — see
`GFP_DMS/.gitignore`, where each exception is a `!` line. Not in git: the two studies' raw tables,
the `sub20k_*`/`sub40k_*` subsample caches, the ESM-2 embedding arrays (tens of GB), every
checkpoint except the brightness head, and run logs.

Tracked anyway, because the notebook and lane A read them: the five artifacts in the ~41 MB table
above — including the two processed `*_sequences.csv` — plus [`figures/`](figures) and the **code**
that rebuilds everything else. So Section 1 of the notebook runs against what is already in a
fresh clone; the transform scripts do *not* have to be re-run first.

The two raw tables are not redistributed here; download them from the sources in
[Scaffolds & sources](#scaffolds--sources) into `DMS_data/` under the filenames in that table, then
run the transform scripts above.

> Note: `embed_dms.py` targets the avGFP CSV/paths only — the orthologue embeddings come from
> `embed_parallel.py`, which is why lane B step 2 runs both. Embedding all four scaffolds' 141,144
> sequences takes ~1.9 h on an L4 (~54 GB for the orthologue cache alone). `build_subsample.py`
> hard-exits with the missing path if the orthologue cache is not there yet.
