# spectrum-to-fp-design

Code and data for the Arcadia Science publication
[doi:10.57844/arcadia-66aw-aa84](https://doi.org/10.57844/arcadia-66aw-aa84).

ESM-2–guided design of **fluorescent proteins (FPs)** conditioned on their photophysical
properties. The core pipeline has three parts:

- **[`fpbase-extractor/`](fpbase-extractor)** — pulls FP sequences, phenotypes, and full
  excitation/emission spectral curves from [FPbase](https://www.fpbase.org).
- **[`dataset_pipeline/`](dataset_pipeline)** — curates the raw export into the peak,
  brightness, and pKa training sets, and builds leakage-safe train/val/test splits.
- **[`in-silico-test/`](in-silico-test)** — designs sequences toward target properties inside a
  structure-defined chromophore pocket and evaluates them against an independent oracle. The
  per-residue proposal is ESM-2 650M's masked-LM logits, with an unguided pocket-resampling arm as
  the null. An earlier controlled swap ran the same search from a family-MSA profile instead (see
  [Findings](#findings)); that arm is archived. It vendors its own copy of the shared model/pocket machinery
  (`peak_models.py`, `pockets.py`) so the experiment folder stays self-contained; the campaign
  layers below import that machinery from **[`fpdesign/`](fpdesign)** instead.

On top of that core, a design-campaign layer applies the same surrogate/oracle machinery to
specific scaffold-recoloring goals, backed by a few supporting datasets/analyses. See
"[Design campaigns](#design-campaigns)" and "[Supporting analyses](#supporting-analyses)" below.

## Findings

### A 763-sequence family profile matches ESM-2 650M as the design proposal

The search needs one generative component: given a scaffold and one editable pocket position,
which residues are worth proposing and how plausible is each. Everything downstream is
discriminative. [`in-silico-test/`](in-silico-test) runs that component two ways as a **controlled
swap** — same 108 scaffold→target tasks, same structural windows, same surrogate and oracle, same
seed, same 3 trials, same score — differing in one line of the loop: the proposal is either ESM-2's
masked-LM log-probability at the edited position, or a static Henikoff-weighted PSSM read off the
763-sequence family alignment from [`msa_conservation/`](msa_conservation).

Oracle-scored mean absolute peak error, nm (`3.1` vs
`3.2`, both archived):

| proposal | scaffold | design (surrogate-selected) | mean of trials | improved | trial spread | identity | fam_logp/pos |
|---|---|---|---|---|---|---|---|
| MSA PSSM | 133.2 | **87.2** | 94.0 | 103/108 | 28.1 nm | 0.922 | −2.11 |
| ESM-2 650M | 133.2 | 89.2 | **92.8** | 97/108 | 24.4 nm | 0.903 | −3.80 |

**On aim they are indistinguishable.** The sign of the difference flips with the statistic — MSA is
2.0 nm better on the surrogate-selected design (Wilcoxon *p* = 0.30), ESM-2 is 1.2 nm better on the
mean over trials (*p* = 0.69) — and both gaps are ~10× smaller than the 24–28 nm spread *within* a
task across trials of the *same* arm. Any single-trial comparison of these arms was measuring its
own noise.

What survives the trials is a trade, not a winner: ESM-2 edits less conservatively (90.3% vs 92.2%
identity to scaffold, family log-likelihood 1.8× worse per position) and puts **3,863 of its 14,463
edits — 27% — outside the family-supported alphabet**, where the PSSM arm cannot leave family
support by construction (0 of 11,627). It improves 6 fewer tasks, and its trials are less variable
(24.4 vs 28.1 nm), the one axis on which it is cleanly ahead.

> The two family-support numbers above are **no longer reproducible from this repo.** They were
> measured against the per-position family alphabets that the design windows used to carry, and
> both the alphabets and the alignment behind them were removed once the MSA proposal arm was
> retired — the surviving arms never consulted them. The design CSVs those runs produced are still
> tracked, so the identity and error figures are checkable; the family columns are not. Restoring
> them means restoring `in-silico-test/archive/msa/` and rebuilding the windows.

### ESM-2 is close to uninformative on the GFP fold

The reason a 763-sequence profile keeps up is measurable
([`msa_conservation/esm_vs_family.py`](msa_conservation/esm_vs_family.py),
`results/esm_calibration.csv`). Masked single-residue recovery:

| protein | length | masked top-1 acc. | mean max prob | median rank of true residue |
|---|---|---|---|---|
| ubiquitin | 76 | 0.803 | 0.754 | 1.0 |
| lysozyme | 129 | 0.659 | 0.682 | 1.0 |
| trypsin | 223 | 0.735 | 0.687 | 1.0 |
| **EGFP** | 239 | **0.126** | **0.128** | **6.0** |
| **avGFP** (wild type) | 238 | **0.101** | **0.125** | **6.0** |
| **mCherry** | 236 | **0.178** | **0.170** | **6.0** |

On ordinary proteins spanning 76–223 residues ESM-2 650M recovers the true residue as its top
choice 66–80% of the time at ~0.7 confidence; on three FPs of comparable length it drops to 10–18%
at ~0.13, with Gly on top at most positions — close to a background amino-acid distribution. Not a
length effect (trypsin at 223 behaves like the short controls), not an engineering
artifact (wild-type avGFP is the worst of the three), not a harness bug (same code produces the
sharp control numbers; unmasked reconstruction of EGFP is 99.2%). Across the EGFP design window the
two priors are nearly orthogonal: mean Spearman(family frequency, ESM-2 probability) = **+0.108**,
negative at 11 of 28 positions, top-1 agreeing at **2 of 28**. *Why* is not established here.

**So the disagreement is not two opinions in conflict — it is one model having almost no opinion**,
which is what makes the cheap family profile a fair substitute for the 650M-parameter model on this
fold, and why both arms are kept rather than picking one on priors.

### The proposal is not the axis that matters — the surrogate is

Against the unguided null (`3.3`, the same search with
λ_ex = λ_em = 0, so the target never enters), measured on the 72 tasks all three arms share:

| | mean of trials | vs null |
|---|---|---|
| MSA PSSM, guided | 92.7 | **+13.0 nm** (*p* = 1e−5) |
| ESM-2, guided | 91.4 | **+14.4 nm** (*p* = 1.7e−8) |
| ESM-2, unguided | 105.8 | — |
| *arm swap (MSA vs ESM-2)* | | *1.1–1.3 nm (p ≈ 0.6)* |

Changing where the proposal comes from is not measurable at this sample size; switching the
surrogate off is, by roughly an order of magnitude in effect size. Two further cautions the same
control forces: **~65% of the headline 133.2 → 87.2 nm gain is reproduced by a search that never
sees the target** (pocket resampling alone reaches 105.8 nm), so the surrogate's real contribution
is "13–14 nm beyond pocket resampling"; and ESM-2's looser editing and worse family likelihood are
properties of the *proposal itself*, not consequences of steering it (unguided ESM-2 already sits at
0.902 identity, far from MSA's 0.921).

Carrying this over needs one caveat: the `design-campaign-EGFP` MSA and ESM-2 strategies retuned λ
and temperature at the same time as the proposal, so "only the proposal differs" is true of the
benchmark and **not** of the campaign.

## Environment (one combined conda env)

From this directory (the project root, so the editable `fpbase-extractor` install resolves):

```bash
conda env create -f environment.yml
conda activate spectrum-to-fp-design
```

Device is auto-detected at runtime: **CUDA → MPS → CPU**. Every result in this repo was produced
on Linux + CUDA (an L4). Apple Silicon works too — macOS has no CUDA, so the Mac GPU is used via
**MPS**; set `PYTORCH_ENABLE_MPS_FALLBACK=1` for any op MPS doesn't implement. The ESM-2 650M
weights (~650 MB) download to the torch cache on first use.

`fpdesign/` is imported but **not installed** — the folders that use it put the repo root on
`sys.path` at the top of each script, so run them from where their own README says to.

## Pipeline

```
fpbase-extractor (fpbase-extract)
        └─ fpbase_output/fpbase_proteins.json    raw sequences + phenotypes
dataset_pipeline/build_dataset.py
        └─ data/<trait>/curated/                 curated peak / brightness / pKa sets
dataset_pipeline/make_dual_split.py
        └─ dual_splits.csv                       coordinated surrogate & oracle splits
in-silico-test/
        └─ property-guided design  +  independent-oracle evaluation
           (self-contained: model/pocket code vendored into its lib/)
```

See each subfolder's `README.md` for details.

**Figures.** Every folder keeps its own `figures/`, written by that folder's notebook — there is
no shared figure directory. [`dataset_pipeline/figures/`](dataset_pipeline/figures)
(`visualize_curation.ipynb`), [`in-silico-test/figures/`](in-silico-test/figures)
(`figures.ipynb`),
[`GFP_DMS/figures/`](GFP_DMS/figures) (`visualization.ipynb`,
`nn_distance_accuracy.py`), [`msa_conservation/figures/`](msa_conservation/figures)
(`plot_conservation.py`, `visualization.ipynb`) and
[`design-campaign-EGFP/figures_benchmark/`](design-campaign-EGFP/figures_benchmark)
(`visualization.ipynb`).

Three earlier experiments led here. All are retired, untracked, and kept only on the authors'
machines; they are named because the write-ups cite them, not because you can open them:

- **`archive/design/`** — the full-spectrum-conditioned predecessor: the same ESM-2
  surrogate-guided design idea, but conditioned on the whole 1002-dim ex/em curve instead of the two
  peaks. Retired for two reasons — its `(sequence, spectrum)` input dataset is gone, and its
  independent re-curation of the curve-bearing proteins is the approach
  [`dataset_pipeline/build_spectra_dataset.py`](dataset_pipeline/build_spectra_dataset.py)
  deliberately replaced (re-curating from curve-bearing states alone let analyte sensors and a FRET
  biosensor back in; filtering the curated peak set inherits that judgment instead). Its
  `fp_models.py` is the ancestor of [`fpdesign/peak_models.py`](fpdesign/peak_models.py).

  **The full-spectrum *dataset* is not retired** — only this experiment. `build_spectra_dataset.py`
  and its 382-row output in `dataset_pipeline/data/spectra/curated/` are current and tracked. What
  the published repo no longer contains is a design experiment that consumes whole curves; the
  design work here is peak-conditioned throughout.
- **`archive/esm2_design/`** — the original peak-conditioned design experiment (campaign and
  visualization notebooks, design outputs, CV/sweep caches, checkpoints, figures). Its shared
  `peak_models.py` / `pockets.py` moved into [`fpdesign/`](fpdesign) and its PDBx cache into
  [`structures/`](structures). Superseded by [`in-silico-test/`](in-silico-test).
- **`archive/scalar_design/`** — the scalar-trait (brightness, pKa) counterpart of the
  peak-conditioned sweeps. Exploratory and orphaned: it imports from a sibling `peak_design/` path
  left over from an earlier layout, so it does not currently run.

## Design campaigns

A second layer runs concrete "recolor this scaffold toward a target FP" campaigns using the
surrogate/oracle/pocket machinery in `fpdesign/` (`peak_models.py`, `pockets.py`):

- **[`fpdesign/`](fpdesign)** — the shared library: `peak_models.py` (ESM-2 utilities +
  surrogate/oracle architectures), `pockets.py` (structure-based chromophore-pocket rules),
  `campaign.py` (`Campaign`/`CampaignConfig`, extracted from the campaign scripts),
  `build_design_windows.py`, and the shared checkpoints in `models/`. Not a campaign itself —
  the engine the campaigns below import, and also the source of the ESM-2 embedding helpers used
  by `dataset_pipeline/` and `GFP_DMS/`. See [`fpdesign/README.md`](fpdesign/README.md).
- **[`design-campaign-EGFP/`](design-campaign-EGFP)** — six parallel strategies (gibbs, guided,
  guided+constraint, brightness-guided, MSA-guided, MSA-gibbs) recoloring EGFP toward EBFP/mOrange.
  The active campaign; draws on `GFP_DMS`'s brightness classifier and `msa_conservation`'s
  alignment/PSSM.

Two earlier/parallel campaigns have been archived (moved to `archive/`, untracked) and are kept
locally for reference only:

- **`archive/design-campaign-conventional/`** — the original campaign: 24 scaffold→target pairs
  among popular, structurally-characterized, non-large-Stokes-shift FPs, compared across
  gibbs-sampling and guided-design strategies. `fpdesign/` was extracted from its scripts.
- **`archive/design-campaign-avGFP/`** — the same head-to-head comparison on the avGFP scaffold,
  toward EBFP/mEmerald/mOrange/mCherry.

## Supporting analyses

Datasets and analyses that feed the design campaigns rather than the core pipeline directly:

- **[`GFP_DMS/`](GFP_DMS)** — curates two published deep-mutational-scanning datasets
  (avGFP, amacGFP/cgreGFP/ppluGFP) into ~141k sequence→brightness rows and trains an ESM-2
  brightness classifier/regressor. Its checkpoint and in-distribution embedding cloud are used by
  `design-campaign-EGFP`'s brightness-guided strategy and to vet shortlisted designs.
- **[`msa_conservation/`](msa_conservation)** — MAFFT alignment of the curated FP set with
  per-position conservation analysis. Its family MSA/PSSM backs the MSA-guided and MSA-gibbs
  strategies in `design-campaign-EGFP`.
- **[`structures/`](structures)** — shared RCSB PDBx cache (`structures/experimental/`) that
  `fpdesign/pockets.py` reads when building edit windows, plus the few whole-barrel references
  (`1GFL.pdbx`) used by `msa_conservation/`. Self-populating: `pockets.py` fetches a missing PDB
  ID from RCSB into this folder on demand.

## What a reproducer can regenerate

Four tiers, and which one a figure falls into is the first thing worth knowing. Each folder's
README has the detail; this is the map.

**Tier 1 — runs from a bare clone.** No download beyond the conda env, no GPU: every input is
tracked.

| entry point | reads |
|---|---|
| `msa_conservation/visualization.ipynb`, `plot_conservation.py` | tracked `data/` (the 763-sequence union + MAFFT alignment) and all 13 files of `results/` |
| `experiment/0831_spectrum/*.ipynb` | the tracked plate reads (`*_raw_well_data.csv`) and `fpbase-extractor/fpbase_output/fpbase_spectra.json` |

**Tier 2 — one download, no training.** Add the `reference-cloud-v1` release (~325 MB; see
[`GFP_DMS/README.md`](GFP_DMS/README.md) lane A).

| entry point | also needs |
|---|---|
| `GFP_DMS/visualization.ipynb`, `nn_distance_accuracy.py --probs` | both clouds + their row-aligned sequence tables. Section 1 of the notebook is Tier 1; the rest is not |
| `design-campaign-EGFP/benchmark_report.py`, `esm2_guided/analyze.py`, `visualization.ipynb` | `sub40k_maxpool.npz` (imported by `make_shortlist_case.py`) **and** the ESM-2 650M weights, to embed designs — so a GPU in practice |

**Tier 3 — needs a gitignored cache rebuilt locally first** (GPU, and see the size table below).
These fail with the command to run, not a traceback.

| entry point | missing input | rebuild with |
|---|---|---|
| `dataset_pipeline/visualize_curation.ipynb` | `data/fpbase_esm2_650M_max.npy` | `python dataset_pipeline/embed_fpbase_maxpool.py` |
| `in-silico-test/figures.ipynb` | the ESM-2 **and** ProstT5 per-residue caches, plus `trained_models/` (checkpoints + `surrogate_cv3.csv`) | `dataset_pipeline/embed.py`, `embed_prostt5.py`, then that folder's sweep/CV scripts |
| `in-silico-test/sweep_results.ipynb` | `trained_models/` (the sweep leaderboards and top checkpoints) | the sweep scripts in `in-silico-test/1_surrogate_oracle_training/` |

**Tier 4 — not regenerable from the published tree**, and the READMEs say so where they cite it:
the frozen `design-campaign-EGFP/figures/` (the notebook that made it was archived), task set 1
and the archived MSA arm of `in-silico-test/`, everything under any `archive/`, and the plate
reads in `experiment/` — primary measurements, tracked precisely because they cannot be
recomputed.

## What's in this repo, what's kept local

Some material this README refers to is deliberately not published, so a reader who goes looking for
it knows it is absent by design rather than missing:

| not in git | why | how to get it back |
|---|---|---|
| `archive/` folders, anywhere in the tree | superseded experiments, old notebooks, dropped targets. Referred to by name in these READMEs because the write-ups cite them, but read by no live code — if active code needs a file, that file is not in `archive/`. | not regenerable; kept on the authors' machines |
| a few exploratory experiment folders | side experiments that are not part of the pipeline or the campaigns above | as above |
| ESM-2 per-residue embedding caches (`esm_residue_fp16.npy`, `fpbase_esm2_650M_max.npy`, `DMS_data/*.npy`) | 0.5–50 GB each, over GitHub's file limit | re-run `dataset_pipeline/embed.py` / `embed_fpbase_maxpool.py`, `GFP_DMS/embed_dms.py` + `embed_parallel.py` |
| ProstT5 per-residue embedding cache (`prostt5_residue_fp16.npy`) — the oracle input | ~0.9 GB | re-run `dataset_pipeline/embed_prostt5.py` (`--verify` / `--spot-check` validate one you already have) |
| `GFP_DMS/DMS_data/sub40k_maxpool.npz` | the 40k in-distribution reference cloud, ~207 MB | download the published copy — [`GFP_DMS/README.md`](GFP_DMS/README.md) lane A. `build_maxpool_cache.py` alone will *not* rebuild it: it streams the 24 GB per-residue cache, so rebuilding means all of lane B (two study downloads + ~85 GB of embedding) |
| `trained_models/`, and `*.pt` generally | regenerable checkpoints and cached predictions | re-run the sweep/training script in the folder that owns them |
| the Atkinson Hyperlegible faces (`dataset_pipeline/fonts/`) | cut from the Google Fonts variable originals, not ours to redistribute | `python dataset_pipeline/fetch_arcadia_fonts.py` |

Two checkpoints **are** tracked, because every campaign loads them and neither is cheap to
reproduce: [`fpdesign/models/surrogate_cnn-max-d1_alldata.pt`](fpdesign/models) and
`fpdesign/models/brightness_cnn-max-d2_40k.pt`.
