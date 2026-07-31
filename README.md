# esm2_fp_design

ESM-2–guided design of **fluorescent proteins (FPs)** conditioned on their photophysical
properties. The core pipeline has three parts:

- **[`fpbase-extractor/`](fpbase-extractor)** — pulls FP sequences, phenotypes, and full
  excitation/emission spectral curves from [FPbase](https://www.fpbase.org).
- **[`dataset_pipeline/`](dataset_pipeline)** — curates the raw export into the peak,
  brightness, and pKa training sets, and builds leakage-safe train/val/test splits.
- **[`esm2_design/`](esm2_design)** — designs sequences toward target properties with ESM-2
  and evaluates them against an independent oracle.

On top of that core, a design-campaign layer applies the same surrogate/oracle machinery to
specific scaffold-recoloring goals, backed by a few supporting datasets/analyses. See
"[Design campaigns](#design-campaigns)" and "[Supporting analyses](#supporting-analyses)" below.

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
fpbase-extractor (fpbase-extract)
        └─ fpbase_output/fpbase_proteins.json    raw sequences + phenotypes
dataset_pipeline/build_dataset.py
        └─ data/<trait>/curated/                 curated peak / brightness / pKa sets
dataset_pipeline/make_dual_split.py
        └─ dual_splits.csv                       coordinated surrogate & oracle splits
esm2_design/
        └─ property-guided design  +  independent-oracle evaluation
```

See each subfolder's `README.md` for details.

The earlier `(sequence, spectrum)` lineage — `design/build_fpbase_dataset.py` feeding
`fpbase-extractor/processed_data/ESM-spectrum/` — has been archived to
`fpbase-extractor/archive/esm_spectrum/`. The full lineage (including the earlier full-spectrum
sibling of `esm2_design/`) is:

- **[`design/`](design)** — the full-spectrum-conditioned predecessor of `esm2_design/`: same
  ESM-2 surrogate-guided design idea, but conditioned on the whole 1002-dim ex/em curve instead
  of just the peaks. Superseded by `esm2_design/`, kept for reference (not actively developed).
- **`archive/scalar_design/`** — the scalar-trait (brightness, pKa) counterpart of
  `esm2_design/`'s peak-conditioned sweeps. Exploratory and orphaned (it imports from a sibling
  `peak_design/` path left over from before `esm2_design/` was renamed, so it doesn't currently
  run); archived and untracked, kept for reference only.

## Design campaigns

A second layer runs concrete "recolor this scaffold toward a target FP" campaigns using the
surrogate/oracle/pocket machinery from `esm2_design/`:

- **[`fpdesign/`](fpdesign)** — shared library extracted from the campaign scripts (`Campaign`,
  `CampaignConfig`, edit-window construction). Reuses `esm2_design/pockets.py`'s pocket rules.
  Not a campaign itself — the engine the active campaign below imports.
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
- **`licensing/`** — flags which curated FPs are likely patent-expired/open for commercial use
  (used to pick campaign targets). Kept local (not published in this repo — see `.gitignore`).
