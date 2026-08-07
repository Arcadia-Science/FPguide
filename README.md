# esm2_fp_design

ESM-2–guided design of **fluorescent proteins (FPs)** conditioned on their photophysical
properties. The core pipeline has three parts:

- **[`fpbase-extractor/`](fpbase-extractor)** — pulls FP sequences, phenotypes, and full
  excitation/emission spectral curves from [FPbase](https://www.fpbase.org).
- **[`dataset_pipeline/`](dataset_pipeline)** — curates the raw export into the peak,
  brightness, and pKa training sets, and builds leakage-safe train/val/test splits.
- **[`in-silico-test/`](in-silico-test)** — designs sequences toward target properties inside a
  structure-defined chromophore pocket and evaluates them against an independent oracle. The
  per-residue proposal comes from either ESM-2 650M or a family-MSA profile, run as a controlled
  swap (see [Findings](#findings)). It vendors its own copy of the shared model/pocket machinery
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

Oracle-scored mean absolute peak error, nm ([`3.1`](in-silico-test/3.1_design_run_MSA) vs
[`3.2`](in-silico-test/3.2_design_run_ESM2)):

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

### ESM-2 is close to uninformative on the GFP fold

The reason a 763-sequence profile keeps up is measurable
([`msa_conservation/esm_vs_family.py`](msa_conservation/esm_vs_family.py),
`results/esm_calibration.csv`). Masked single-residue recovery:

| protein | length | masked top-1 acc. | mean max prob | median rank of true residue |
|---|---|---|---|---|
| ubiquitin | 76 | 0.803 | 0.754 | 1.0 |
| lysozyme | 129 | 0.659 | 0.682 | 1.0 |
| adenylate kinase | 214 | 0.706 | 0.692 | 1.0 |
| **EGFP** | 239 | **0.126** | **0.128** | **6.0** |
| **avGFP** (wild type) | 238 | **0.101** | **0.125** | **6.0** |
| **mCherry** | 236 | **0.178** | **0.170** | **6.0** |

On ordinary proteins spanning 76–214 residues ESM-2 650M recovers the true residue as its top
choice 66–80% of the time at ~0.7 confidence; on three FPs of comparable length it drops to 10–18%
at ~0.13, with Gly on top at most positions — close to a background amino-acid distribution. Not a
length effect (adenylate kinase at 214 behaves like the short controls), not an engineering
artifact (wild-type avGFP is the worst of the three), not a harness bug (same code produces the
sharp control numbers; unmasked reconstruction of EGFP is 99.2%). Across the EGFP design window the
two priors are nearly orthogonal: mean Spearman(family frequency, ESM-2 probability) = **+0.108**,
negative at 11 of 28 positions, top-1 agreeing at **2 of 28**. *Why* is not established here.

**So the disagreement is not two opinions in conflict — it is one model having almost no opinion**,
which is what makes the cheap family profile a fair substitute for the 650M-parameter model on this
fold, and why both arms are kept rather than picking one on priors.

### The proposal is not the axis that matters — the surrogate is

Against the unguided null ([`3.3`](in-silico-test/3.3_design_run_gibbs), the same search with
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

Full write-up: [`in-silico-test/generative_models.md`](in-silico-test/generative_models.md).
Caveat when carrying this over: the `design-campaign-EGFP` MSA and ESM-2 strategies retuned λ and
temperature at the same time as the proposal, so "only the proposal differs" is true of the
benchmark and **not** of the campaign.

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
in-silico-test/
        └─ property-guided design  +  independent-oracle evaluation
           (self-contained: model/pocket code vendored into its lib/)
```

See each subfolder's `README.md` for details.

The earlier `(sequence, spectrum)` lineage — `design/build_fpbase_dataset.py` feeding
`fpbase-extractor/processed_data/ESM-spectrum/` — has been archived to
`fpbase-extractor/archive/esm_spectrum/`. The full lineage (including the earlier full-spectrum
sibling of the peak-conditioned design experiment) is:

- **[`design/`](design)** — the full-spectrum-conditioned predecessor of the peak-conditioned
  design experiment: same ESM-2 surrogate-guided design idea, but conditioned on the whole
  1002-dim ex/em curve instead of just the peaks. Superseded, kept for reference (not actively
  developed).
- **`archive/esm2_design/`** — the original peak-conditioned design experiment (campaign and
  visualization notebooks, design outputs, CV/sweep caches, checkpoints, figures). Archived and
  untracked. The `esm2_design/` folder itself is gone: its shared `peak_models.py` / `pockets.py`
  moved into [`fpdesign/`](fpdesign) and its PDBx cache into [`structures/`](structures).
  Superseded by [`in-silico-test/`](in-silico-test).
- **`archive/scalar_design/`** — the scalar-trait (brightness, pKa) counterpart of the
  peak-conditioned sweeps. Exploratory and orphaned (it imports from a sibling `peak_design/`
  path left over from an earlier layout, so it doesn't currently run); archived
  and untracked, kept for reference only.

## Design campaigns

A second layer runs concrete "recolor this scaffold toward a target FP" campaigns using the
surrogate/oracle/pocket machinery in `fpdesign/` (`peak_models.py`, `pockets.py`):

- **[`fpdesign/`](fpdesign)** — the shared library: `peak_models.py` (ESM-2 utilities +
  surrogate/oracle architectures), `pockets.py` (structure-based chromophore-pocket rules),
  `campaign.py` (`Campaign`/`CampaignConfig`, extracted from the campaign scripts),
  `build_design_windows.py`, and the shared checkpoints in `models/`. Not a campaign itself —
  the engine the campaigns below import, and also the source of the ESM-2 embedding helpers used
  by `dataset_pipeline/` and `GFP_DMS/`.
- **[`design-campaign-EGFP/`](design-campaign-EGFP)** — six parallel strategies (gibbs, guided,
  guided+constraint, brightness-guided, MSA-guided, MSA-gibbs) recoloring EGFP toward EBFP/mOrange.
  The active campaign; draws on `GFP_DMS`'s brightness classifier and `msa_conservation`'s
  alignment/PSSM.
- **[`EGFP-full-spectra/`](EGFP-full-spectra)** — the same campaign's MSA-guided strategy with the
  objective swapped from the two peak wavelengths to the **whole ex/em curve**: a `cnn-max-d1`
  surrogate over the 382-protein full-spectrum set (85/15 split, 1,222 outputs) guiding a 48-cell λ
  sweep toward mOrange. Reads the window, PSSM, pairs and brightness head straight out of
  `design-campaign-EGFP/`, so the objective is the only difference and the two are comparable.
  Revisits `design/`'s full-spectrum idea on the current curated data and machinery.

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
- **`licensing/`** — flags which curated FPs are likely patent-expired/open for commercial use
  (used to pick campaign targets). Kept local (not published in this repo — see `.gitignore`).
