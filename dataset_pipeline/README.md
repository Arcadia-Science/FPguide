# FP dataset pipeline

A unified builder and curator for the fluorescent-protein (FP) sequence→property datasets used in `esm2_fp_design`. The actively maintained datasets are **peak** (ex_max, em_max) and **spectra** (full excitation/emission curves); **brightness** and **pKa** were built by the same code path but are now archived (see [Archived: brightness & pKa](#archived-brightness--pka)).

Everything here reads from `../fpbase-extractor/fpbase_output/`. `build_dataset.py` reads `fpbase_proteins.json` and writes the peak (and, on request, brightness/pKa) dataset to `data/<trait>/curated/`. `build_spectra_dataset.py` additionally cross-references `fpbase_spectra.json` and writes to `data/spectra/curated/`. The train/val/test split and visualization are deliberately **not** part of this stage — they live downstream.

## Guiding principle

Every rule below is one expression of a single idea:

> Keep a `(protein → target)` row only if the target is a well-defined function of the folded amino-acid sequence under standard conditions.

If a protein's color, brightness, or pH response comes from something the sequence doesn't encode — a bound cofactor, an added fluorogen, a supplied analyte, light activation — then its target isn't a clean label for a sequence model, and we drop or resolve it. The rules just make that judgment operational and auditable.

## Pipeline overview — peak

The peak dataset runs in five stages. Stages A and B produce a "genuine-FP" superset; Stages C–E resolve it down to one row per sequence.

```
A  std-AA intake          drop non-standard residues (X/B/Z/U/O)
B  genuine-FP filters     B1 emission gate  +  B2 exogenous-signal exclusion   ── shared superset ──
C  sequence resolution    collapse each identical-sequence group to ONE row, target-aware
D  target gate            keep the resolved row only if it carries the target
E  NN-4mer filter         drop sequence-isolated rows (char-4-gram cosine < 0.10)
```

Ordering matters in two places:
- **B2 before C:** exogenous proteins are removed before the nearest-neighbor step, so the NN neighborhood only ever contains genuine FPs.
- **C before D:** resolution sees every state of a sequence before the target gate throws any away — so an analyte- or activation-specific value can never be silently mislabeled onto the sequence.

The full-spectrum dataset (`build_spectra_dataset.py`) does not re-run this pipeline — it *filters* its output, keeping only the peak rows whose resolved state has a measured excitation+emission curve pair. See [Full-spectrum dataset (spectra)](#full-spectrum-dataset-spectra) below.

## Stage-by-stage rules

### Stage A — standard-AA intake

**Rule:** keep only sequences composed entirely of the 20 standard amino acids; drop any containing `X`, `B`, `Z`, `U`, or `O`.

**Reasoning:** this is a data-quality choice, *not* an ESM-2 limitation — ESM-2 tokenizes `X/B/Z/U/O` as real tokens. But `X` marks an unknown residue and `B`/`Z` are ambiguity codes (Asx, Glx), so these sequences carry unresolved positions we'd rather not train on.

**Excludes:** 14 proteins (32 `X`, 3 `B`, 1 `Z`; no `U`/`O` present) — e.g., acanFP, cgigGFP, echFP, StayRose. A further 51 entries have no sequence at all and are skipped here too.

### Stage B1 — emission gate

**Rule:** a state is kept only if it reports both `ex_max` and `em_max`.

**Reasoning:** requiring an emission peak is the concrete test for "is this actually fluorescent." It excludes non-emissive chromoproteins (absorption only) and garbage rows — e.g., Channelrhodopsin2's placeholder `em_max` of 1,000 nm.

**Excludes:** 138 proteins whose states all lack a reported peak.

### Stage B2 — exogenous-signal exclusion

**Rule:** drop a protein if its signal comes from a bound molecule rather than the folded sequence. Four sub-rules, each logged separately:

| Sub-rule | Signal | How it's detected | Count |
|---|---|---|---|
| `cofactor` | biliverdin, flavin, bilirubin, phycocyanobilin, etc. | FPbase `cofactor` field is non-empty (BV/FL/BR/PC/RL) | 63 |
| `fluorogen` | added HBR/HMBR-type dyes | name or aliases match `\b(?:fr\|nir\|p)?FAST\b` (case-sensitive) | 4 |
| `opsin` | retinal | name or aliases match `channelrhodopsin\|rhodopsin\|opsin\|ChR\d?` | 1 |
| `manual_irfp` | biliverdin, untagged | explicit name list (`iFP2.0`) | 1 |

**Reasoning:**
- **Cofactor / fluorogen / opsin** are all the same category — the chromophore is a molecule the cell or the experimenter supplies (a bilin, a fluorogenic dye, retinal), so the spectrum isn't encoded by the sequence. See [`docs on the biology`](#note-on-the-biliverdin--phytochrome-class) below for why the bilin class matters.
- **FAST detection is case-sensitive** because the genuine fluorogen tags are always written all-caps (FAST, pFAST, frFAST, nirFAST). A case-insensitive match wrongly caught Fast-FT (a timer), sf:fast.3 (a designed variant), and ffDronpa (via a lowercase-"fast" alias).
- **Fluorogen/opsin matching uses name and aliases only, never the source organism** — matching organism caught the whole FAST family through its host *Halorhodospira halophila* ("…rhodo…").
- **`iFP2.0` is dropped by name** rather than by a broad organism rule. It's a biliverdin IR-FP from *Deinococcus radiodurans* that FPbase left untagged. It's the only untagged bilin protein with spectra that would otherwise leak in — the other untagged phytochromes (hfriFP, RpBphP2, RpBphP6) report no peaks and are already removed by Stage B1.

**Excludes:** 69 proteins total.

### Stage C — sequence resolution (target-aware collapse)

A sequence can appear under several states (a photoconvertible's green and red forms, a sensor's pH series, an on/off switch). A per-sequence regressor needs exactly one target per sequence, and a sequence sitting in multiple states would both give inconsistent labels and leak across the split. Stage C collapses each identical-sequence group to one row, using **this trait's target** and running before the target gate.

The decision for a group, in order:

1. **No state carries the target** → drop (`no_target`).
2. **Analyte sensor** (only when the trait drops sensors): the group is tagged with an analyte condition (`calcium`, `pH`, `acidic`, `alkaline`, `ecliptic`) *and* the target isn't consistent across its states → drop (`analyte_sensor`). A tagged group whose target *is* consistent is kept — e.g., ecliptic pHluorin has a fixed 395/509 nm peak, so it's a valid peak sample.
3. **States agree** on the target within tolerance → keep any (`dedup_identical` / `dedup_consistent` / `single_state`).
4. **Emission-separated disagreement** (emission spread ≥ 10 nm) → keep the shortest-emission state (`keep_min_emission`). Green→red photoconversion is irreversible and only red-shifts, so the bluest state is the native, un-converted precursor.
5. **Same-color on/off disagreement** for a light-controlled protein (`photoactivatable`, `photoswitchable`, `photoconvertible`) → keep the on state (`keep_on_state`): the non-dark state with the highest target value.
6. **Otherwise** → drop (`drop_ambiguous`).

Peak drops analyte sensors (rule 2 is on). This is also why the spectra dataset filters peak's *output* rather than re-resolving from the spectra-bearing population directly: an earlier version did the latter and it let a couple of analyte sensors back in, because restricting resolution to only the curve-bearing states can hide the very disagreement across states that this rule is meant to catch (see [Full-spectrum dataset (spectra)](#full-spectrum-dataset-spectra)).

### Stage D — target gate

The resolved row is kept only if it carries this trait's target. For peak this is automatic (Stage B1 guarantees ex/em).

### Stage E — nearest-neighbor 4-mer filter

**Rule:** represent each sequence as character 4-gram counts, take the maximum cosine similarity to any other sequence in the trait's dataset, and drop rows below 0.10.

**Reasoning:** sequence-isolated proteins have no close relatives in the set, so they're unreliable for both learning and evaluation. The filter is dataset-relative — a protein's neighborhood differs across traits — so it must run per trait and can't be precomputed once. It uses character 4-mers, not ESM-2 embeddings, so no model is needed.

**Excludes:** peak → BP02, CheGFP4, nanoLuc. (The spectra dataset doesn't run its own Stage E — it filters peak's already-NN-filtered output, so this exclusion is inherited rather than repeated.)

## Peak dataset configuration

| | Peak |
|---|---|
| Target | (ex_max, em_max), nm |
| Output array | `peaks.npy` (N, 2) |
| Drop analyte sensors | yes |
| Agreement tolerance | exact (ex, em) pair |
| Curated count | 758 |

## Exclusion-reason labels

Every dropped or resolved sequence is tagged with a reason, recorded in `curate_meta.json` (`dropped_names`) and in the `resolve_reason` column of the assignments CSV:

| Label | Meaning |
|---|---|
| `nonstandard_aa` | Stage A — non-standard residue |
| `exogenous_cofactor` / `_fluorogen` / `_opsin` / `_manual_irfp` | Stage B2 — bound-molecule signal |
| `no_target` | Stage C/D — no state reports the target |
| `no_spectra_measured` | spectra only — protein has no measured curves at all |
| `no_curve_pair_for_resolved_state` | spectra only — has measured curves, but not both types for the specific state peak resolved to |
| `analyte_sensor` | Stage C — analyte-modulated, target inconsistent |
| `drop_ambiguous` | Stage C — inconsistent target, not resolvable |
| `dedup_identical` / `dedup_consistent` / `single_state` | Stage C — kept; states agree |
| `keep_min_emission` | Stage C — kept; native precursor of a photoconversion |
| `keep_on_state` | Stage C — kept; on state of an on/off switch |
| `nn_isolated` | Stage E — no near neighbor |

## Outputs

Per trait, in `data/<trait>/curated/`:

- `<target>.npy` — target array, row-aligned to the FASTA and CSV
- `sequences.fasta` — one record per row, header `>index|slug|state`
- `<target>_assignments.csv` — full metadata table, including `resolve_reason` and `n_states_in_group`
- `curate_meta.json` — provenance: parameters, per-stage counts, and the dropped-name lists by reason

The spectra dataset's outputs follow the same conventions but swap the single target array for a curve pair — see below.

## Full-spectrum dataset (spectra)

`build_spectra_dataset.py` curates (sequence → full excitation curve, full emission curve) pairs — the measured spectral shape, not just its peak — by **filtering the already-curated peak dataset**, not by re-running the pipeline above against FPbase directly.

An earlier version of this script did re-curate independently from the 455-protein spectra-bearing population, reusing Stage A/B2/C/E but swapping the scalar emission gate for a curve-completeness gate. That produced 384 rows, but 3 of them (mKeima, pHmScarlet, GRvT) shouldn't have been there: they're a pH sensor, a pH sensor, and a FRET biosensor whose ex/em disagree wildly across their several reported states, and peak's own Stage C correctly drops all three as `analyte_sensor`/`drop_ambiguous`. The bug was that FPbase only ever measured a *complete* excitation+emission curve pair for one of each protein's states, so restricting resolution to curve-bearing states left only one candidate — nothing to disagree with, so the sensor/ambiguity check had nothing to catch.

The fix is to not re-derive that judgment at all: peak's Stage C already saw every reported state for these proteins and made the right call. So this script instead:

1. Reads `data/peak/curated/peaks_assignments.csv` (758 rows; run `build_dataset.py --target peak` first).
2. For each row, looks up its slug in `fpbase_spectra.json` and checks whether the **specific state peak resolved to** (the `state` column, matched back to FPbase's spectra-file state labels the same way as before — stripping the protein-name prefix and outer parentheses, unambiguous for all 455 spectra-bearing proteins) has both a measured excitation curve and a measured emission curve.
3. Keeps the row (with those two curves) if so, drops it otherwise.

No sequence resolution, sensor handling, or NN filtering happens in this script — all of that is peak's, inherited by construction. Of the 758 peak rows, 375 have no measured spectra at all and 1 (AvicFP3) has curves but not for its resolved state, leaving **382 curated rows** — matching the simpler "peak ∩ has full spectra" count already used as an informational check in `visualize_curation.ipynb`.

**Resampling onto a shared grid.** Raw curves are 1 nm-stepped but span different, protein-specific wavelength ranges. Each curve is linearly interpolated onto a fixed grid, zero-filled outside its own measured domain:

| | Excitation | Emission |
|---|---|---|
| Grid | 230–800 nm, 1 nm steps (571 points) | 250–900 nm, 1 nm steps (651 points) |
| Coverage | lossless — spans every curated excitation curve's full measured range | <0.5% of any curve's intensity mass falls outside this range (checked against the curated set; a handful of far-red proteins carry a long near-zero tail out past 900 nm that isn't worth padding for) |

**Outputs**, in `data/spectra/curated/`:

- `excitation.npy` (382, 571) float32 — resampled excitation curves
- `emission.npy` (382, 651) float32 — resampled emission curves
- `excitation_wavelengths.npy` (571,) / `emission_wavelengths.npy` (651,) float32 — the wavelength grids the arrays are sampled on
- `sequences.fasta`, `spectra_assignments.csv` (peak's own assignment columns, subset to these rows), `curate_meta.json` — same conventions as the peak dataset

```bash
python build_dataset.py --target peak --outdir data/peak/curated   # if not already built
python build_spectra_dataset.py --outdir data/spectra/curated
```

## Archived: brightness & pKa

Brightness and pKa were built by the same `build_dataset.py` code path as peak, but are no longer part of the actively maintained pipeline. Their previously generated data (curated arrays, FASTA, assignments CSV, `curate_meta.json`, and splits) now live under `archive/data_brightness/` and `archive/data_pka/` rather than `data/brightness/`/`data/pka/`.

For reference, their configuration differed from peak as follows:

| | Brightness | pKa |
|---|---|---|
| Target | brightness (EC × QY / 1,000) | pKa (pH units) |
| Drop analyte sensors | yes | **no** (pKa is the signal — mKeima reports 6.5 across all five of its pH states, and that state-invariance is exactly the property a pKa model wants to learn) |
| Agreement tolerance | 15% relative | 0.15 pH units |
| Curated count | 533 | 368 |

`build_dataset.py --target brightness` / `--target pka` still work and will regenerate these datasets from current FPbase data if ever needed; `--all` no longer includes them (it only builds peak) so it doesn't silently regenerate data the project has moved away from.

Two historical notes worth keeping: pHluorin4 and PSLSSmKate were newly kept in brightness (ratiometric/pH-invariant and on-state respectively) while mKeima, pHmScarlet, and iFP2.0 were newly dropped; mKeima, pHluorin4, CAR-GECO1, and PSLSSmKate were newly kept in pKa, since sensors are valid pKa examples. The refactor that produced this pipeline also fixed a bug where KFP1's bright-state brightness had silently been kept while dedup-ing on the peak value. Full detail is preserved in each archived `curate_meta.json`.

## Usage

```bash
# peak (the maintained target)
python build_dataset.py --target peak --outdir data/peak/curated
python build_dataset.py --all             # same thing, peak only

# full-spectrum dataset
python build_spectra_dataset.py --outdir data/spectra/curated

# regenerate an archived dataset if needed (writes back under data/<trait>/curated, not archive/)
python build_dataset.py --target brightness --outdir data/brightness/curated
python build_dataset.py --target pka        --outdir data/pka/curated

# coordinated surrogate/oracle train/val/test split for the peak (ex/em) set
python make_dual_split.py        # -> data/peak/curated/dual_splits.csv

# compare against the archived peak_design data-processing outputs
python compare.py
```

`build_dataset.py`, `build_spectra_dataset.py`, `make_dual_split.py`, and `compare.py` run on CPU in seconds — no ESM-2 or GPU needed.

Embedding is the one heavy step and is its own script:

```bash
# embed each trait's curated sequences with ESM-2, cached per trait (needs GPU/MPS + ESM-2 weights)
python embed.py                  # all three; skips traits already cached
python embed.py --dry-run        # report N / Lmax / cache size without loading ESM-2
python embed.py --trait pka --force
```

`embed.py` writes `esm_residue_fp16.npy` + `esm_residue_len.npy` into each `data/<trait>/curated/`, embedding
every trait **independently** (a shared sequence is embedded once per trait). That wastes a little compute but
keeps each dataset self-contained. These are exactly the caches the learning-curve notebooks load.

## Visualizing the curation

`visualize_curation.ipynb` is the visual companion to the peak rules above: where the curated set sits in FPbase
sequence space, what each stage removes, and what the surviving targets look like. (It predates the spectra
dataset and doesn't cover it yet.) It covers

1. a t-SNE of a max-pooled ESM-2 embedding of all 990 sequenced FPbase proteins, with the 758 curated rows
   highlighted and the major lineages labelled by ancestral organism,
2. the curation funnel plus a complete by-reason breakdown of all 283 dropped proteins,
3. excitation vs. emission for the 758, with marginal histograms,
4. PCA on the (ex, em) targets — PC1 is overall colour, PC2 is the Stokes shift,
5. PCA on the sequence embeddings of the 382 curated proteins that also have measured spectral curves.

It reads `curate_meta.json` directly, so the funnel can't drift away from the data on disk. Sections 1 and 5
need a whole-FPbase embedding cache, which is its own script:

```bash
python embed_fpbase_maxpool.py           # ~1.5 min on MPS; skips if a valid cache exists
python embed_fpbase_maxpool.py --force   # rebuild
```

This writes `data/fpbase_esm2_650M_max.npy` (990 × 1280) plus a metadata sidecar carrying an md5 over the
sequence list, which the notebook asserts against so a stale cache can never be silently misaligned. Unlike
`embed.py`, it **max**-pools rather than mean-pools and covers the *whole* export — including the cofactor,
FAST and opsin proteins the pipeline drops — so the map can show where the excluded families sit. The cache is
gitignored; regenerate it rather than committing it.

Section 5 also reads `../fpbase-extractor/fpbase_output/fpbase_spectra.json` to find which curated proteins
have measured excitation and emission curves.

### Figure style

Figures follow the 2026 Arcadia style guide, which `arcadia_pycolor` already encodes: 15pt Medium axis titles,
15pt Regular labels, 14.5pt Atkinson Hyperlegible Mono numerals, 0.75pt black axis lines with 5px ticks, and
17pt SemiBold key titles over a 1.5pt Chateau rule. Charts are titled in the markdown caption rather than in
the figure, and panels are sized in the guide's pixels (1000 / 650 / 490 wide) — `arcadia_pycolor` pins the
figure DPI to 72, so one matplotlib point is exactly one style-guide pixel.

The guide's weights ship from Google Fonts only as *variable* fonts, and matplotlib cannot select a position
on a variable weight axis. One script cuts the needed weights into static faces:

```bash
python fetch_arcadia_fonts.py           # a few seconds; skips if the faces already exist
python fetch_arcadia_fonts.py --force   # re-download and rebuild
```

This writes `fonts/` next to the script — nothing is installed into the system font directories — and the
notebook loads it with `apc.mpl.setup(font_dirpath="fonts")`, falling back to matplotlib's defaults (with a
printed note) if it is missing. The fonts are OFL-licensed and gitignored; regenerate rather than commit them.

Emission wavelength is shaded continuously on the Arcadia **magma** gradient, one of the guide's sequential
gradients for dots, with the palest end trimmed so the reddest proteins stay distinct from the page. A single
scale spans the full curated emission range, so a colour means the same thing in every figure.

One departure from the library, flagged in the notebook: keys are drawn by a local helper rather than
`apc.mpl.add_legend_line`, which sizes its rule in display pixels while drawing in points (so the rule
overruns at any DPI other than 72) and inserts it into the legend's first column only (which stretches that
column when a key has more than one).

The notebook was generated by `_build_visualize_curation.py`; edit the `.ipynb` directly from here on.

## Downstream (peak_design)

The old data-processing code and data in `peak_design/` were archived to
`peak_design/archive/data_processing/` once this pipeline replaced them. What remains wired up:

- **Peak surrogate/oracle split.** `make_dual_split.py` writes `data/peak/curated/dual_splits.csv` using the
  same logic as before (seed 0, 70/15/15, coordinated so `S_test ⊆ O_train` and `O_test ⊆ S_train`).
- **Embeddings.** Run `python embed.py` once to build the peak ESM-2 cache (see Usage). The
  learning-curve notebooks load these directly; if a cache is missing they fall back to building it inline,
  so `embed.py` is the clean pre-step but not strictly required.
- **Learning-curve evaluation** stays in `peak_design/` — `learning_curve.ipynb` — repointed to
  `data/peak/curated/` and its `esm_residue_*.npy` cache. (`learning_curve_brightness.ipynb` and
  `learning_curve_pka.ipynb` evaluate the now-archived datasets; repoint them at
  `archive/data_brightness/`/`archive/data_pka/`, or regenerate fresh data per the Usage section above, before
  relying on them.)
- **Not yet repointed:** `surrogate_oracle_peak_dual.ipynb` and `guided_design_peak.ipynb` still read the
  archived `training_data/`. They'll need repointing to `dataset_pipeline/data/` before they run again.

## What changed vs. the notebook pipeline

`compare.py` diffs the peak output against the original `peak_design/training_data/curated/` set. peak is
unchanged at 758 rows: ecliptic pHluorin is now kept (its peak is unambiguous) and iFP2.0 is now dropped
(biliverdin). The equivalent brightness/pKa deltas versus their old notebook-pipeline outputs are recorded in
[Archived: brightness & pKa](#archived-brightness--pka) above, since those datasets are no longer generated by
default.

## Note on the biliverdin / phytochrome class

The biggest exogenous group is the biliverdin near-infrared FPs (43 proteins: 39 cofactor-tagged plus iFP2.0 and three peak-less phytochromes). These are engineered from bacterial and cyanobacterial **phytochromes**, which don't build a chromophore from their own residues the way GFP-family proteins do — they covalently bind a **bilin** (a linear tetrapyrrole from heme breakdown). The long conjugated bilin is why they emit in the far red (670–720 nm), and the fact that the chromophore is a supplied cofactor is why they don't belong in a sequence→spectrum dataset. Source organism is the cleanest discriminator (phytochromes come from *Rhodopseudomonas*, *Deinococcus*, *Nostoc*, etc.; GFP-family reds come from cnidarians), but on the current data the cofactor tag plus a one-name exclusion for iFP2.0 covers the class exactly.

## Contact

Rules and thresholds here encode specific biological judgment calls (sensor handling, on-state policy, the NN cutoff). If you change one, rerun `compare.py` and check the diff — the deltas should always be explainable in terms of the principle at the top of this file.
