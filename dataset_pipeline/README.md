# FP dataset pipeline

A unified builder and curator for the fluorescent-protein (FP) sequence→property datasets used in `esm2_fp_design`. It replaces the old split of `peak_design/build_peak_dataset.py` plus the three `build_curate_split_*` notebooks with a single code path that produces the **peak** (ex_max, em_max), **brightness**, and **pKa** datasets from one set of rules.

Everything here reads `../fpbase-extractor/fpbase_output/fpbase_proteins.json` and writes curated datasets to `data/<trait>/curated/`. The train/val/test split and visualization are deliberately **not** part of this stage — they live downstream.

## Guiding principle

Every rule below is one expression of a single idea:

> Keep a `(protein → target)` row only if the target is a well-defined function of the folded amino-acid sequence under standard conditions.

If a protein's color, brightness, or pH response comes from something the sequence doesn't encode — a bound cofactor, an added fluorogen, a supplied analyte, light activation — then its target isn't a clean label for a sequence model, and we drop or resolve it. The rules just make that judgment operational and auditable.

## Pipeline overview

The pipeline runs in five stages. Stages A and B are trait-independent and produce a shared "genuine-FP" superset; Stages C–E are applied per trait.

```
A  std-AA intake          drop non-standard residues (X/B/Z/U/O)
B  genuine-FP filters     B1 emission gate  +  B2 exogenous-signal exclusion   ── shared superset ──
C  sequence resolution    collapse each identical-sequence group to ONE row, target-aware   ┐
D  target gate            keep the resolved row only if it carries the target               ├ per trait
E  NN-4mer filter         drop sequence-isolated rows (char-4-gram cosine < 0.10)            ┘
```

Ordering matters in two places:
- **B2 before C:** exogenous proteins are removed before the nearest-neighbor step, so the NN neighborhood only ever contains genuine FPs.
- **C before D:** resolution sees every state of a sequence before the target gate throws any away — so an analyte- or activation-specific value can never be silently mislabeled onto the sequence.

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
2. **Analyte sensor** (only when the trait drops sensors — see the config table): the group is tagged with an analyte condition (`calcium`, `pH`, `acidic`, `alkaline`, `ecliptic`) *and* the target isn't consistent across its states → drop (`analyte_sensor`). A tagged group whose target *is* consistent is kept — e.g., ecliptic pHluorin has a fixed 395/509 nm peak, so it's a valid peak sample.
3. **States agree** on the target within tolerance → keep any (`dedup_identical` / `dedup_consistent` / `single_state`).
4. **Emission-separated disagreement** (emission spread ≥ 10 nm) → keep the shortest-emission state (`keep_min_emission`). Green→red photoconversion is irreversible and only red-shifts, so the bluest state is the native, un-converted precursor.
5. **Same-color on/off disagreement** for a light-controlled protein (`photoactivatable`, `photoswitchable`, `photoconvertible`) → keep the on state (`keep_on_state`): the non-dark state with the highest target value.
6. **Otherwise** → drop (`drop_ambiguous`).

**Reasoning for the sensor handling:** an analyte sensor's target depends on the environment (pH, Ca²⁺), not the folded sequence, so for peak and brightness it's ill-defined and dropped. But **pKa is state-invariant** — mKeima reports 6.5 for all five of its pH states — and it *is* the property a pKa model wants to learn, so pKa keeps sensors (rule 2 is off). The `\bbasic\b` token from the old regex is gone: it only ever caught ordinary FPs whose state is labeled "(basic)."

**Reasoning for on-state (rule 5):** a photoactivatable or photoswitchable protein's brightness is bimodal (on vs off), and the on state is the meaningful, commonly-reported value — so we keep it rather than drop the protein. Photoconvertibles are emission-separated and resolve at rule 4 to the native green precursor, which is their default on-state, keeping them consistent with the peak dataset.

### Stage D — target gate

The resolved row is kept only if it carries this trait's target. For peak this is automatic (Stage B1 guarantees ex/em); for brightness and pKa it removes sequences whose surviving state has no reported value (folded into the `no_target` count above).

### Stage E — nearest-neighbor 4-mer filter

**Rule:** represent each sequence as character 4-gram counts, take the maximum cosine similarity to any other sequence in the trait's dataset, and drop rows below 0.10.

**Reasoning:** sequence-isolated proteins have no close relatives in the set, so they're unreliable for both learning and evaluation. The filter is dataset-relative — a protein's neighborhood differs across the three traits — so it must run per trait and can't be precomputed once. It uses character 4-mers, not ESM-2 embeddings, so no model is needed.

**Excludes:** peak → BP02, CheGFP4, nanoLuc; brightness → BP02, anm1GFP1; pKa → Jred.

## Per-trait configuration

| | peak | brightness | pKa |
|---|---|---|---|
| Target | (ex_max, em_max), nm | brightness (EC × QY / 1,000) | pKa (pH units) |
| Output array | `peaks.npy` (N, 2) | `brightness.npy` (N, 1) | `pka.npy` (N, 1) |
| Drop analyte sensors | yes | yes | **no** (pKa is the signal) |
| Agreement tolerance | exact (ex, em) pair | 15% relative | 0.15 pH |
| Curated count | 758 | 533 | 368 |

## Exclusion-reason labels

Every dropped or resolved sequence is tagged with a reason, recorded in `curate_meta.json` (`dropped_names`) and in the `resolve_reason` column of the assignments CSV:

| Label | Meaning |
|---|---|
| `nonstandard_aa` | Stage A — non-standard residue |
| `exogenous_cofactor` / `_fluorogen` / `_opsin` / `_manual_irfp` | Stage B2 — bound-molecule signal |
| `no_target` | Stage C/D — no state reports the target |
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

## Usage

```bash
# one trait
python build_dataset.py --target peak --outdir data/peak/curated

# all three at once
python build_dataset.py --all

# coordinated surrogate/oracle train/val/test split for the peak (ex/em) set
python make_dual_split.py        # -> data/peak/curated/dual_splits.csv

# compare against the archived peak_design data-processing outputs
python compare.py
```

`build_dataset.py`, `make_dual_split.py`, and `compare.py` run on CPU in seconds — no ESM-2 or GPU needed.

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

`visualize_curation.ipynb` is the visual companion to the rules above: where the curated set sits in FPbase
sequence space, what each stage removes, and what the surviving targets look like. It covers

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
- **Embeddings.** Run `python embed.py` once to build the per-trait ESM-2 caches (see Usage). The
  learning-curve notebooks load these directly; if a cache is missing they fall back to building it inline,
  so `embed.py` is the clean pre-step but not strictly required.
- **Learning-curve evaluations** stay in `peak_design/` — `learning_curve.ipynb` (peak),
  `learning_curve_pka.ipynb`, `learning_curve_brightness.ipynb` — repointed to `data/<trait>/curated/` and
  its `esm_residue_*.npy` cache.
- **Stale caches to clear before re-running the learning curves:** the old sweep results computed on the
  previous data still sit in `peak_design/trained_models/` and `trained_models_brightness/`
  (`*_learning_curve_sweep.json`, `lc_models/`). Delete them so the notebooks recompute on the new data.
  (`trained_models_pka/` is already absent.)
- **Not yet repointed:** `surrogate_oracle_peak_dual.ipynb` and `guided_design_peak.ipynb` still read the
  archived `training_data/`. They'll need repointing to `dataset_pipeline/data/` before they run again.

## What changed vs. the notebook pipeline

`compare.py` diffs these outputs against the original `peak_design/training_data*/curated/` sets. Every difference is an intended, documented consequence of the rules above:

- **peak (758, unchanged):** ecliptic pHluorin is now kept (its peak is unambiguous); iFP2.0 is now dropped (biliverdin).
- **brightness (533 vs. 534):** pHluorin4 is kept (its brightness is pH-invariant — it's ratiometric) and PSLSSmKate is kept at its on state; mKeima, pHmScarlet (on/off sensors) and iFP2.0 are dropped.
- **pKa (368 vs. 364):** mKeima, pHluorin4, CAR-GECO1, and PSLSSmKate are now kept — sensors are valid pKa examples, and their pKa is well-defined.

The refactor also surfaced two flaws in the old pipeline: it retained iFP2.0 (an untagged biliverdin IR-FP) and it silently kept KFP1's bright-state brightness while dedup-ing on the peak. Both are now handled explicitly.

## Note on the biliverdin / phytochrome class

The biggest exogenous group is the biliverdin near-infrared FPs (43 proteins: 39 cofactor-tagged plus iFP2.0 and three peak-less phytochromes). These are engineered from bacterial and cyanobacterial **phytochromes**, which don't build a chromophore from their own residues the way GFP-family proteins do — they covalently bind a **bilin** (a linear tetrapyrrole from heme breakdown). The long conjugated bilin is why they emit in the far red (670–720 nm), and the fact that the chromophore is a supplied cofactor is why they don't belong in a sequence→spectrum dataset. Source organism is the cleanest discriminator (phytochromes come from *Rhodopseudomonas*, *Deinococcus*, *Nostoc*, etc.; GFP-family reds come from cnidarians), but on the current data the cofactor tag plus a one-name exclusion for iFP2.0 covers the class exactly.

## Contact

Rules and thresholds here encode specific biological judgment calls (sensor handling, on-state policy, the NN cutoff). If you change one, rerun `compare.py` and check the diff — the deltas should always be explainable in terms of the principle at the top of this file.
