# ESM-spectrum dataset (FPbase)

Processed (sequence, excitation+emission spectrum) pairs for the ESM-2
property-guided design pipeline. Built by `esm2_seq_design/build_fpbase_dataset.py`
from the fpbase-extractor export.

## Contents

| File | Shape / format | Description |
|------|----------------|-------------|
| `spectra.npy` | float32 `(453, 1002)` | Per sample: excitation curve `[0:501]` then emission curve `[501:1002]`, each peak-normalized to 1.0 |
| `grid_nm.npy` | float32 `(501,)` | Wavelength grid, 300–800 nm @ 1 nm; indexes **each** half of `spectra.npy` |
| `sequences.fasta` | 453 records | `>slug|state` headers; sequence per sample (row-aligned to `spectra.npy`) |
| `metadata.csv` | 453 rows | Row-aligned metadata incl. `seq`; columns below |
| `meta.json` | — | Provenance, counts, grid, ex/em split index |

`metadata.csv` columns: `index, slug, name, state, seq_group, parent_organism,
switch_type, oligomerization, is_dark, ex_max, em_max, ref_year, seq_len, aliases, seq`.

- **453 samples**, **430 unique sequences** (23 are multi-state duplicates of a
  sequence; `seq_group` gives the identical-sequence group id).
- ex_max 338–702 nm, em_max 414–720 nm; seq length 106–476.

## Load

```python
import numpy as np, csv
spectra = np.load("spectra.npy")          # (N, 1002)
grid    = np.load("grid_nm.npy")          # (501,)
rows    = list(csv.DictReader(open("metadata.csv")))
sequences = [r["seq"] for r in rows]
```

To wire into `guided_design_approach1.ipynb` Section 1: set `GRID = grid`,
`SPECTRUM_DIM = spectra.shape[1]`, `sequences = [...]`, `spectra = spectra`.

## Train/test split — candidate schemes (NOT yet applied)

Many FPbase entries are derivatives of a handful of ancestors (avGFP, DsRed,
mScarlet, …), so a naive random split leaks near-identical sequences across
train/test and inflates metrics. Options, roughly increasing in rigor:

1. **Random (baseline only).** Easy, but optimistic — use only as an upper bound.
2. **Identity-cluster split (recommended default).** Cluster sequences (CD-HIT /
   MMseqs2, or greedy single-linkage on pairwise % identity) at e.g. 70–90%
   identity; assign whole clusters to train or test. Start by collapsing exact
   duplicates via `seq_group`.
3. **Family / lineage split.** Group by ancestor using name roots + `aliases`
   (e.g. all `mScarlet*`, all avGFP-derived greens) and hold out whole families.
   Coarser than clustering but interpretable.
4. **Organism split.** Hold out whole `parent_organism` groups — tests transfer
   across source organisms.
5. **Phenotype-extrapolation split (stress test).** Hold out a spectral region
   (e.g. train on `em_max < 580`, test red-shifted) to measure generalization to
   novel spectra rather than interpolation.
6. **Temporal split.** Train on older `ref_year`, test newer — mimics prospective
   discovery.

Suggested: report **(2) identity-cluster** as the primary metric and **(5)
phenotype-extrapolation** as the honest stress test; keep (1) random as a ceiling.
