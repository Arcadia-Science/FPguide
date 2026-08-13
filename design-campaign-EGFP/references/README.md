# `references/` — the scaffold + target reference rows, checked in

Every shortlist xlsx opens with two **reference** rows: the EGFP scaffold and the case's target,
carrying their TRUE measured peaks and full sequences. `make_shortlist_case.py` reads them for
each case, and `benchmark_report.py` reads the target peaks to score its pools.

Those six fields used to be lifted out of `.iloc[0]` of whichever design CSV happened to be
convenient — first `guided-design/`, then `brightness-guided/`. When those runs were retired to
`archive/` (gitignored), **every** shortlist case broke, including the four MSA ones that have
nothing to do with the retired strategies. The fields are run-invariant metadata, so tying them to
a run tree was always the wrong coupling.

They now live here, in two ~1 KB tracked CSVs, so a shortlist can be rebuilt from a fresh clone
without any design run present.

| file | scaffold | target | peaks |
|---|---|---|---|
| `reference_EGFP-mOrange.csv` | EGFP (239 aa) | mOrange (236 aa) | 488/507 → 548/562 |
| `reference_EGFP-EBFP.csv` | EGFP (239 aa) | EBFP (239 aa) | 488/507 → 380/440 |

Columns: `scaffold_name, target_name, scaffold_seq, target_seq, scaffold_ex, scaffold_em,
target_ex, target_em`. Peaks are bare integers, matching the `int(...)`/`float(...)` reads in
`make_shortlist_case.build`.

## Provenance

Extracted from `msa-guided/designs/lam-ex0.25_lam-em0.25_lam-bright0.5_lam-edit0/design_EGFP-<target>.csv`
row 0. Before extraction, all six fields were confirmed **identical** across every source that has
ever carried them, for both targets:

- `archive/superseded-unmatched-runs/guided-design/designs/` (the old mOrange source)
- `archive/superseded-unmatched-runs/brightness-guided/guided_design/designs_lam-bright60_lam-edit10/`
  (the old EBFP source)
- all 125 `msa-guided/designs/` cells
- all 125 `esm2_guided/designs/` cells

So repointing `REF_CSV` here could not perturb the frozen shortlists — and it did not: all four
rebuilt byte-for-byte identically afterwards.

## Keeping it honest

```bash
python make_shortlist_case.py --verify-refs
```

re-compares these rows against every live design CSV that carries them and fails loudly on any
mismatch. Run it if a design run is ever regenerated against a different pairs CSV.

The upstream source of truth is `pairs/campaign_pairs_egfp.csv` plus the FP dataset — that CSV
records the scaffold/target **indices** (`scaffold_idx=171`, `target_idx=156/479`) but not their
sequences or peaks, which is why these files are extracted from a run rather than from it.
