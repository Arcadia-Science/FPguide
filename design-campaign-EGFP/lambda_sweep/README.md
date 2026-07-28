# lambda_sweep/ — how the four guidance weights trade off (EGFP → mOrange & EBFP)

A 3 x 3 x 3 grid over the **DMS-guide** strategy's guidance weights, holding everything else fixed.
The strategy itself is unchanged from
[`../brightness-guided/guided_design/`](../brightness-guided/guided_design/): same EGFP scaffold
(`4EUL`, idx 171), same Tier-B 5 Å chromophore window (25 editable positions, pos2 restricted to
aromatics, H-bond partners restricted to H-bond-capable residues), same surrogate (train MAE 5.2 nm),
same `cnn-max-d2` brightness classifier (val AUROC 0.982), `T=10`, `k=10`, 12 trials x 3 iterations.

| axis | values | what it weights |
|---|---|---|
| `λ_ex = λ_em` | 10, 20, 30 | pull toward the target's excitation / emission peaks |
| `λ_bright` | 40, 50, 60 | pull toward the brightness classifier's positive class |
| `λ_edit` | 10, 15, 20 | penalty per position that differs from the scaffold |

27 cells x **both campaign targets**, which bracket the difficulty:

- **mOrange** — the far shift, ~30 % identity to EGFP, `(488, 507) → (548, 562)` nm, scaffold ~57 nm off.
- **EBFP** — near in *sequence* but the larger spectral jump, `(488, 507) → (380, 440)` nm, scaffold
  ~88 nm off.

Each target was launched as its own pass (`PAIRS=...`) and writes `design_EGFP-<target>.csv` into the
same 27 cell folders; the engine's cache check is per pair, so the two passes do not interfere.

Every term is **z-scored across the candidate set before its λ is applied**
(`fpdesign/campaign.py::_zc`), so these numbers are in units of "standard deviations of that term
across the k=10 candidates at one position", and they are all divided by `T=10` inside the softmax.
That is why the useful range is 10–60 and not ~1: at λ≈1 the summed score spans only a few units and
the sampler is ~96 % of the way to uniform. See the "unit-scale probes" in
[`../archive/README.md`](../archive/README.md) for that negative control.

## Layout

```
design_windows_egfp_tierB.json                     the Tier-B window (copy of the campaign's)
design_campaign.py                                 driver: per-cell outdir, no pseudo-perplexity
run_sweep.sh                                       sequential loop over the 27 cells, one target
import_existing.sh                                 pulls in the 5 cells already run elsewhere
visualize_sweep.ipynb                              scores every cell and plots the grid as heatmaps
sweep_metrics.csv                                  per-cell metrics cache written by the notebook
sweep_designs.csv                                  per-design metrics cache (gates, error, mutations)
figures/sweep_<metric>_<target>.png                heatmaps, + sweep_designs_gates.png / sweep_tradeoff.png
designs/lam-ex{P}_lam-em{P}_lam-bright{B}_lam-edit{E}/design_EGFP-{mOrange,EBFP}.csv
logs/sweep_<timestamp>.log                         one log per pass (.last_log points at the newest)
```

Reproduce, from this folder:

```bash
bash import_existing.sh                  # optional: reuse the 5 pre-existing cells
bash run_sweep.sh                        # mOrange, ~34 min on one GPU; complete cells are skipped
PAIRS=EBFP bash run_sweep.sh             # EBFP, ~36 min
jupyter lab visualize_sweep.ipynb
```

[`visualize_sweep.ipynb`](visualize_sweep.ipynb) heatmaps mutation load, `|Δex|`/`|Δem|`/mean peak
error, and the in-distribution and brightness hit rates across the grid (one panel per `λ_ex/λ_em`
level, rows `λ_bright`, columns `λ_edit`). It then plots cost against accuracy twice — once with **every
individual design** coloured by which gates it passed, carrying two Pareto frontiers (all designs vs
gate-passing only, whose gap is what the screen costs in nm), and once aggregated to one point per
setting for choosing λs — and prints the per-axis marginals and per-target leaderboards. Every figure is
drawn **once per target with its own colour scale**, since the two scaffold errors differ by ~30 nm and a
shared scale would flatten both: ranks are comparable across targets, colours are not. Its ESM-2 scoring
pass is cached in `sweep_metrics.csv` (per cell) and `sweep_designs.csv` (per design) and is
incremental: cells already in the cache are skipped, so the notebook is cheap to re-run while the
sweep is still filling cells in (a cell whose CSV has since grown is re-scored, because the trial
count is part of the cache key). Set `RESCORE = True` to rebuild from scratch.

## Three things to know before comparing cells

**1. The folder name is the authoritative record of the weights.** The engine writes `lam_ex`,
`lam_em` and `lam_bright` columns but has **no `lam_edit` column**, so only the folder name carries
all four. The notebook therefore parses the folder name and cross-checks it against the columns that
do exist, reporting any mismatch.

This is also why the driver names its own output folder instead of using the engine's
`outdir_lambda_suffix`: that suffix encodes only `λ_bright` and `λ_edit`, so the three `λ_ex/λ_em`
levels would have collided in one folder — and since `existing_pair` validates only
trials/rounds/temp/k and never the λs, the later cells would have been silently skipped as "cached"
instead of run. The sweep would have quietly produced a 9-cell grid labelled as 27.

**2. The `ppl` column is deliberately blank.** Pseudo-perplexity costs one masked forward pass per
residue per sequence — about 49 s of what would otherwise be a 129 s cell, more than a whole design
iteration — and nothing in this campaign reads it. It also sat flat at 16.2–16.7 across every setting
explored so far, so it carries no signal here. `SweepCampaign.ppl_batched` returns `NaN` and the
engine leaves the cell empty. Fill it in later with `--backfill-ppl` if it is ever wanted.

**3. Five cells were imported, not re-run**, since identical settings had already been executed. All
sit at `λ_ex=λ_em=20`, the only level the earlier campaign used:

| cell | targets | trials | source |
|---|---|---|---|
| `..._lam-bright40_lam-edit10` | mOrange, EBFP | 24 | `../archive/brightness-guided/guided_design/designs_lam-bright40_lam-edit10/` |
| `..._lam-bright60_lam-edit10` | mOrange, EBFP | 96 | `../brightness-guided/guided_design/designs_lam-bright60_lam-edit10/` (the campaign's kept DMS-guide run) |
| `..._lam-bright60_lam-edit20` | mOrange only | 48 | `../archive/brightness-guided/guided_design/designs_lam-bright60_lam-edit20/` |

They came from the same driver, scaffold, window, pairs CSV, `T`, `k` and `λ_ex=λ_em=20` at `iters=3`,
so they are byte-for-byte copies. Two differences to keep in mind: they hold **more trials** than the
sweep's 12 and their **`ppl` column is populated**. The notebook truncates to the first 12 trials so
all cells are compared at equal depth — and because the engine seeds RNG **per trial**
(`per_trial_rng=True`), trials 0–11 of a deeper run are bit-identical to a 12-trial run, making that
truncation exact rather than a subsample. Set `TRIALS = 0` in the notebook to use every trajectory
available instead.
