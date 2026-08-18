# fpdesign — the shared design/modelling library

Not an experiment. This is the engine the design campaigns import: the ESM-2 utilities, the
surrogate/oracle architectures, the structure-based pocket rules, the campaign loop, and the two
checkpoints every campaign loads. Nothing here produces results on its own.

## Modules

| module | what it is |
|---|---|
| [`peak_models.py`](peak_models.py) | ESM-2 per-residue embedding helpers plus the peak-wavelength surrogate/oracle architectures — CNN and Transformer backbones over a `mean \| min \| max \| std \| concat \| concatstd \| attn \| cov` masked-pooling readout. `load_model` / `wrap` reconstruct a checkpoint identically wherever it is loaded, which is why saved weights are portable across folders. Regresses the two peaks `(ex_max, em_max)`, not the full curve. |
| [`pockets.py`](pockets.py) | the chromophore edit window from an **experimental** RCSB structure: the chromophore tripeptide plus every residue with a heavy atom within 5 Å, mapped onto the dataset sequence by alignment. `experimental_window()` is the generalized entry point, with a local-alignment quality gate so a structure that doesn't cleanly contain the FP is rejected rather than silently mis-mapped. Reads (and self-populates) the repo-level [`structures/`](../structures) PDBx cache. |
| [`campaign.py`](campaign.py) | the parameterized campaign engine — `Campaign` / `CampaignConfig`, extracted from what used to be a duplicated `design_campaign.py` per campaign folder. Iterates scaffold→target pairs, runs `--trials` independent any-order masked-LM design trials over the edit window, batches them on one GPU forward, and writes one CSV per pair (round 0 = scaffold). Two selection strategies, `guided` (surrogate-scored, optionally with the brightness head) and `gibbs` (pure masked-LM sampling), ported verbatim from the original drivers so results stay bit-for-bit reproducible. |
| [`build_design_windows.py`](build_design_windows.py) | applies `pockets.py`'s rule across a whole scaffold set and writes one portable JSON, so a design run loads windows instead of re-deriving them. The Tier-B variant additionally restricts chromophore H-bond partners to the H-bond-capable alphabet via `position_constraints`. |

## Checkpoints

Both are **tracked in git** — every campaign loads them and neither is cheap to reproduce. They are
the two exceptions to the repo-wide `*.pt` ignore rule.

| checkpoint | what | selected by |
|---|---|---|
| `models/surrogate_cnn-max-d1_alldata.pt` | the `(ex, em)` peak surrogate, `cnn-max` depth 1, refit on all curated data (train MAE ≈ 5.2 nm). `campaign.DEFAULT_SURROGATE`. | the peak surrogate architecture sweep — see [`in-silico-test/`](../in-silico-test) for the current sweep and its held-out numbers |
| `models/brightness_cnn-max-d2_40k.pt` | bright/dim **classifier** head, `cnn-max` depth 2, trained on the stratified sub-40k GFP DMS set (val AUROC ≈ 0.98). `campaign.DEFAULT_BRIGHTNESS`, the optional `+ lam_bright` term in the guided strategy. | [`GFP_DMS/sweep_classify_parallel.py`](../GFP_DMS) |

A classifier head carries no `(mean, std)` metadata, so `peak_models.wrap` treats it as identity; a
regression head bakes its train-split standardization into the checkpoint. Either works as the
brightness head.

## Importing it

`fpdesign` is a package but is **not installed**. Callers put the repo root on `sys.path` and import
through the package:

```python
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]     # depth depends on the caller
sys.path.insert(0, str(REPO))
from fpdesign import peak_models as pm
from fpdesign.campaign import Campaign, CampaignConfig
```

Consumers: [`design-campaign-EGFP/`](../design-campaign-EGFP) (the campaign engine and both
checkpoints), [`dataset_pipeline/embed.py`](../dataset_pipeline/embed.py) and
`embed_fpbase_maxpool.py` (the ESM-2 helpers), and
[`GFP_DMS/nn_distance_accuracy.py`](../GFP_DMS/nn_distance_accuracy.py).

[`in-silico-test/`](../in-silico-test) deliberately does **not** import from here — it vendors its
own copies of `peak_models.py` and `pockets.py` under `in-silico-test/lib/` so that experiment folder
stays self-contained. Those are copies; a fix here does not reach them.
