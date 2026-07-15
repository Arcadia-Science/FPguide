# ESMFold structure pipeline for FP scaffolds

An alternative to fetching experimental RCSB structures for defining the
structure-based chromophore edit window in
[`../../guided_design_peak_structure_multiscaffold.ipynb`](../../guided_design_peak_structure_multiscaffold.ipynb).
Folds the **exact dataset sequence** of a scaffold with ESMFold (`esmfold_v1`) so
the predicted-structure numbering matches the dataset sequence 1:1 (no RCSB fetch,
no homolog alignment — the step eqFP578 currently needs against TagRFP `3M22`).

- [`../../esmfold_fold.py`](../../esmfold_fold.py) — load `esmfold_v1` on MPS (CPU
  fallback), fold sequences, save PDBs + `esmfold_meta.json`.
- [`../../pockets.py`](../../pockets.py) — drop-in `struct_pocket_esmfold` +
  `struct_pocket_experimental`, and the pocket comparison.

## TL;DR — do NOT use ESMFold structures for the edit window

**ESMFold predicts the GFP-family β-barrel poorly and the derived pocket is wrong.**
Keep using the experimental structures in the notebook. Details below.

## Licensing (verified commercial-OK)

| Component | License | Notes |
|---|---|---|
| ESMFold model + code (`facebookresearch/esm`) | **MIT** | Weights trained from scratch by Meta; **not** derived from AlphaFold → no DeepMind encumbrance. |
| OpenFold (`aqlaboratory/openfold`, code dep only) | **Apache-2.0** | We use only its Python modules, **not** its trained weights (those are CC BY 4.0). |

Both permit commercial use.

## Install (macOS / Apple Silicon, no CUDA) — one-time

OpenFold's official install requires `nvcc`/Linux; ESMFold's *forward path* only
needs OpenFold's pure-Python modules. Done in env `esm2-fp-design`:

1. `pip install ml_collections einops 'dllogger @ git+https://github.com/NVIDIA/dllogger.git' dm-tree omegaconf pytorch_lightning transformers` (transformers optional, used only for the cross-check).
2. OpenFold pinned commit `4b41059` installed with the **CUDA extension stripped**
   (`setup.py` rewritten to `ext_modules=[]`; the upstream file also probes `nvcc`
   at import and crashes on Mac).
3. A stub `attn_core_inplace_cuda` module on `sys.path` so OpenFold's top-level
   `importlib.import_module("attn_core_inplace_cuda")` succeeds. The real kernel is
   only used by OpenFold's memory-efficient attention, which ESMFold never calls.
4. Three OpenFold files patched for `pytorch_lightning >= 2.x` / `torch >= 2.9`
   compatibility (all inference-irrelevant, only imported by OpenFold's eager
   `__init__`): `utils/seed.py`, `utils/callbacks.py`, `utils/tensor_utils.py`
   (`batched_gather` now indexes with an explicit tuple). Each edit carries an
   Apache-2.0 modification note.

The env's `esm/esmfold/v1/esmfold.py` casts the ESM-2 LM to fp16; `esmfold_fold.py`
undoes that at load (`model.esm.float()`) so CPU works and MPS is fp32-precise.

## Usage

```bash
conda activate esm2-fp-design
cd peak_design
python esmfold_fold.py            # folds DsRed(138)/avGFP(52)/eqFP578(179) -> *.pdb here
python pockets.py                 # ESMFold-vs-experimental 5 A pocket comparison
```

## Quality finding (validated, reproducible) — why not to use it

Folding metrics (mean pLDDT is well-calibrated; >70 ≈ reliable backbone):

| scaffold | mean pLDDT | pTM | CA-RMSD vs experimental |
|---|---|---|---|
| DsRed (vs 1G7K)   | 31.4 | 0.27 | 19.8 Å |
| avGFP (vs 1GFL)   | 40.1 | 0.46 | 15.9 Å |
| eqFP578 (vs 3M22) | 32.8 | 0.28 | 18.0 Å |

Control proteins on the same install fold correctly: ubiquitin 86, T4 lysozyme 90,
ESM's designed 65-mer example 88. So the install is sound — ESMFold simply predicts
the 11-stranded Greek-key FP barrel badly (its many long-range antiparallel strand
contacts are exactly what a single-sequence model gets wrong).

This is **not** an artifact of this environment. Identical low pLDDT was obtained on:
torch 2.12.1 **and** 2.2.2; CPU **and** MPS; fp16 **and** fp32 LM; the `fair-esm`
implementation, HuggingFace `transformers` `EsmForProteinFolding`, **and Meta's own
reference ESMFold at the ESM Atlas API** (avGFP mean pLDDT 0.43 there). Other
GFP-family FPs fail too (sfGFP 44, mCherry 35).

Pocket comparison (see [`pocket_comparison.txt`](pocket_comparison.txt)): the ESMFold
5 Å pocket recovers only **21–39 %** of the experimental pocket (Jaccard 0.19–0.32).
It keeps the sequence-local residues near the chromophore but **misses the long-range
color-tuning residues** the structure window exists to capture — e.g. for avGFP it
misses T203/S205/H148 (dataset 203/205/148), the canonical color residues. Since a
misfolded barrel places those far from the chromophore, the ESMFold window collapses
toward roughly the old ±10 sequence window it was meant to improve on.

**Recommendation:** keep fetching the experimental structures (1G7K / 1GFL / 3M22) for
the edit window. Use this ESMFold pipeline only for scaffolds with no experimental
structure *and no usable homolog*, and treat its window as exploratory — ideally
gated on a per-residue pLDDT / barrel-closure sanity check first.
