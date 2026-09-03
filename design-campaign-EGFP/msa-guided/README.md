# msa-guided/ — the family alignment as the generative model (EGFP → mOrange & EBFP)

A fifth strategy for the EGFP campaign. Everything is shared with the ESM-2 guided arm
([`../esm2_guided/`](../esm2_guided/), which superseded the T=10
`../lambda_sweep/` this was originally written against) — same EGFP scaffold (`4EUL`, idx 171), same Tier-B 5 Å
chromophore window (25 editable positions), same all-data surrogate (train MAE 5.2 nm), same
`cnn-max-d2` brightness classifier (val AUROC 0.982), 12 trials × 3 iterations — except for the
one thing that decides which residues are even on the table:

> **ESM-2 does not propose the candidates. The 763-sequence FP family alignment does.**

## Why replace ESM-2 here

ESM-2 650M is close to uninformative on this fold. Masked-marginal top-1 accuracy is **12.6 % on
EGFP** and 10.1 % on avGFP, against **66–80 % on ubiquitin, lysozyme and trypsin** of
comparable length (`../../msa_conservation/results/esm_calibration.csv`). Mean max probability is
0.128 on EGFP versus 0.68–0.75 on the controls. That is a property of the model on this family,
not a bug in how it is called — the controls are scored by the same code path.

A near-flat proposal filtered to top-`k` is close to an arbitrary 10-residue alphabet, and the
audit of every design the four ESM-based EGFP strategies have produced
(`../../msa_conservation/archive/design_audit_*.csv` — local-only archive, 51,731
position-edits over the four superseded ESM-2 runs) shows what that costs:

| | share of all edits |
|---|---|
| introduce a residue **no aligned FP uses at that column** | **22.6 %** |
| fall outside the family's 90 %-mass alphabet | 53.7 % |
| **bury a formal charge** (D/E/K/R at RSA < 0.05) | **12.9 %** |

Worst offenders were F47 (50 % of its edits family-implausible, including Arg × 947 at RSA 0.000),
Y93 (68 % class-breaking at a column that is 84 % aromatic) and L61 (53 %, at a column where all
763 FPs are aliphatic). The window's guardrails are geometric and catalytic; nothing in them
encodes what the fold tolerates chemically, and that job had been implicitly delegated to a model
that has no opinion here.

The family alignment is decisive exactly where ESM-2 is not, so this strategy uses it directly.

## What the strategy is

**Proposal.** At each visited position the candidate distribution is the Henikoff-weighted family
frequency of each residue in the aligned column EGFP's own residue occupies. This is a PSSM —
position-specific but **context-independent**, so unlike a masked-LM conditional it does not
change as the sequence is edited, and it is free to evaluate. `build_msa_pssm.py` writes it to
`msa_pssm_egfp.json`; `MSACampaign.esm_logits_at` returns it in place of an ESM-2 forward.

**Hard support constraint.** A residue whose weighted family frequency is **0** is removed from
the position's alphabet outright, so it can never be selected. Intersected with the Tier-B
aromatic / H-bond constraints, this blocks **211 of 500** position-residue combinations:

| | |
|---|---|
| alphabet size | min **4**, median **12**, max **20** |
| tightest | **L61 → `LIVM`** — the family's only options in 763 sequences |
| loosest | **I168 → all 20** — genuinely permissive, so nothing is imposed |
| chromophore | **Y67 → `YWHF`** — W and H retained, so BFP/CFP chemistry stays reachable |
| Q95 | `WQYNT` — the family is 56 % aromatic here, which the Tier-B H-bond alphabet alone did not capture (53 % of its ESM-era edits were family-implausible) |

EGFP's own residue survives at every position — it is in the alignment by construction — so the
edit penalty can always choose to stay put, and the scaffold is always reachable.

### The window itself is unchanged; only the alphabet is

`design_windows_egfp_tierB.json` is **byte-identical** across all strategy folders (md5
`9ad48573…` for gibbs, esm2_guided, msa-gibbs and this one — and for the retired T=10
folders now in `../archive/superseded-unmatched-runs/`). Same 25 editable positions, same three fixed (68 Gly, 97 Arg, 223 Glu), same Tier-B
constraints. The support constraint is layered on top and is **strictly tighter at 23 of the 25
positions and never looser at any**: 457 Tier-B residue options → **289**, i.e. 63 %.

Two consequences worth recording.

**It derives the L61 restriction automatically.** Tier-B leaves position 61 fully free at 20
residues even though it is the most constrained column in the window (RSA 0.00, 4.8 Å from the
chromophore, no non-aliphatic residue in any of 763 FPs). The support constraint cuts it to exactly
`{L, I, V, M}` without anyone hand-writing a `position_constraints` entry, and does the equivalent
at 22 other positions (Y93 20 → 8, V113 20 → 8, F47 20 → 8).

**But Tier-B still blocks residues the family uses.** The constraint runs both ways, and this is
the direction the support mask cannot fix, because it only ever intersects. Tier-B constrains four
positions; at the other 21 it permits all 20, so it can remove nothing there. At those four:

| position | Tier-B alphabet | family-supported residues blocked | family mass discarded |
|---|---|---|---|
| Y67 (avGFP Y66) | `FHWY` | R, L | 4.7 % |
| Q95 (avGFP Q94) | `DEHKNQRSTWY` | V, I, L, F, M | 13.7 % |
| **H149 (avGFP H148)** | `DEHKNQRSTWY` | **V, C, F, I, G, A** | **25.2 %** |
| T204 (avGFP T203) | `DEHKNQRSTWY` | I, A, V, L, M, C, F, G | 12.8 % |

H149 is the outlier: the family uses 16 residues at avGFP H148 and Tier-B discards a quarter of the
observed distribution, most of it Val (11.6 %), Cys (6.3 %) and Phe (4.7 %). T204 loses Ile at
7.5 % — the well-known T203I dark/photoactivatable variant.

This has a concrete cost for blue-shifted design. The Tier-B H-bond alphabet exists to *preserve*
hydrogen bonding at the chromophore's polar contacts, but **mKalama1** — the one *Aequorea* blue FP
that keeps the chromophore tyrosine instead of using Y66H, at 385/456 nm and 90 % identity to EGFP
— works by *destroying* that proton-relay network: `Y145M / H148G / S205V` in avGFP numbering.
Tier-B **forbids two of those four**:

| mKalama1 mutation | EGFP position | family frequency | Tier-B |
|---|---|---|---|
| Y145M | 146 | 0.018 | allows |
| **H148G** | **149** | **0.008** | **forbids** (Gly not in the H-bond set) |
| **T203V** | **204** | **0.014** | **forbids** (Val not in the H-bond set) |
| S205V | 206 | 0.091 | allows (position unconstrained) |

The family does use both; it is Tier-B, not the data, that rules them out. So for a blue-shifting
campaign the H-bond constraint encodes close to the opposite of the right prior. The shortlisted
EBFP designs reach the mKalama1 mechanism only through the doors Tier-B leaves open — all ten
mutate S206, and the five that touch 149 use H-bond-set substitutions (S/D/Q/T) rather than the
glycine mKalama1 actually uses. Widening 149 and 204 to include the family-supported aliphatics and
Gly is a one-line change, but it should be made as a **separate window variant** so the six
existing strategies stay comparable.

**Weighting.** Every term in the guided score is z-scored across the `k` candidates
(`fpdesign/campaign.py::_zc`), so each already has unit variance and λ *is* the relative weight.
The inherited defaults did not reflect that — at `λ_ex=λ_em=20, λ_bright=60, λ_edit=10, T=10` the
effective weights were `0.1 / 2 / 2 / 6 / 1`, i.e. the proposal counted for **1/60th** of
brightness. Here every default is **1.0 at `T=1`**:

```
score = z(logp_MSA) − λ_ex·z(|Δex|) − λ_em·z(|Δem|) + λ_bright·z(bright) − λ_edit·z(is_edit)
```

so λ = 1 everywhere weights all five terms equally, and the sweep varies them around that centre.

## The sweep

5 × 5 × 5 = **125 cells**, log-spaced and centred on the all-equal point, 12 trials × 3 iterations,
**both targets in one process per cell**.

| axis | values | what it weights |
|---|---|---|
| `λ_ex = λ_em` | 0.25, 0.5, 1, 2, 4 | pull toward the target's excitation / emission peaks |
| `λ_bright` | 0, 0.5, 1, 2, 4 | pull toward the brightness classifier's positive class |
| `λ_edit` | 0, 0.5, 1, 2, 4 | penalty per position that differs from the scaffold |

`λ_bright = 0` and `λ_edit = 0` are deliberate controls: now that the family support hard-limits
the alphabet, it is an open question whether either penalty is still earning its place.

### The λ≈1 trap, and why this is not it

`../archive/README.md` records a negative control worth not repeating: `λ = 1` at `T = 10` puts the
softmax ~96 % of the way to uniform, degenerating the search into random sampling from the
proposal's top-k (22.6 / 25 positions mutated, mOrange ending *worse* than the untouched
scaffold). This campaign also uses λ ≈ 1 — but at `T = 1`, ten times sharper.

That is an argument, not evidence, so `check_scale.py` measures it. It runs a real design pass and
reports the entropy of the actual selection distribution, normalized so **1.0 is exactly uniform**
over the `k` allowed candidates and **0.0 is deterministic**:

| setting | λ (peaks/bright/edit) | T | H / H<sub>max</sub> | mean p(chosen) |
|---|---|---|---|---|
| grid centre | 1 / 1 / 1 | 1 | **0.194** | 0.868 |
| grid corner | 4 / 4 / 4 | 1 | 0.031 | 0.974 |
| weakest cell | 0.5 / 0 / 0 | 1 | 0.694 | 0.448 |
| *archived negative control* | 1 / 1 / 1 | 10 | **0.980** | 0.189 |

The last row reproduces the documented failure, which is what makes the other three
trustworthy. The whole grid sits between 0.03 and 0.69 — decided, and clear of the degenerate
corner. It also shows the sharp end saturates: beyond λ ≈ 2 the sampler is effectively `argmax`
and extra weight buys nothing, which is why the peak axis is log-spaced to 4 rather than stepping
1, 2, 3, 4.

## Layout

```
build_msa_pssm.py                  setup: alignment → per-position family distributions
msa_pssm_egfp.json                 the PSSM (alphabet + probs per 0-based window position)
design_windows_egfp_tierB.json     the Tier-B window (copy of the campaign's)
design_campaign.py                 driver: MSACampaign, per-cell outdir, no pseudo-perplexity
check_scale.py                     selection-entropy probe (the table above)
run_sweep.sh                       sequential loop over the 125 cells, both targets
score_sweep.py                     scores every cell, re-checks family support, writes the caches
sweep_metrics.csv                  per-cell metrics (250 rows: 125 cells × 2 targets)
sweep_designs.csv                  per-design metrics (3000 final-round designs)
designs/lam-ex{P}_lam-em{P}_lam-bright{B}_lam-edit{E}/design_EGFP-{EBFP,mOrange}.csv
logs/sweep_<timestamp>.log         one log per pass (.last_log points at the newest)
```

Reproduce, from this folder:

```bash
python build_msa_pssm.py           # rebuild the PSSM (prints the per-position alphabets)
python check_scale.py              # confirm the λ/T scale is in the decided regime
bash run_sweep.sh                  # 125 cells, ~5 h on one GPU; complete cells are skipped
python score_sweep.py              # score the grid (writes sweep_metrics.csv, sweep_designs.csv)
```

## Results

125 cells × 2 targets × 12 trials × 3 iterations, 17,748 s (4.9 h) on one RTX PRO 4500, every cell
exiting clean. `score_sweep.py` writes `sweep_metrics.csv` (per cell) and `sweep_designs.csv` (per
design) and re-verifies the central claim:

> **0 of 20,282 edits fell outside the family support** — against 22.6 % for the ESM-based
> strategies on the same window.

All three axes behave monotonically and each does the job it was given:

| λ_ex = λ_em | 0.25 | 0.5 | 1 | 2 | 4 | |
|---|---|---|---|---|---|---|
| EBFP mean peak error (nm) | 73.5 | 63.0 | 45.0 | 20.6 | **2.9** | scaffold 88.3 |
| mOrange mean peak error (nm) | 54.5 | 45.9 | 32.5 | 18.4 | **7.1** | scaffold 56.7 |

| λ_bright | 0 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| EBFP fraction predicted bright | 0.19 | 0.24 | 0.33 | 0.54 | **0.80** |
| mOrange fraction predicted bright | 0.32 | 0.40 | 0.51 | 0.77 | **0.95** |

| λ_edit | 0 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| mean mutations from scaffold | 15.2 | 9.5 | 5.7 | 2.5 | 1.0 |

**The accuracy-optimal and usable optima are different cells**, because peak accuracy is bought by
moving away from EGFP and the brightness classifier is less confident the further you go:

| target | selection | λ (peaks/bright/edit) | mean err | best | bright | mutations |
|---|---|---|---|---|---|---|
| mOrange | accuracy-optimal | 4 / 0 / 0 | 2.4 nm | 0.29 | 0 / 12 | 17.7 |
| mOrange | **usable** | **4 / 2 / 1** | **5.4 nm** | 1.91 | 6 / 12 | 8.2 |
| mOrange | usable, brighter | 4 / 4 / 0.5 | 6.8 nm | 2.92 | 9 / 12 | 9.6 |
| EBFP | accuracy-optimal | 4 / 0.5 / 0 | 0.7 nm | 0.12 | 0 / 12 | 16.8 |
| EBFP | **usable** | **2 / 4 / 0.5** | **23.0 nm** | 2.31 | 10 / 12 | 8.1 |

EBFP pays much more for brightness than mOrange does (23.0 vs 5.4 nm), consistent with its far
larger spectral jump from the same scaffold.

The trade-off is not specific to the brightness classifier — it shows up just as clearly on the
family log-likelihood of the design under the PSSM (`fam_logp`, the GPU-free analogue of
pseudo-perplexity):

| correlation across cells | EBFP | mOrange |
|---|---|---|
| mutations vs predicted brightness | −0.73 | −0.85 |
| **family log-likelihood vs predicted brightness** | **+0.79** | **+0.81** |
| **family log-likelihood vs peak error** | **+0.88** | **+0.66** |

Designs that stay typical of the family are predicted brighter *and* miss the target by more. That
is the real frontier here, and it is a statement about the objective, not about this proposal
distribution: the family profile constrains *which* residues are legal, but pursuing a large
spectral shift still requires leaving the part of sequence space the family densely occupies.

Trial diversity rises rather than collapses at high λ (8.2 → 11.9 distinct designs per 12 trials
from λ_peaks 0.25 → 4): with the peak term dominating, different random visit orders reach
genuinely different optima, so the sharp cells are not just duplicating one answer.

## Three things to know before comparing cells

**1. The folder name is the authoritative record of the weights.** The engine writes `lam_ex`,
`lam_em` and `lam_bright` columns but has **no `lam_edit` column**, so only the folder name carries
all four. This is also why the driver names its own output folder rather than using the engine's
`outdir_lambda_suffix`, whose suffix omits `λ_ex`/`λ_em` — all five peak levels would have collided
in one folder, and since `existing_pair` validates only trials/rounds/temp/k and never the λs, the
later cells would have been silently skipped as "cached". The sweep would have quietly produced a
25-cell grid labelled as 125.

**2. The `ppl` column is deliberately blank.** ESM-2 pseudo-perplexity is not this strategy's
naturalness measure and costs about as much as a whole design iteration. The natural analogue —
the design's family log-likelihood under the PSSM — is a pure function of the sequence and
`msa_pssm_egfp.json`, so it can be computed from the CSVs at any time without a GPU. Nothing is
lost by leaving the column empty.

**3. z-scoring normalizes away *how decided* the family is.** L61 (84 % Leu over 4 allowed
residues) and I168 (20 allowed, nearly flat) both contribute one unit of variance to the score.
The strength of the prior is carried by the **hard support mask**, not by the z-scored term, which
only carries its shape. This matches how the ESM strategies treat their own proposal term, so
strategy-to-strategy comparison stays like-for-like — but it does mean λ_MSA is not a knob for
"trust the family more at the positions where it is certain".

## Caveats

- **"Zero" means zero in 763 observed sequences at N_eff 272.** A true frequency below ~0.4 % can
  read as 0, so the blocked residues are "the family never does this", not "this cannot fold".
  The strategy is deliberately conservative: it will not propose a functional residue that
  evolution happens not to have sampled in this dataset.
- **A PSSM has no epistasis.** Positions are treated independently, so the profile cannot express
  "Thr here *given* Ser there". ESM-2 could in principle, which is the one thing given up — though
  its calibration on this family suggests there was not much to give up.
- **No oracle.** As everywhere in this campaign, the surrogate that guides the search also scores
  the result, and it was trained on all FP data. `pred_ex/pred_em/pred_bright` are in-sample and
  optimistic; the real judge is experiment.
