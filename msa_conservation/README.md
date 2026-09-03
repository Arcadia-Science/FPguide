# Whole-dataset FP alignment and chemistry conservation

A multiple sequence alignment of **every unique sequence in the curated FP dataset**
(763 proteins, 82 source organisms) and a per-position analysis of what side-chain
**chemistry** the family holds fixed — as opposed to what amino-acid **identity** it
holds fixed.

Headline result: **conservation in this family is overwhelmingly about the fold, not
about the chromophore.** Chemistry is locked at exactly twice as many positions as
identity (28 vs 14 at the 90% level), and the strength of that locking tracks how
buried a position is (Spearman ρ = −0.40 with solvent accessibility) while having *no*
relationship at all to how close it sits to the chromophore (ρ = −0.003, p = 0.96).
The colour-tuning pocket is the family's variable real estate, pinned by only three
invariant chemical anchors: **Gly67**, **Arg96⁺** and **Glu222⁻**.

## Pipeline

```bash
python build_msa_input.py     # curated union -> data/fp_all.fasta          (763 seqs)
                              #   ^ needs the archived brightness + pKa sets; see below
./run_msa.sh                  # MAFFT FFT-NS-i -> data/fp_all.aln.fasta     (~4 min)
python conservation.py        # -> results/column_conservation.csv, summary.json
python validate.py            # -> results/validation.json   (4 robustness tests)
python report.py              # -> results/findings.txt      (numbers quoted below)
python window_vs_family.py    # -> results/window_vs_family_egfp.csv, window_alphabet_egfp.csv
python esm_vs_family.py       # -> results/esm_vs_family_egfp.csv, esm_calibration.csv (needs GPU)
python esm_profiles.py        # -> results/esm_profiles.npz, esm_sweep_profiles.npz,
                              #    esm_family_sweep.csv   (needs GPU, ~9 min)
python plot_conservation.py   # -> figures/*.png
```

### Step 1 needs two archived datasets — you probably don't need to run it

The 763-sequence input is the **union of all three** curated trait sets (peak 758, plus the 2 and 3
that brightness and pKa add). But brightness and pKa were archived after this alignment was built
([`dataset_pipeline/README.md`](../dataset_pipeline/README.md#archived-brightness--pka)), so on a
fresh clone only `data/peak/curated/` exists and step 1 cannot rebuild the union as-is. It fails
loudly and prints these three options rather than quietly emitting 758 sequences:

1. **Skip step 1.** `data/fp_all.fasta`, `data/fp_all_meta.csv` and `data/fp_all.aln.fasta` are all
   tracked in git, so steps 2 onward run from a bare clone. This is the normal path.
2. **Rebuild the two archived sets**, which is cheap (seconds, CPU) and bit-reproducible from the
   tracked FPbase export — the rebuilt union reproduces the committed `fp_all.fasta` and
   `fp_all_meta.csv` byte-for-byte:
   ```bash
   cd ../dataset_pipeline
   python build_dataset.py --target brightness   # -> data/brightness/curated/  (533 seqs)
   python build_dataset.py --target pka          # -> data/pka/curated/         (368 seqs)
   cd ../msa_conservation && python build_msa_input.py
   ```
3. **Build the peak-only variant**, `python build_msa_input.py --peak-only` — 758 sequences, no
   prerequisite. This is **not** the published input: it drops the 5 sequences only brightness/pKa
   readmit (`CAR-GECO1`, `mKeima`, `pHluorin4`, `pHmScarlet`, `PSLSSmKate`), so every conservation
   count will differ from the numbers quoted below. It writes `data/fp_peak_only.*` — deliberately
   different filenames, since everything downstream reads `fp_all*` by name and pairs alignment
   rows to metadata rows by `msa_id`, so a 758-row file under those names would silently invalidate
   the published results. `run_msa.sh` aligns `fp_all.fasta`; the script prints the `mafft` line for
   the peak-only file.

`visualization.ipynb` then renders the seven-figure ESM-2-vs-family comparison
(`figures/esm_vs_msa_*.png`) from those cached distributions; it needs no GPU of its own.

Everything this folder needs — `mafft` plus biopython, biotite, pandas, scipy and
matplotlib — is declared in the project [`environment.yml`](../environment.yml).

## What was aligned

The union of the three curated trait sets — peak (758) plus what brightness and pKa
add (2 + 3) = **763 unique sequences**. The 5 extras are proteins the peak curation
dropped as analyte sensors (`CAR-GECO1`, `mKeima`, `pHluorin4`, `pHmScarlet` — a single
(ex, em) label is ill-defined when the peak moves with analyte) or as an unresolvable
multi-state entry (`PSLSSmKate`); all 5 carry a brightness or pKa measurement, so the
union readmits them as *sequences* even though they have no peak label. That union, not the raw FPbase export, is the
right universe for a *family* alignment: `dataset_pipeline` has already removed the
biliverdin/phytochrome near-infrared class, which does not share the GFP fold and would
contribute nothing but junk columns.

MAFFT FFT-NS-i (`--maxiterate 1000`) rather than `--auto`, which at N = 763 falls back
to a single progressive pass. `run_msa.sh` pins `--thread 1 --randomseed 0`: MAFFT's
parallel refinement visits subalignments in nondeterministic order, so multithreaded
runs of the same input differ by a few columns and shift downstream counts by ±1.
Single-threaded is bit-identical run to run, at a cost of ~4 min instead of ~30 s.

Result: 763 × 1861 columns, of which **233 are core** (occupancy ≥ 50%) and **208 are
near-complete** (occupancy ≥ 90%) — matching the 238-residue avGFP barrel. The sparse
remainder is insertions from the 18 tandem-dimer and Ca²⁺-sensor constructs
(`tdTomato`, `tdStayGold`, `GCaMP2`, `RCaMP`, …), whose second barrel copy cannot align
to the first; `results/sequence_qc.csv` flags them by `core_frac` (0.37–0.50, i.e. one
barrel's worth). All analysis uses the 208 near-complete columns, so those insertions
never enter a statistic.

Positions are reported in **avGFP numbering**, and structural context (relative solvent
accessibility, distance to the chromophore, secondary structure) comes from the local
wild-type crystal structure **1GFL**, mapped onto the dataset avGFP sequence by
alignment rather than by trusting PDB residue numbers.

## How conservation is measured

Three choices matter more than the choice of alignment tool:

**1. Redundancy correction.** This dataset is a mutant library, not a phylogenetic
sample: 276 of 763 sequences are Aequorea-lineage, most of them avGFP point mutants.
Raw column frequencies would report "conserved" for anything the avGFP lineage happens
to share. Every number here uses **Henikoff position-based sequence weights**, which
cut the effective sample size from 763 to **N_eff = 272**. Clustering at 90% identity
gives 117 clusters (N_eff = 173) and is used as an independent cross-check.

**2. Background normalization.** Conservation is the fraction of the family's *own*
uncertainty removed at a column, `C = 1 − H_column / H_background`, where the background
is the weighted composition of the alignment core. Raw entropy would score an all-Leu
column and an all-Trp column identically even though Leu is ~5× more likely a priori.

**3. Identity and chemistry, separately.** `C_id` is computed over the 20 amino acids;
`C_chem` over 8 exclusive side-chain classes — aliphatic (AVLIM), aromatic (FWY),
polar (STNQ), acidic (DE), basic (KRH), and glycine, proline and cysteine each on their
own, since what distinguishes those three is backbone flexibility, backbone rigidity and
a thiol rather than anything shared with a bulk group. `C_chem − C_id` isolates the
positions this analysis is about: chemistry pinned, identity free. Continuous
properties (Kyte–Doolittle hydropathy, Zamyatnin volume, formal charge, aromaticity,
H-bond capacity, Grantham polarity) get a variance-reduction score
`ρ = 1 − Var_column / Var_background` on the same footing.

## Findings

### Chemistry is conserved at ~2× as many positions as identity

| threshold | same amino acid | same chemistry class | ratio |
|---|---|---|---|
| ≥95% | 9 | 16 | 1.78 |
| ≥90% | 14 | 28 | 2.00 |
| ≥80% | 26 | 46 | 1.77 |
| ≥70% | 34 | 71 | 2.09 |

Only **14 of 208** positions keep the same residue in ≥90% of the family. That set is
almost entirely structural rather than photophysical: seven glycines (G20, G33, G35,
G40, G67, G91, G127), two prolines (P75, P196), three aromatics (F27, F130, Y66) and
the catalytic pair R96 / E222. F27 is the single perfectly invariant position — Phe in
all 761 sequences that occupy the column.

### The 28 chemistry-locked positions

| class | n | positions | mean identity freq |
|---|---|---|---|
| aliphatic | 12 | M1, V12, L18, V22, L53, L60, M78, I136, L137, I152, I161, A226 | 0.65 |
| glycine | 7 | G20, G33, G35, G40, G67, G91, G127 | 0.97 |
| aromatic | 5 | F27, Y66, F71, F100, F130 | 0.84 |
| proline | 2 | P75, P196 | 0.93 |
| basic | 1 | R96 | 0.99 |
| acidic | 1 | E222 | 0.95 |

26 of the 28 are buried (median RSA 0.019), and 17 of 28 sit in loops rather than
strands. The classes split cleanly by *mechanism*:

- **Glycine and proline are conserved as identity** (mean identity frequency 0.97 and
  0.93) because their "chemistry" is backbone conformation, and nothing substitutes for
  it. These are the turns and the tight barrel closures.
- **Aliphatic positions are conserved only as chemistry** (mean identity frequency
  0.65). This is where the two measures diverge most: the buried greasy core specifies
  "hydrophobic and roughly this big" and lets Met/Leu/Ile/Val interchange freely.
  L18 is the clearest case — 99% aliphatic but only 51% Met (M 0.51 / L 0.32 / I 0.16);
  I152 is 91% aliphatic with no single residue above 33%.
- **Aromatics** sit in between: F100 is 93% aromatic but only 59% Phe, trading with Tyr
  (0.34). Y66, the chromophore ring itself, is 94% aromatic and 91% Tyr — the residual
  being the Trp and His of engineered cyan and blue variants.

### Conserved chemistry tracks burial, not the chromophore

Across the 208 core positions:

| property (weighted per-column mean) | Spearman ρ vs RSA | p |
|---|---|---|
| charge constraint ρ | −0.640 | 3e−25 |
| mean polarity (Grantham) | +0.587 | 1e−20 |
| mean hydropathy (Kyte–Doolittle) | −0.574 | 1e−19 |
| mean H-bond capacity | +0.475 | 4e−13 |
| mean side-chain volume | −0.242 | 4e−04 |
| **chemistry conservation `C_chem`** | **−0.403** | 2e−09 |
| identity conservation `C_id` | −0.207 | 3e−03 |

Buried positions average hydropathy **+0.53** against **−1.85** for exposed ones, a
2.4-unit swing on the Kyte–Doolittle scale. Note that `C_chem` tracks burial about
twice as strongly as `C_id` does: the family conserves *chemistry* as a function of
structural role, and lets identity drift.

Against distance to the chromophore, `C_chem` gives **ρ = −0.003, p = 0.96** — no
gradient whatsoever. Mean `C_chem` is 0.50 in the 5 Å pocket, 0.49 in the 5–10 Å shell
and 0.51 beyond 10 Å. `C_id` actually *rises* slightly with distance (ρ = +0.20,
p = 0.004), i.e. the pocket is marginally *less* identity-conserved than the scaffold.
This is the expected signature of a family whose members differ mainly in colour: the
pocket is exactly the surface selection has been free to repaint.

### Buried charge is excluded except at the catalytic dyad

Of the 113 buried positions, **81 are both charge-free and strongly charge-constrained**
(|mean charge| < 0.1, ρ ≥ 0.7) — the barrel interior actively excludes formal charge.
Exactly **two** buried positions carry a conserved charge, and they are the two residues
that catalyse chromophore maturation:

- **R96** — 100% basic, 99.1% Arg, 2.7 Å from the chromophore
- **E222** — 94.7% acidic, 94.6% Glu, 2.7 Å from the chromophore

A weaker second tier appears only when the bar drops to 70% (D82, K85, H199, E213). On
the solvent-exposed surface the picture inverts: mean charge constraint is **ρ = −0.30**,
i.e. exposed positions are *more* charge-variable than the family background. Surface
electrostatics is the least conserved chemistry in the entire protein.

### The pocket has three anchors and is otherwise plastic

Within the 5 Å contact shell (28 positions), only five are chemically locked, and each
does a distinct job:

| position | chemistry | frequency | role |
|---|---|---|---|
| G67 | glycine | 99.8% | the Gly of the X-Y-G triad; backbone cyclization is impossible without it |
| R96 | basic | 100% | catalytic Arg, polarizes the imidazolinone carbonyl |
| E222 | acidic | 94.7% | catalytic Glu, general base for dehydration |
| Y66 | aromatic | 94.5% | the chromophore ring |
| L60 | aliphatic | 100% | fully buried packing position (RSA 0.00) |

Everything else in the pocket is a mosaic. Chromophore position 65 — the X of X-Y-G —
is the single most variable position in the shell (`C_chem` 0.17): Gln 0.20, Gly 0.19,
Met 0.16, Ser 0.10, Thr 0.09, recovering exactly the QYG of DsRed, the MYG of mCherry
and the (S/T)YG of the GFPs. The avGFP H-bond network residues are similarly free
across the family — H148 is only 45% polar, T203 60% basic-dominated, S205 68%
aliphatic-dominated.

### The barrel's two-residue periodicity is visible in chemistry

Within the nine contiguous β-strand runs of ≥6 residues, autocorrelation of the
per-position mean hydropathy alternates in sign (lag 1 −0.37, lag 2 +0.07, lag 3 −0.38,
lag 4 +0.34), and polarity does the same more strongly (−0.42 / +0.20 / −0.39 / +0.38).
The structural control — RSA itself — gives the textbook pattern (−0.67 / +0.68 /
−0.63 / +0.42). So the in/out alternation is imprinted on the family's chemistry, but
with a visibly weaker amplitude than the geometry implies. That is consistent with GFP's
unusual barrel: the interior is not a dry greasy core but a polar cavity holding an
H-bond network and buried waters, so inward-facing positions are not uniformly greasy.

## Comparison with the campaign design windows

`window_vs_family.py` scores the EGFP edit window against this table. The EGFP scaffold is
itself a member of the alignment, so window positions map onto alignment columns exactly —
no re-alignment, no numbering assumptions — and EGFP's 1-based 68 resolves to the same
column as avGFP's 67, which is why avGFP-equivalent numbering is reported throughout.

**Scope: the EGFP window only** (28 positions). Earlier versions of this analysis also pooled
the 24 conventional scaffolds and the avGFP campaign — 723 scaffold-positions across 26
windows. Both of those campaigns were archived out of the repo and their window JSONs are not
published, so those rows cannot be regenerated from a clone. The analysis now stops at the
window it can defend; where it used to report a mean over 26 scaffolds it reports the
per-position value for one. One conclusion changed as a result, flagged below.

**The hard-fixed set is exactly right.** All three fixed positions are chemistry-locked in
the family: the chromophore Gly (EGFP 68) at 99.8% glycine, the catalytic Arg (97) at 100%
basic, the catalytic Glu (223) at 94.7% acidic. `pockets.py` derives these geometrically, by
nearest-Arg/nearest-Glu to the chromophore; the family evidence confirms it picks the right
residues. (The retired pooled run showed the same holding for all 78 fixed positions across
26 scaffolds, including the red/Anthozoa ones where the numbering shifts — that is no longer
reproducible from a clone, so it is not claimed here.)

**The window is depleted of constrained positions, as intended.** 13.5% of barrel columns
are chemistry-locked, against 8.0% of the editable window positions (2/25), and mean `C_chem`
is 0.459 inside the window against 0.503 for the 5 Å pocket as a whole. Excluding the three
anchors is what produces the depletion — the 5 Å criterion itself is chemically neutral (see
the absent distance gradient above).

**One real gap: avGFP L60 / EGFP L61.** Only two chemistry-locked positions remain editable
in the window, and one of them is already handled:

| avGFP pos | EGFP pos | class | family frequency | RSA | current constraint |
|---|---|---|---|---|---|
| 66 | Y67 | aromatic | 94.5% | 0.01 | aromatic `{Y,W,H,F}` — correct |
| **60** | **L61** | **aliphatic** | **100%** | **0.00** | **none — all 20 residues allowed** |

Position 60 (EGFP 61) is the single most constrained position in the entire EGFP window
(`C_chem` = 1.00), fully buried at RSA 0.00, 4.8 Å from the chromophore, and **not one of
763 aligned FPs puts a non-aliphatic residue there** — the family uses only Leu (84%),
Ile (11%), Val (3%) and Met (2%). It is currently free to become Asp, Lys or Pro. The
cheapest fix is a one-line `position_constraints` entry restricting it to `LIVM`. (The
retired pooled run found it unrestricted in 12 of the 26 windows, so this was never an
EGFP-specific oversight.)

**Tier-B's H-bond alphabet is defensible in the EGFP window.** `HBOND_AA` =
`{S,T,Y,N,Q,D,E,H,K,R,W}` is applied wherever a side-chain N/O sits within 3.5 Å of a
chromophore N/O in the scaffold's own structure. It lands on three EGFP positions, and across
them retains 83% of the family's weighted mass:

| avGFP pos | EGFP pos | family's dominant class | mass kept by `HBOND_AA` |
|---|---|---|---|
| 148 | H149 | polar (45%) | 0.748 |
| 94 | Q95 | aromatic (56%) | 0.863 |
| 203 | T204 | basic (60%) | 0.872 |

None of the three forbids the class the family prefers, so within this window the hand-written
alphabet and the family broadly agree. Position 148 is the loosest fit: the family's plurality
there is polar at only 45%, so a quarter of the natural mass sits outside `HBOND_AA`.

> **This is the one conclusion that narrowing to EGFP changed.** The retired pooled run found
> `HBOND_AA` retaining only 75% of family mass over 69 scaffold-positions, with four positions
> (avGFP 165, 205, 167, 220) where it *forbids* the family's preferred aliphatic class and
> discards 66–75% of the natural distribution — a per-scaffold geometric call disagreeing with
> the family consensus. Every one of those four came from a non-EGFP scaffold (mTagBFP2,
> PA-GFP, W1C, mCerulean2.D3, DsRed-Express, E2-Red/Green), so the finding does not apply to
> the EGFP campaign and is no longer reproducible from a clone. It is recorded here as prior
> observation, not as a claim this repo supports: if the conventional campaign is ever
> revived, it is the first thing to re-check.

**An empirical alternative.** `results/window_alphabet_egfp.csv` gives, for every
window position, the smallest residue set covering 90% of the weighted
family distribution — an evolution-derived alphabet that could replace or augment the
hand-written Tier-B sets. It is sharply position-dependent, which a single global
alphabet cannot be: EGFP L61 → `LI` (2 residues), Y93 → `YFML`, L221 → `LQIV`, while
T66, the chromophore X position, → `QGMSTACWH` (9) and I168 → `MILWKRVQTAE` (11). The
scaffold's own residue is inside the 90% set at all 28 window positions, so this is a soft
prior for the design search, not a rewrite of the scaffolds.

![design window vs conservation](figures/design_window_vs_conservation.png)

### Where the window sits in the barrel-wide distribution

The EGFP window is 25 editable positions plus the 3 fixed ones. Placed against the
barrel-wide distribution of `C_chem` (quartiles 0.34 / 0.47 / 0.62), the window sits
slightly on the permissive side but is not selected for permissiveness: median
percentile 0.38, mean `C_chem` 0.459 against 0.500 for the barrel. Six of the 25
editable positions fall in the barrel's most permissive quartile and four in its most
constrained quartile. Ranked by family constraint the window spans nearly the full
range, from L61 (`C_chem` 1.00, 99th percentile) down to T66 (0.17, 3rd percentile):

| EGFP pos | family says | `C_chem` | current handling |
|---|---|---|---|
| L61 | 100% aliphatic, only `LI` at 90% mass | 1.00 | **unrestricted** |
| Y67 | 94.5% aromatic, `Y` alone at 90% mass | 0.88 | aromatic `{YWHF}` |
| Y93 | 84% aromatic, `YFML` | 0.71 | unrestricted |
| F47 | 73% aliphatic, `LFIMAY` | 0.69 | unrestricted |
| L45 | 79% aliphatic, `LMAIVGNS` | 0.63 | unrestricted |
| … | … | … | … |
| H149 | 45% polar, `THVESCFQ` | 0.04 | Tier-B H-bond |
| T66 | chromophore X, `QGMSTACWH` | 0.03 | unrestricted |

So for EGFP the practical readout is: one position to constrain (L61 → `LIVM`), one
where the family is tighter than the current rule but the current rule is deliberately
looser for good reason (Y67 — the family says Tyr, but `{Y,W,H,F}` is what buys you
cyan and blue), and two positions where Tier-B is imposing H-bond capability on columns
the family fills with something else (H149 at 0.75 mass kept, T204 at 0.87 — both
tolerable; EGFP escapes the severe cases like PA-GFP S206).

### The MSA is not redundant with the ESM-2 term (EGFP)

`esm_vs_family.py` compares the family distribution against the ESM-2 650M masked
marginal at each EGFP window position, generated exactly as `campaign.py` generates it
(`esm_logits_at`: mask one position of the unmodified scaffold, restrict to the 20
standard amino acids, softmax). The two signals turn out to be close to orthogonal:

- mean Spearman(family frequency, ESM-2 probability) across the 28 window positions =
  **+0.108**, negative at 11 of 28;
- ESM-2's top-1 matches the family's top-1 at **2 of 28** positions;
- mean family-class mass 0.637 vs 0.205 under ESM-2 — at 17 of 25 editable positions
  the family is decided by more than 0.25 of class mass where ESM-2 is not;
- the family's preferred residue is inside ESM-2's top-10 — the campaign's default `k`,
  i.e. the set it can actually sample from — at only **61%** of window positions.

The reason is not that the two disagree about which residue belongs; it is that ESM-2
has almost no opinion on this fold. A calibration panel (`results/esm_calibration.csv`)
makes that concrete, and it is the reason the comparison is reported at all:

| protein | length | masked top-1 | mean max prob | median rank of true residue | true residue in top-10 |
|---|---|---|---|---|---|
| ubiquitin | 76 | 0.803 | 0.754 | 1.0 | 1.00 |
| lysozyme | 129 | 0.659 | 0.682 | 1.0 | 0.99 |
| trypsin | 223 | 0.735 | 0.687 | 1.0 | 0.99 |
| **EGFP** | 239 | **0.126** | **0.128** | **6.0** | **0.74** |
| **avGFP** (wild type) | 238 | **0.101** | **0.125** | **6.0** | **0.73** |
| **mCherry** | 236 | **0.178** | **0.170** | **6.0** | **0.73** |

On three ordinary proteins spanning 76–223 residues ESM-2 650M recovers the true
residue as its top choice 66–80% of the time with mean confidence ~0.7. On three FPs of
comparable length it drops to 10–18% with mean confidence ~0.13, and its top prediction
is Gly at most positions — close to a background amino-acid distribution. This is not a
length effect (trypsin at 223 behaves like the short controls), not an
artifact of EGFP being engineered (wild-type avGFP is the *worst* of the three), and not
a bug in the harness (the same code path produces the sharp control numbers, and
unmasked reconstruction of EGFP is 99.2%).

**Why** is not established here. A plausible contributor is that the GFP superfamily is
taxonomically narrow and collapses to few UniRef50 clusters, so ESM-2 saw little of it
relative to its structural distinctiveness — but that is a hypothesis, not something
this analysis tests.

The consequence for the campaigns is concrete, though. `campaign.py` scores candidates
with `z(logp_ESM)` and restricts to `topk(logp, k=10)`; on FP scaffolds that term is
close to flat, so it is contributing far less ranking signal than it would on a typical
protein, and the top-10 truncation is discarding the family-preferred residue at 39% of
window positions. A per-position prior from `window_alphabet_egfp.csv` would supply
exactly the information ESM-2 is missing here, and it is cheap — a lookup table, no
extra forward passes. Worth checking on the existing sweep outputs before changing
anything: if λ_ex/λ_em dominate the score anyway, the flat ESM term may already be
effectively inert rather than actively harmful.

**Possible window expansion.** 16 columns within 10 Å of the chromophore have
`C_chem` < 0.35 and are outside the EGFP window — the family tolerates wide variation
there. The most interesting is **avGFP C70** (`C_chem` 0.199, RSA 0.00, 6.1 Å): fully
buried, second shell, and highly variable across the family. Others are 164, 166, 168,
202, 206, 207, 219, 223, 225. Note the trade-off honestly: these sit outside the 5 Å
contact shell, so they are less likely to move the spectrum directly than a first-shell
edit, and low family constraint means *tolerant*, not *effective*. They are candidates
for a second-shell tier, not a replacement for the current window.

**What this does not say.** Family conservation and design objective are different
things. The campaigns deliberately edit positions the family varies — that is precisely
how the family itself changes colour — so a low-`C_chem` position being editable is the
strategy working, not a warning. The comparison is only informative in one direction:
a position the whole family refuses to vary is a position a design is unlikely to
improve by varying, and there is exactly one of those currently left open.

### The same comparison over the whole barrel and the whole family

`esm_profiles.py` and `visualization.ipynb` extend that EGFP-window comparison to all 208
near-complete columns and to 96 proteins sampled across the alignment's 81 source organisms. The
window result is the general one:

- **Neither flat nor sharp about the same positions.** Family column entropy spans the full range
  (median 2.35 bits, 12% of columns below 1 bit); ESM-2's is pinned at the ceiling (median 4.09 of
  a possible 4.32, 98% above 3.5, none below 1). The two are uncorrelated column by column,
  Spearman ρ = +0.04 (p = 0.59), and top-1 choices agree at 14% of columns.
- **They diverge most where the family is most decided.** Jensen-Shannon divergence rises with
  `C_chem` (ρ = +0.70): 0.69 bits at the 27 chemistry-locked columns against 0.30 at the 55 most
  permissive, and 0.51 at buried positions against 0.38 at exposed ones.
- **The flatness is a property of the fold.** Mean masked-marginal confidence is 0.11–0.17 for all
  96 sampled FPs (Aequorea 0.12, Anthozoa 0.14, other 0.13) against 0.68–0.75 for the three
  controls. Nothing distinguishes EGFP.
- **Only the family predicts real FP sequences.** Scoring each sampled protein's own residues at the
  core columns under a *leave-one-out* family profile (its own Henikoff weight subtracted from every
  column it occupies) gives perplexity **4.9** and 53% top-1, against **16.0** and 15% for ESM-2 —
  which is barely better than the 18.2 obtained from the family's background composition alone. The
  family profile wins for 96 of 96 proteins.
- **The `k = 10` truncation is indiscriminate.** ESM-2 ranks the family's consensus residue 5th on
  median, and the campaign's top-10 keeps it at 72% of columns overall but only 63% of the
  chemistry-locked ones (61% across the EGFP window) — being certain in the family buys no
  protection.

The practical reading is the same as above, now with the whole barrel behind it: a per-position
family prior is not redundant with the ESM-2 term, and the cheapest fix is to *widen* the proposal
support to the union of ESM-2's top-10 with `window_alphabet_egfp.csv`, rather than to replace
either.

## What the figures establish, taken together

`plot_conservation.py` writes seven panels from `results/column_conservation.csv`, all restricted to
the 208 columns at occupancy ≥ 0.9. They are not seven separate observations — they are one argument
in three movements, and the third is what motivated
[`design-campaign-EGFP/msa-guided/`](../design-campaign-EGFP/msa-guided/).

**1 — the constraint is chemical, not identity-based.** `identity_vs_chemistry.png` plots every core
column as `C_id` vs `C_chem` against the diagonal. Points *on* it — G67, R96, F27, E222 — are
invariant either way and tell a designer only "don't touch". The interesting mass sits *above* it and
is almost entirely aliphatic: L18 (0.63 → 0.98), I161 (0.61 → 0.95), V12 (0.59 → 0.88), positions
where half the family disagrees about which residue goes there and essentially none of it disagrees
about what kind. `class_conservation_summary.png` turns that into counts and shows the ~2× gap holds
at every threshold (≥95%: 9 vs 16 · ≥90%: 14 vs 28 · ≥80%: 26 vs 46 · ≥70%: 34 vs 71), carried by
the aliphatics (mean `C_id` 0.45 → `C_chem` 0.58) while glycine and aromatics are conserved by
identity outright. `conservation_tracks.png` is the positional reference view, laying the same gap
along avGFP numbering against RSA, chromophore distance and the β-strands.

This is the clause that makes the family *usable* as a prior rather than merely descriptive. A
position conserved by identity yields one residue; a position conserved by chemistry alone yields an
alphabet of 4–12 residues this fold has actually tolerated in 763 real proteins — which is the shape
a proposal distribution needs.

**2 — the constraint comes from the fold, not the chromophore.** `chemistry_vs_burial.png` puts six
properties against RSA and everything moves coherently: charge constraint ρ = −0.640, polarity mean
+0.587, hydropathy mean −0.574, H-bond capacity +0.475, `C_chem` −0.403, volume −0.242. Buried means
greasy, apolar and — most strongly — electrostatically pinned. `pocket_vs_scaffold.png` runs the same
test against distance to the chromophore and is deliberately a null: ρ = −0.003, p = 0.96, with
median `C_chem` 0.417 in the ≤5 Å pocket against 0.464 in the 5–10 Å shell and 0.476 in the outer
barrel. The colour-tuning pocket is, if anything, the least constrained zone in the protein.
`pocket_composition.png` shows what that looks like residue by residue — a few solid single-class
bars in a field of mosaics.

**3 — therefore the family is orthogonal to the design window.** The window selects positions by
proximity to the chromophore; the family constrains positions by burial; the two are uncorrelated, so
the window's editable set is close to a random draw with respect to how hard the family constrains
it. `design_window_vs_conservation.png` is the consequence: the EGFP window sorted by family
constraint, coloured by what the window's own rules do with each position. L61 ties for the tallest
bar in the figure — `C_chem` = 1.000, two residues (`LI`) covering 90% of family mass, alongside the
catalytic R97 — and it is left completely unrestricted, while H149 (0.18) and T204 (0.37) carry
Tier-B H-bond restrictions at positions the family barely constrains. The lower panel gives the
actionable form: I168 needs 11 residues to cover 90% of family mass, L61 needs 2, and the window
treats them identically. (This panel is in **EGFP** numbering while the other six are in **avGFP**;
the offset past position 65 is +1 — see "One real gap: avGFP L60 / EGFP L61" above.)

### The one-sentence version

The FP family constrains **side-chain chemistry rather than residue identity**; that constraint is
imposed by the **fold**, not by the chromophore; and it is therefore **orthogonal to the geometric
criterion the design window was built from**. The window was specified on one axis and left the other
unspecified — a job implicitly delegated to ESM-2, which has no opinion on this family.

### Why the null result is the load-bearing panel

`pocket_vs_scaffold.png` reports no effect, and that is precisely its value. Had `C_chem` tracked
chromophore distance, the alignment would have been largely redundant with the 5 Å criterion — an
expensive route to information the window already encodes. ρ = −0.003 against ρ = −0.40 for burial is
what makes the family an *independent* source of constraint, and therefore what makes a
family-profile proposal worth building at all. The same question is asked a second time in
`visualization.ipynb`, against ESM-2 instead of against geometry: does this signal add to what we
already have? Both times the answer is yes, for unrelated reasons.

### What the figures do not establish

Beyond the modelling caveats at the end of this README:

- **Conservation is not function.** A zero-frequency residue means "763 observed sequences at
  N_eff = 272 never do this", not "this cannot fold". The hard support constraint the campaigns
  apply is stricter than the evidence strictly licenses — a deliberate choice, but a choice.
- **Nothing here is causal.** These panels show what the family avoids. Whether designs that ignored
  it are actually worse proteins is a question only the campaigns, and ultimately the bench, answer.

## Robustness

`validate.py` runs four tests; results in `results/validation.json`.

1. **Alignment integrity.** The chromophore X-[YWHF]-G tripeptide lands in the *same
   three columns* for **757/763 = 99.2%** of sequences, and the G67 column is Gly in
   99.9%. The six exceptions are real biology, not misalignment: `avGFP454`, `shBFP`
   and two shBFP mutants carry the engineered Y66L substitution, and `HriCFP`/`HriGFP`
   are 134-residue partial sequences. Independently, the R96 column is Arg in 762/763
   and the E222 column is Glu in 738/763.
2. **Redundancy.** Recomputing on one representative per 90%-identity cluster (117
   sequences, unweighted) reproduces the weighted full-set profile: corr(`C_chem`) =
   0.953, corr(`C_id`) = 0.961. Henikoff and cluster weightings also agree
   (0.950 / 0.956).
3. **Clade independence.** Splitting into Aequorea-lineage (n = 276, N_eff = 59) and
   non-Aequorea (n = 487, N_eff = 212) and recomputing each half separately:
   **all 28 chemistry-locked positions hold at ≥80% in both clades**. The per-position
   `C_chem` profiles themselves correlate only 0.46 between halves — expected, since the
   Aequorea half has low effective diversity and so inflates conservation broadly — but
   the specific locked set is fully reproduced on both sides.
4. **Threshold sensitivity.** The chemistry:identity ratio stays in 1.8–2.1 across all
   occupancy cutoffs (0.5–0.95) and all frequency thresholds (0.7–0.9).

## Outputs

| file | contents |
|---|---|
| `data/fp_all.fasta`, `data/fp_all_meta.csv` | 763-sequence input and metadata (the published union; tracked) |
| `data/fp_peak_only.*` | optional 758-sequence peak-only input from `build_msa_input.py --peak-only`; not published, not tracked |
| `data/fp_all.aln.fasta` | the MSA (763 × 1861) |
| `results/column_conservation.csv` | per-column table: occupancy, `C_id`, `C_chem`, per-class and per-residue weighted frequencies, six property means and ρ scores, avGFP position, RSA, chromophore distance, SSE |
| `results/sequence_qc.csv` | per-sequence `core_frac` (flags the fusion constructs) |
| `results/summary.json`, `results/validation.json`, `results/findings.txt` | run metadata, robustness tests, all quoted numbers |
| `figures/conservation_tracks.png` | conservation, chemistry class and structure along avGFP numbering |
| `figures/identity_vs_chemistry.png` | `C_id` vs `C_chem`, coloured by class |
| `figures/chemistry_vs_burial.png` | six property panels against RSA |
| `figures/pocket_vs_scaffold.png` | the absent chromophore-distance gradient |
| `figures/class_conservation_summary.png` | per-class conservation and the ~2× count gap |
| `figures/pocket_composition.png` | class composition of all 28 pocket positions |
| `results/window_vs_family_egfp.csv` | each of the 28 EGFP window positions scored against the family |
| `results/window_alphabet_egfp.csv` | evolution-derived 90%-mass residue alphabet per window position |
| `results/esm_vs_family_egfp.csv` | ESM-2 masked marginal vs family frequency at each EGFP window position |
| `results/esm_calibration.csv` | ESM-2 masked-marginal sharpness on FPs vs matched control proteins |
| `figures/design_window_vs_conservation.png` | the EGFP window ranked by family constraint |
| `results/esm_profiles.npz`, `esm_profiles_meta.json` | full (L, 20) ESM-2 masked marginals for avGFP, EGFP, mCherry and the three control proteins |
| `results/esm_sweep_profiles.npz`, `esm_family_sweep.csv` | the same for 96 proteins sampled across 81 source organisms, plus per-protein sharpness and divergence from the family |
| `visualization.ipynb`, `figures/esm_vs_msa_*.png` | the family-vs-ESM-2 distribution comparison, six figures (see below) |

## Caveats

- **`C_chem` measures constraint, not presence.** For a binary property such as
  aromaticity, a high ρ means the property is consistently *absent* just as often as
  consistently present; always read ρ together with the reported mean.
- **The class partition is a modelling choice.** His is filed under basic, so its
  aromaticity is only visible through the `aromatic` property column, not the class
  entropy. Coarser or finer partitions shift absolute `C_chem` values, though the
  ~2× identity/chemistry gap survives every threshold tested.
- **Structural annotations are avGFP-specific.** RSA and chromophore distances come from
  1GFL; a DsRed-family member's own barrel differs in detail, and columns where avGFP
  has a gap carry no structural annotation. Those columns are listed with `ref_pos = -1`
  in the CSV — the strongest of them is a 96.7%-conserved glycine occupied by 412
  sequences, essentially all anthozoan (Entacmaea, Discosoma, Lobophyllia, Clavularia),
  i.e. a clade-specific insertion just upstream of the chromophore that avGFP lacks.
- **Sequence weighting corrects redundancy, not sampling bias.** FPbase is a record of
  what laboratories have engineered; the underlying natural family is sampled unevenly
  no matter how the weights are set.
