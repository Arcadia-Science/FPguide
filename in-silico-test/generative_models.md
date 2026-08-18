# The two generative models: family MSA and ESM-2

The design search never enumerates sequences. At each step it needs a *proposal*: given a
scaffold and one editable position, which amino acids are worth putting there at all, and how
plausible is each. That proposal is the only generative component in the pipeline — everything
downstream (surrogate, oracle, selection) is discriminative. This document covers why there are
two of them, how each turns into a per-residue probability, and how the alignment was built.

Code: `archive/3.1_design_run_MSA/design_knownstruct.py`,
[3.1_design_run_ESM2/design_knownstruct_esm2.py](3.1_design_run_ESM2/design_knownstruct_esm2.py),
[2_design_task_specification/build_windows.py](2_design_task_specification/build_windows.py),
[msa/conservation.py](msa/conservation.py), and the alignment build in
[../msa_conservation/](../msa_conservation/).

## 1. Why a generative model is needed, and why exactly one thing is swapped

A 230-residue FP has ~20²³⁰ sequences. Structure narrows that to the chromophore contact shell —
14–34 editable positions per scaffold — which is still ~13²⁶. Within that window the surrogate
alone is not a safe search signal: it saw a few hundred FPs, so a search free to pick any residue
finds sequences that score well *because* they are off-manifold. The generative term is what keeps
the walk inside protein-like sequence space, and it is also what supplies the candidate set the
surrogate is allowed to rank.

Both arms run the identical search — same 108 tasks, same windows, same surrogate and oracle, same
seed, same 3 trials, same random visit order, same score:

```
score(candidate) = z(logp_proposal) − 1.0·z(|λ_ex_pred − λ_ex_target|) − 1.0·z(|λ_em_pred − λ_em_target|)
```

with the z-scores taken over the k = 10 candidates at the current position and the substitution
drawn by multinomial sampling at T = 1. The only difference between the arms is where
`logp_proposal` comes from — one line in the loop
(`design_knownstruct.py:324` vs
[design_knownstruct_esm2.py:432](3.1_design_run_ESM2/design_knownstruct_esm2.py#L432)). That makes
the comparison a controlled swap rather than two pipelines that happen to differ.

The design window is shared but is not entirely model-neutral, so it is worth being precise about
what "same window" means. `design_windows.json` carries two separable things:

| part | content | ESM-2 arm |
|---|---|---|
| **structural** | editable set (chromophore positions 1–2 + the 5 Å pocket); per-position alphabet constraints — aromatics `{Y,W,H,F}` at chromophore position 2, H-bond-capable residues at chromophore H-bond partners | kept verbatim |
| **family** | those alphabets intersected with what the alignment supports at that column, plus the frequencies themselves | **replaced** |

So in the ESM-2 arm a position's candidate set is its structural constraint, or all 20 amino acids
where it has none, and the ranking over that set is ESM-2's. The family PSSM is still loaded there,
but only to compute the `fam_logp` diagnostic — it gates nothing and ranks nothing.

## 2. Model A — the family MSA profile (PSSM)

### 2.1 How the alignment was made

Built once in [`../msa_conservation/`](../msa_conservation/) and vendored into
[msa/data/fp_all.aln.fasta](msa/data/fp_all.aln.fasta) (byte-identical copy, so the windows here
are reproducible from this folder alone).

**Input.** `build_msa_input.py` takes the union of the three curated trait sets — peak (758) plus
what brightness and pKa add (2 + 3) = **763 unique sequences**, 82 source organisms. The union of
the *curated* sets, not the raw FPbase export, is the right universe for a family alignment:
curation has already dropped the biliverdin/phytochrome near-infrared class, which does not share
the GFP fold and would contribute nothing but junk columns.

The 763 vs 758 gap is worth stating explicitly, since 758 is the peak set every surrogate and
oracle here is trained and tested on. The 5 extra sequences are proteins the *peak* curation
rejected but that carry a brightness or pKa measurement, so the union readmits them as sequences
without a peak label: four analyte sensors (`CAR-GECO1`, `mKeima`, `pHluorin4`, `pHmScarlet`),
whose (ex, em) is ill-defined because the peak moves with analyte concentration, and one
unresolvable multi-state entry (`PSLSSmKate`). They contribute to column frequencies and to
nothing else. Note also that the alignment covers the peak set *including* its held-out test
proteins — but it uses only their sequences, never their ex/em labels, so the PSSM carries no
label information into the surrogate or the oracle.

**Alignment.** MAFFT **v7.526** (2024-04-26), FFT-NS-i (progressive + iterative refinement,
`--maxiterate 1000`, BLOSUM62, default gap parameters op = 1.53 / offset = 0.12; the refinement
converged well inside the cap) rather than `--auto`, which at N = 763 falls back to the single-pass
FFT-NS-2; the refinement measurably tightens the barrel columns. `run_msa.sh` pins
`--thread 1 --randomseed 0`, because MAFFT's
parallel refinement visits subalignments in nondeterministic order and multithreaded reruns of the
same input differ by a few columns (1959 vs 1965 observed), which shifts downstream counts by ±1.
Single-threaded is bit-identical run to run at ~4 min instead of ~30 s. Input order is preserved
(no `--reorder`) so alignment rows stay keyed to `fp_all_meta.csv` by `msa_id`.

**Result.** 763 × 1861 columns, of which **233 are core** (occupancy ≥ 50%) and 208 near-complete
(≥ 90%) — matching the 238-residue avGFP barrel. The sparse remainder is insertions from the 18
tandem-dimer and Ca²⁺-sensor constructs (`tdTomato`, `tdStayGold`, `GCaMP2`, `RCaMP`, …) whose
second barrel copy cannot align to the first.

**Software.** MAFFT 7.526 for the alignment itself; Biopython 1.87 (`AlignIO`) to parse it, NumPy
1.26.4 and pandas 2.3.3 for the weighting and frequency calculations, all in the `spectrum-to-fp-design`
conda environment. The proposal side uses fair-esm 2.0.0 (`esm2_t33_650M_UR50D`) on PyTorch 2.13,
with biotite 1.2.0 for the structure→sequence mapping that defines the window.

### 2.2 Sequence weighting — the step that matters most

This dataset is a mutant library, not a phylogenetic sample: 276 of 763 sequences are
Aequorea-lineage, most of them avGFP point mutants. Raw column frequencies would report
"conserved" for anything the avGFP lineage happens to share, and would hand the design a proposal
distribution that is really a census of what people have already published.

Every frequency used here is therefore computed under **Henikoff & Henikoff position-based
weights** ([msa/conservation.py:121](msa/conservation.py#L121)): a sequence's weight is the mean
over its own non-gap columns of `1/(r·n)`, where `r` is the number of distinct residue types in the
column and `n` the count of this sequence's residue there. Averaging over each sequence's *own*
occupied columns rather than all columns keeps partial sequences from being penalized for being
short. Weights are computed on the 233 core columns and normalized to mean 1.

This cuts the effective sample size from 763 to **N_eff = 272**. Clustering at 90% identity (117
clusters, N_eff = 173) is computed as an independent cross-check; per-column conservation under the
two weightings correlates at r = 0.96 (identity) / 0.95 (chemistry), so the result is not an
artifact of the weighting scheme.

### 2.3 From alignment column to per-residue probability

For each scaffold, `build_windows.py` locates its own row in the alignment (exact sequence match,
and the row must ungap to exactly the scaffold length — otherwise the scaffold is rejected, since a
window is a set of indices *into* that sequence and a mis-mapped column would produce a wrong
window rather than a worse one). That gives `col_of[p]`: alignment column for sequence position
`p`. Then for every editable position
([build_windows.py:138-155](2_design_task_specification/build_windows.py#L138-L155)):

1. `f = F[col_of[p]]` — the 20-vector of Henikoff-weighted frequencies at that column, gaps
   excluded from the denominator.
2. `support = {a : f[a] > 0}` — residues the family is actually observed to use there.
3. `keep = support ∩ constraint`, where `constraint` is the structural alphabet at that position
   (aromatics at chromophore 2, the 11 H-bond-capable residues at H-bond partners) or unrestricted.
4. `probs = f[keep] / Σ f[keep]` — renormalized over the survivors.

The result is stored per position as `{alphabet, probs}` and read at design time as
`logp = log(probs)` with `−inf` outside the alphabet. There are **no pseudocounts**: a residue the
weighted family never uses at that column has probability zero and is unreachable. That is
deliberate — the point of this arm is a proposal that cannot leave family support — but it is also
the sharpest difference from ESM-2 and the reason the two arms' edit sets diverge.

Empty intersections fall back to the constraint alone, or the wild-type residue. That path is
**never taken**: 0 fallbacks across all 3,443 editable positions in `design_windows.json` (0 of
2,702 restricted to the 108-task cohort). Across the 108 scaffolds the resulting alphabets run 2–20
residues, median 13.

At search time the PSSM is a **static lookup**: `logits = pssm_vec[pos]`, no forward pass, no
dependence on what the design currently looks like. Editing position 145 does not change the
proposal at position 203. Consequently the visit order matters only through which candidates the
surrogate scores first, and `fam_logp` — the design's log-likelihood under its own scaffold's
PSSM — is free to compute every round.

### 2.4 The same profile in `design-campaign-EGFP/`

The wet-lab campaign's `msa-guided/` and `msa-gibbs/` strategies use the **same** prior, built by
[`../design-campaign-EGFP/msa-guided/build_msa_pssm.py`](../design-campaign-EGFP/msa-guided/build_msa_pssm.py):
it imports `henikoff_weights` / `weighted_freqs` from `msa_conservation/conservation.py` and reads
`msa_conservation/data/fp_all.aln.fasta` directly (referenced, not copied — the same 763 × 1861
alignment, byte-identical to this folder's vendored copy), weights on the same 233 core columns to
the same N_eff = 272, maps through EGFP's own alignment row, intersects with the same Tier-B
alphabet constraints, and drops zero-frequency residues without smoothing. Every statement in §2.1–
§2.3 about how a per-residue probability is produced therefore holds verbatim for the campaign. Its
window is the single EGFP Tier-B window (25 editable positions; alphabet size min 4 / median 12 /
max 20, blocking 211 of 500 position-residue combinations), and it raises on an empty intersection
rather than falling back, which never triggers.

**What is *not* shared is the sampling scale**, and this matters if the two settings are described
in one breath:

| | in-silico benchmark (3.1 / 3.2) | `design-campaign-EGFP` |
|---|---|---|
| top-k | 10 | 10 |
| temperature | 1.0 both arms | **1.0** MSA arms, **10.0** ESM-2 guided arms (1.0 for ESM-2 gibbs) |
| λ_ex / λ_em | 1.0 / 1.0 both arms | **1.0** MSA-guided, **20 / 20** ESM-2 guided |
| extra score terms | none | brightness (λ_bright) and edit penalty (λ_edit) on some strategies |
| trials × cycles | 3 × 2, 108 scaffold→target tasks | 12 × 3, EGFP → EBFP / mOrange |

So the benchmark is a clean controlled swap — proposal only, everything else pinned including the
seed — whereas in the EGFP campaign the λ scale was retuned at the same time as the proposal was
replaced (the z-scored family term already has unit variance, so the inherited λ = 20 / T = 10
defaults would have swamped it). `msa-gibbs/` exists precisely to separate those two changes.
Claims of the form "only the proposal differs" are true of the benchmark and **not** of the
campaign.

## 3. Model B — ESM-2 650M masked-LM

`esm2_t33_650M_UR50D`, final layer 33, 1280-dim. The proposal is the **masked marginal** at the
edited position ([design_knownstruct_esm2.py:270-284](3.1_design_run_ESM2/design_knownstruct_esm2.py#L270-L284)):

1. Tokenize the design's **current** sequence (scaffold plus every edit made so far this cycle).
2. Replace the token at the edited position with `<mask>` (`+1` for the BOS token).
3. Forward pass; take the logit row at that column.
4. Mask out everything outside the position's structural alphabet — or outside the 20 standard
   amino acids where the position has no constraint — with `−inf`.
5. `log_softmax` over what remains, then `topk(k = 10)`.

So `p(a | position)` is `p_ESM2(x_p = a | x_{−p})`, a single-position pseudo-likelihood, not a
joint. It is single-sequence: ESM-2 sees no alignment and no homologs, only this one sequence.

Two consequences follow from the conditioning that do not apply to the PSSM arm:

- **Order dependence.** Because the proposal reads the current sequence, the order in which
  positions are visited changes the distribution at every later position, not merely the order in
  which candidates get scored. The random per-trial visit order is doing more work here.
- **Cost.** One forward pass per (design, position, step), batched at 64 across tasks — 25 min vs
  21 min for 324 searches. The pseudo-perplexity diagnostic (mask every residue in turn) costs
  about a whole design cycle per round and is off by default (`--no-ppl`); `fam_logp` is still
  written, so the naturalness axis survives.

ESM-2 also encodes the **surrogate's** input representation (layer-33 residue embeddings under a
`cnn-max-d1` head) in both arms. In the ESM-2 arm the proposal and the surrogate therefore share a
representation, a coupling the MSA arm does not have. The oracle is deliberately outside that
family — a `cnn-max-d1` head over **ProstT5** residue embeddings — so the held-out judge does not
inherit the proposal's or the surrogate's blind spots.

## 4. Why both — the two priors are close to orthogonal on this family

The obvious expectation is that a 650M-parameter protein language model subsumes a 763-sequence
profile. On FPs it does not, and the reason is measurable:
[`../msa_conservation/esm_vs_family.py`](../msa_conservation/esm_vs_family.py) puts the two
distributions side by side at each EGFP window position, generating the ESM-2 side through exactly
the campaign's code path.

- mean Spearman(family frequency, ESM-2 probability) over the 28 window positions = **+0.108**,
  negative at 11 of 28;
- ESM-2's top-1 matches the family's top-1 at **2 of 28** positions;
- mean family chemistry-class mass 0.637 vs 0.205 under ESM-2;
- the family's preferred residue is inside ESM-2's top-10 — the set the search can actually sample
  from — at only **61%** of window positions.

The disagreement is not two opinions in conflict; it is one model having almost no opinion. The
calibration panel (`results/esm_calibration.csv`) is what makes that legible:

| protein | length | masked top-1 acc. | mean max prob | median rank of true residue |
|---|---|---|---|---|
| ubiquitin | 76 | 0.803 | 0.754 | 1.0 |
| lysozyme | 129 | 0.659 | 0.682 | 1.0 |
| adenylate kinase | 214 | 0.706 | 0.692 | 1.0 |
| **EGFP** | 239 | **0.126** | **0.128** | **6.0** |
| **avGFP** (wild type) | 238 | **0.101** | **0.125** | **6.0** |
| **mCherry** | 236 | **0.178** | **0.170** | **6.0** |

On ordinary proteins spanning 76–214 residues ESM-2 650M recovers the true residue as its top
choice 66–80% of the time at ~0.7 confidence. On three FPs of comparable length it drops to 10–18%
at ~0.13, with Gly as its top prediction at most positions — close to a background amino-acid
distribution. Not a length effect (adenylate kinase at 214 behaves like the short controls), not an
engineering artifact (wild-type avGFP is the worst of the three), not a harness bug (the same code
produces the sharp control numbers, and unmasked reconstruction of EGFP is 99.2%). *Why* is not
established here; a taxonomically narrow superfamily collapsing to few UniRef50 clusters is a
plausible contributor, but this analysis does not test it.

That is the case for running both rather than picking one on priors: the two arms are the two
plausible answers to "what does a protein-like residue look like here," they carry different
information, and on this fold the general-purpose model is the flatter of the two.

## 5. What the swap actually bought

108 tasks, 3 trials each, identical windows/models/weights/seed; oracle-scored, mean absolute peak
error in nm.

| | scaffold | design (surrogate-selected) | mean of trials | improved | trial spread | identity | fam_logp/pos |
|---|---|---|---|---|---|---|---|
| MSA PSSM | 133.2 | **87.2** | 94.0 | 103/108 | 28.1 nm | 0.922 | −2.11 |
| ESM-2 | 133.2 | **89.2** | 92.8 | 97/108 | 24.4 nm | 0.903 | −3.80 |

**On aim, the two are indistinguishable.** The sign of the difference depends on which statistic
you pick — MSA is 2.0 nm better on the surrogate-selected design (ESM-2 wins 48/108, Wilcoxon
*p* = 0.30), ESM-2 is 1.2 nm better on the mean over trials (55/108, *p* = 0.69) — and both gaps
are ~10× smaller than the 24–28 nm within-task spread across trials of the *same* arm. Any
single-trial comparison of these arms was measuring its own noise.

What the trials do not wash out is the trade:

- ESM-2 edits **less conservatively** — 90.3% vs 92.2% identity to scaffold, family log-likelihood
  1.8× worse per position. The PSSM arm cannot leave family support by construction (verified: 0
  of 11,627 edits outside it); ESM-2 puts **3,863 of 14,463 edits (27%) outside** the
  family-supported alphabet.
- **6 fewer tasks improve at all** (97 vs 103) — the wider proposal costs some reliability.
- ESM-2's trials are **less variable** (24.4 vs 28.1 nm), the one axis on which it is cleanly
  ahead: a sequence-conditioned proposal is more repeatable than a static profile sampled at T = 1.

Read all of this against the unguided null in
[3.2_design_run_gibbs/](3.2_design_run_gibbs/), not against zero: resampling the pocket from the
same proposal with the surrogate switched off already reaches 105.8 nm, so ~65% of the
133.2 → 87.2 gain is the window and the prior, not the guidance. The same control shows that
ESM-2's looser editing and worse family likelihood are properties of the *proposal* — unguided
ESM-2 already sits at 0.902 identity and −100.9 fam_logp — not consequences of steering it.

## 6. Limits worth stating

- The MSA prior is a **per-position marginal**. It carries no epistasis: it cannot express that
  position 203 should be Thr *given* that 145 is Phe, which is precisely the kind of coupling that
  sets a chromophore environment. A pair-coupled model (Potts/DCA) would, and is the obvious next
  arm.
- Zero pseudocounts make the MSA arm's support strictly closed. Every design it produces is
  family-plausible by construction, which also means it can never propose the substitution that
  is right but unobserved.
- ESM-2's masked marginal is a pseudo-likelihood evaluated on a partially-edited sequence, so its
  "naturalness" claim degrades as a cycle proceeds and the conditioning context drifts from
  anything the model was trained on.
- Neither prior knows anything about fluorescence. Both are constrained to a structurally defined
  window and screened by the surrogate precisely because the generative term cannot tell a folded
  dark protein from a bright one; that job is left to the window's fixed positions (chromophore
  position 3, the catalytic Arg/Glu) and, in the EGFP campaigns, to a separate DMS-trained
  brightness head.
