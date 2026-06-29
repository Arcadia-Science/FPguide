# Attempt — multi-target from a fixed green scaffold (λ = 5, 15)

**Date:** 2026-06-26 (early run)
**Artifact:** `designs.fasta` (only the FASTA survived; the CSV was overwritten by later runs)

## Design process
- **Scaffold:** a single fixed **green** FP (avGFP-like, emission ≈ 510 nm), reused for every target.
- **Targets:** a spread of FPs across the spectrum, split by whether the **surrogate** saw them:
  - in-sample (`Srole==train`): **EBFP** (em 440, blue), **hmGFP** (em 510, green), **mEos4b** (em 580, red)
  - OOD (`Srole==test`): **oxBFP** (em 448, blue), **YuzuFP** (em 511, green), **mEos3.2** (em 580, red)
- **Generator:** ESM-2 masked-LM windowed refinement, **one sweep** over a 40-residue window centered on the scaffold; candidates scored by `naturalness + λ·cosine(surrogate spectrum, target)`.
- **Guidance objective:** per-half **cosine similarity** (excitation|emission), surrogate = `cnn-max`, oracle = `cnn-max`.
- **λ sweep:** {5, 15}.

## Result (oracle cosine of the design vs target)
| target | em (nm) | role | λ5 | λ15 |
|---|---|---|---|---|
| EBFP | 440 | in-sample | 0.171 | 0.163 |
| hmGFP | 510 | in-sample | **0.842** | **0.819** |
| mEos4b | 580 | in-sample | 0.162 | 0.161 |
| oxBFP | 448 | OOD | 0.260 | 0.255 |
| YuzuFP | 511 | OOD | **0.966** | **0.974** |
| mEos3.2 | 580 | OOD | 0.201 | 0.166 |

## Takeaway
From a green scaffold the guided edit matches **same-color (green)** targets well (cos 0.82–0.97) but **fails on large color jumps** to blue (~0.17–0.26) or red (~0.16–0.20), regardless of λ. Motivated switching to **sequence-similar scaffold→target pairs** so the spectral gap is achievable (see the seqsim-pairs attempt).
