# experiment/ — wet-lab validation of the shortlisted EGFP designs

The wet-lab side of the campaign. [`design-campaign-EGFP/`](../design-campaign-EGFP)
shortlisted ten EGFP variants aimed at EBFP and mOrange
([`shortlists/FPdesign-batch1.xlsx`](../design-campaign-EGFP/shortlists/FPdesign-batch1.xlsx));
this folder holds the plasmids they were ordered as, and the plate-reader spectra of the resulting
*E. coli* cultures.

Two things live here:

- **[`plasmid constructs/`](plasmid%20constructs)** — the 13 ordered pET-28a maps (10 designs +
  3 references), the physical form of batch 1.
- **[`0831_spectrum/`](0831_spectrum)** — the 8/31/2026 plate read of those cultures: nine
  excitation/emission scans, two OD600 checks, and two notebooks that turn them into figures.

Earlier plate reads (8/25, 8/26, 8/27) and the notebook that tracked them across days are retired
to `archive/` — untracked, kept locally, per the repo-wide `archive/` rule. Nothing here reads or
cites them.

## Identity: clone IDs

The 8/31 notebooks label their designs `B1`, `C1`, … `C2`. Those are **delivery-plate well
locations** — not read-plate positions, and not design names. Nothing in the notebooks or the
FASTAs resolves them, so the mapping is recorded here, from the vendor's delivery record for
plate `pSHPs0807B2335016N`. Its three controls (`A1`, `D2`, `E2`) match the clone IDs the
notebook's own control legend already carries ("EGFP (clone D2)", "EBFP (clone A1)",
"mOrange (clone E2)"), which is what ties that delivery to this plate read.

The full chain, delivery well → construct → read-plate well:

| clone | construct | target | read-plate well | notebook label |
|---|---|---|---|---|
| A1 | `His-eBFP` | — (reference) | B2 | EBFP (clone A1) |
| D2 | `eGFP_taglessctrl` | — (scaffold ref) | B1 | EGFP (clone D2) |
| E2 | `His-mOrange` | — (reference) | B3 | mOrange (clone E2) |
| B1 | `His-eBFP-MSA_01` | EBFP | C1 | design B1 |
| C1 | `His-eBFP-MSA_02` | EBFP | C2 | design C1 |
| D1 | `His-eBFP_MSA_06` | EBFP | C3 | design D1 |
| E1 | `His-mOrange_MSA_01` | mOrange | D1 | design E1 |
| F1 | `His-mOrange_MSA_03` | mOrange | D2 | design F1 |
| G1 | `His-mOrange_MSA_04` | mOrange | D3 | design G1 |
| **H1** | **`His-mOrange_MSA_07`** | **mOrange** | **D4** | **design H1** |
| A2 | `His-mOrange_MSA_10` | mOrange | D5 | design A2 |
| B2 | `His-mOrange_MSAgib_01` | mOrange | D6 | design B2 |
| C2 | `His-mOrange_MSAgib_02` | mOrange | D7 | design C2 |

Three coordinate systems, and two of them collide by coincidence — clone `B1` is a design, well
`B1` is the EGFP control. **Always say which.** This README writes clone IDs bare (`H1`) and
read-plate wells as "well D4".

All 13 passed the vendor's NGS and yield QC on delivery. Every insert goes in at
`NcoI_XhoI_no_HIS_no_thrombin`, so the vector's own His-tag and thrombin site are bypassed and the
His6 is encoded in the insert itself — which is why tag position varies by construct rather than
by vector.

## plasmid constructs/ — batch 1 as ordered

Thirteen single-record FASTAs, each the **complete circular pET-28a(+) map** (~5,985 bp) rather
than an insert. Translating the ORF out of each one and diffing against the shortlist confirms all
thirteen: twelve carry a shortlisted protein sequence **verbatim** downstream of an
`MG-HHHHHH-GSGS-` N-terminal leader (251 aa total for a 239-aa design). The thirteenth differs:

| file | header | protein | tag |
|---|---|---|---|
| `his-morange_msa_01/03/04/10`, `(bright)his-morange_msa_07` | `His-mOrange_MSA_0*` | the 5 mOrange-target MSA-guided designs | N-term His6 |
| `his-morange_msagib_01/02` | `His-mOrange_MSAgib_0*` | the 2 mOrange-target MSA-gibbs designs | N-term His6 |
| `his-ebfp-msa_01`, `his-ebfp-msa_02`, `his-ebfp_msa_06` | `His-eBFP-MSA_0*` | the 3 EBFP-target MSA-guided designs | N-term His6 |
| `his-morange`, `his-ebfp` | `His-mOrange`, `His-eBFP` | the two target references | N-term His6 |
| `egfp_ctermtag` | `eGFP_taglessctrl` | EGFP (Gly inserted after the initiator Met) | **C-term**: `MG`+EGFP[2:]+`LEHHHHHH` |

Note the last row's filename and header disagree: the file is `egfp_ctermtag`, the record header is
`eGFP_taglessctrl`, and the translated ORF puts the His6 at the **C-terminus**. It carries no
N-terminal tag; it is not untagged.

The ten designs and what they were predicted to do, from
[`FPdesign-batch1.xlsx`](../design-campaign-EGFP/shortlists/FPdesign-batch1.xlsx):

| construct | target | strategy | mutations vs EGFP | predicted peak MAE | brightness logit |
|---|---|---|---|---|---|
| mOrange_MSA_01 | mOrange | MSA guide + bright | 9 | 3.4 nm | 1.09 |
| mOrange_MSA_03 | mOrange | MSA guide + bright | 6 | 4.1 nm | 0.78 |
| mOrange_MSA_04 | mOrange | MSA guide + bright | 9 | 4.2 nm | 1.21 |
| mOrange_MSA_07 | mOrange | MSA guide + bright | 6 | 4.8 nm | 0.90 |
| mOrange_MSA_10 | mOrange | MSA guide + bright | 14 | 5.1 nm | 1.55 |
| EBFP_MSA_01 | EBFP | MSA guide + bright | 8 | 12.5 nm | 0.76 |
| EBFP_MSA_02 | EBFP | MSA guide + bright | 11 | 14.6 nm | 3.11 |
| EBFP_MSA_06 | EBFP | MSA guide + bright | 12 | 18.8 nm | 2.30 |
| mOrange_MSAgib_01 | mOrange | MSA gibbs | 21 | 29.2 nm | −11.70 |
| mOrange_MSAgib_02 | mOrange | MSA gibbs | 20 | 32.1 nm | −9.22 |

The two MSA-gibbs rows come from the campaign's unguided arm and are the only two predicted
not-bright and out-of-distribution.

## 0831_spectrum/ — the 8/31 plate read

One SoftMax Pro run, `##BLOCKS= 11`: nine spectral scans plus two OD600 endpoint reads. The `.xls`
is the raw UTF-16 export; the `_raw_well_data.csv` beside it is already tidy per
(plate, well, wavelength), so the notebooks read the CSV and leave the `.xls` as the provenance
copy.

**Plate layout.** Row B is a control row every scan re-reads; each scan adds either row C
(3 designs) or row D (7 designs):

| row | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| **B** (every scan) | EGFP `D2` | EBFP `A1` | mOrange `E2` | pUC (empty vector) | PBS (blank) | | |
| **C** (2 scans) | eBFP-MSA_01 `B1` | eBFP-MSA_02 `C1` | eBFP_MSA_06 `D1` | | | | |
| **D** (7 scans) | mOr-MSA_01 `E1` | mOr-MSA_03 `F1` | mOr-MSA_04 `G1` | mOr-MSA_07 `H1` | mOr-MSA_10 `A2` | mOr-MSAgib_01 `B2` | mOr-MSAgib_02 `C2` |

Row C holds the three EBFP-target designs, row D the seven mOrange-target ones. The two row-C
scans sweep 300–430 and 380–500 nm; the seven row-D scans sweep 350–600 nm. Backticked labels are
clone IDs; they do **not** encode well position (clone `B1` sits in well C1). The notebooks plot
the clone ID, so use the table above to read a legend.

**The nine windows.** Plate names encode the fixed channel: `Em480` is an *excitation* sweep with
emission held at 480 nm; `Ex340` is an *emission* sweep with excitation held at 340 nm.

| Fig | plate | sweep | fixed | range | wells |
|---|---|---|---|---|---|
| 1 | `p1: Em480` | excitation | em 480 | 300–430 nm | 8 (row C) |
| 2 | `p2: Ex340` | emission | ex 340 | 380–500 nm | 8 (row C) |
| 3 | `p1: Em580` | excitation | em 580 | 350–540 nm | 12 (row D) |
| 4 | `p2: Em520` | excitation | em 520 | 350–480 nm | 12 |
| 5 | `p3: Em560` | excitation | em 560 | 350–520 nm | 12 |
| 6 | `p4: Ex390` | emission | ex 390 | 450–600 nm | 12 |
| 7 | `p5: Ex460` | emission | ex 460 | 490–550 nm | 12 |
| 8 | `p1: Ex500` | emission | ex 500 | 540–600 nm | 12 |
| 9 | `p2: Em560` | excitation | em 560 | 470–520 nm | 12 |

Figs 10–11 are the two OD600 blocks, drawn as plate-view heatmaps.

### The two notebooks

**`spectra_normalization.ipynb`** — the survey. One figure per scan, every well that scan actually
read, log y-axis (linear for Figs 1–2). Values are **raw instrument readings (a.u.)**, not
OD-normalized, and the notebook is explicit about why: this run's two OD600 blocks occupy a
different region of the plate than the fluorescence wells (`4x diluted 200uL` covers rows A–C ×
cols 1–12; `4x diluted 100 uL` covers rows C–D × cols 1–8), so there is no well-for-well
correspondence to normalize against. They are shown as culture-density QC and nothing more.
**Curves are comparable within a plate, not across plates.**

**`design_D4_vs_EGFP_avGFP.ipynb`** — a single-well workup. Well D4 — clone `H1`,
**`His-mOrange_MSA_07`** — on the `p3: Em560` / `p4: Ex390` pair, the pair whose sweep ranges
bracket this well's peaks. Both curves are **background-subtracted by well B4** (pUC,
plain *E. coli*, no FP) wavelength-for-wavelength before peak-normalizing; the subtraction is
asserted positive everywhere, so nothing is clipped. The measured EGFP control (well B1) gets the
identical treatment, so it is comparable to D4 directly and not only to the FPbase reference.

Peak wavelengths the notebook prints, alongside the shortlist's prediction for the same construct:

| | excitation | emission |
|---|---|---|
| `mOrange_MSA_07` (well D4), measured | 390 nm (secondary 500) | 510 nm |
| EGFP control (well B1), measured | 490 nm | 520 nm |
| mOrange target, FPbase | 548 nm | 562 nm |
| `mOrange_MSA_07`, surrogate prediction | 540.2 nm | 563.8 nm |

mOrange's target peaks come from FPbase, not from this run's own mOrange well (B3): B3 sits on the
same two scans as D4, but `p3: Em560` only sweeps 350–520 nm (548 nm is off the end) and
`p4: Ex390` excites 158 nm away from mOrange's optimum, so B3's background-subtracted signal on
this pair goes partly negative and is not usable as a target.

### Residue numbering

A conversion table, because two numbering systems are in play and they differ by one. The shortlist
counts from the initiator Met (M = 1); the GFP literature uses avGFP numbering, and EGFP's extra
Val at position 2 accounts for the offset. Both columns are checked against this repo's own
sequences (`FPdesign-batch1.xlsx` and FPbase's avGFP export), for `mOrange_MSA_07`:

| shortlist | conventional | avGFP | EGFP | design |
|---|---|---|---|---|
| F47L | F46L | F | F | L |
| L65F | F64L | F | L | F |
| T66M | S65M | S | T | M |
| V69S | V68S | V | V | S |
| Y146F | Y145F | Y | Y | F |
| T204H | T203H | T | T | H |

The same −1 offset applies to every construct in this batch. Shortlist positions 66–68 are the
chromophore tripeptide (Thr-Tyr-Gly in EGFP), conventionally numbered 65–67.

## Layout

```
plasmid constructs/*.fasta          13 complete pET-28a(+) maps, ~5,985 bp, one record each
0831_spectrum/
  20260831_jeb_zz_FP.xls            raw SoftMax Pro UTF-16 export (11 blocks) — provenance copy
  20260831_jeb_zz_FP_raw_well_data.csv   tidy (plate, well, wavelength, value) — what both
                                    notebooks actually read
  spectra_normalization.ipynb       Figs 1–9 (per-scan survey) + Figs 10–11 (OD600 heatmaps)
  design_D4_vs_EGFP_avGFP.ipynb     Figs 1–3 (well D4, background-subtracted, vs EGFP/avGFP)
  figures/                          both notebooks write here
archive/                            8/25, 8/26, 8/27 reads + ebfp_design_tracking.ipynb (untracked)
```

**The two notebooks share one `figures/` and both number from Fig1.** Nothing collides, because
the slug after the number differs (`Fig1_p1-Em480-Excitation-scan_Float` vs
`Fig1_design-D4-H1_EGFP-overlay_Float`), but the numbers alone are ambiguous — cite figures here
by full filename. Survey figures are written as PDF+PNG; the two D4 overlays are SVG+PNG (Fig1,
Fig3) and the stacked panel is PDF+PNG.

Reproduce, from `0831_spectrum/`:

```bash
jupyter lab spectra_normalization.ipynb     # then Run All
jupyter lab design_D4_vs_EGFP_avGFP.ipynb
```

Both need the repo-wide figure style: Arcadia colors via `arcadia_pycolor`, and the Atkinson
Hyperlegible faces at `../../dataset_pipeline/fonts/` (untracked — `python
dataset_pipeline/fetch_arcadia_fonts.py` from the repo root). Missing fonts degrade to a
substitute with a printed warning rather than failing. `design_D4_vs_EGFP_avGFP.ipynb`
additionally reads `../../fpbase-extractor/fpbase_output/fpbase_spectra.json` for the EGFP,
avGFP, and mOrange reference curves.

## Caveats

- **The notebooks name clones, not constructs.** Their figures and legends carry `H1`, `B2`, …
  and nothing in either notebook, the FASTAs, or the shortlist resolves those to a construct. The
  [Identity](#identity-clone-ids) table above is the only copy of that mapping held here; the
  vendor delivery record it came from is not in this folder. Without it every figure is unreadable,
  so keep the table with the data if either ever moves.
- **Raw a.u., one plate at a time.** Nothing in `spectra_normalization.ipynb` is OD-corrected, so
  a taller curve may just be a denser culture. Only `design_D4_vs_EGFP_avGFP.ipynb` subtracts
  background, and only within its own two scans.
- **One well, one read, no replicates.** Only `mOrange_MSA_07` (well D4) has a dedicated
  background-subtracted workup; the other nine designs appear in the survey figures only. Nothing
  here is repeated across days or wells.
- **`experiment/` is tracked.** The plate data, plasmid FASTAs, figures and both notebooks are
  committed: they are primary records that cannot be regenerated, so they are published with the
  rest of the repo rather than kept local like `archive/`.
