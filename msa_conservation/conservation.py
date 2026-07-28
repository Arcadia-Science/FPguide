#!/usr/bin/env python
"""Per-column chemistry conservation across the whole-FP-dataset MSA.

Reads ``data/fp_all.aln.fasta`` (763 curated FP sequences, MAFFT FFT-NS-i) and writes
``results/column_conservation.csv`` + ``results/summary.json``.

Three things make the numbers mean what they claim to mean:

1. **Redundancy correction.** The dataset is a mutant library, not a phylogenetic
   sample: hundreds of near-identical avGFP and DsRed descendants. Unweighted column
   frequencies would report "conserved" for anything the avGFP lineage happens to
   share. Every statistic here is computed under Henikoff position-based sequence
   weights, which drop the effective sample size from 763 to ~N_eff. Cluster weights
   (1/size at 90% identity) are computed as an independent check.

2. **Background normalization.** Conservation is reported as the fraction of the
   family's own uncertainty removed at a column, ``1 - H_col / H_background``, using
   the weighted amino-acid composition of the alignment core as background. A raw
   entropy would call an all-Leu column and an all-Trp column equally conserved even
   though Leu is 5x more likely a priori.

3. **Identity vs chemistry, separately.** ``C_id`` is over the 20 amino acids;
   ``C_chem`` is over 8 side-chain classes. Their difference isolates the positions
   this analysis is about: chemistry pinned, identity free.

Continuous properties (hydropathy, volume, charge, aromaticity, H-bonding, polarity)
get a variance-reduction score ``rho = 1 - Var_col / Var_background`` on the same
"fraction of family variability removed" footing, so scales are comparable.

Structural context (RSA, distance to chromophore, secondary structure) comes from
the local wild-type avGFP crystal structure 1GFL, mapped onto the dataset avGFP
sequence by alignment rather than by trusting PDB residue numbering.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import biotite.sequence as bseq
import biotite.sequence.align as balign
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from Bio import AlignIO

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STRUCT_PATH = REPO / "esm2_design" / "structures" / "1GFL.pdbx"

OCC_MIN = 0.50        # column occupancy to count as barrel core
CLUSTER_ID = 0.90     # single-linkage identity for the cluster-weight cross-check
POCKET_CUTOFF = 5.0   # A, chromophore contact shell
REF_SLUG = "avgfp"    # reference numbering: wild-type A. victoria GFP

AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AAS)}

# Side-chain chemistry classes (exclusive partition). G, P and C are their own
# classes because their distinguishing chemistry is backbone flexibility, backbone
# rigidity and a thiol -- not anything shared with the bulk groups. His is filed
# under basic here; its aromaticity is captured by the `aromatic` property instead.
CLASSES = {"aliphatic": "AVLIM", "aromatic": "FWY", "polar": "STNQ",
           "acidic": "DE", "basic": "KRH", "glycine": "G", "proline": "P", "cysteine": "C"}
CLASS_OF = {a: c for c, aas in CLASSES.items() for a in aas}
CLASS_NAMES = list(CLASSES)
CLASS_VEC = np.array([CLASS_NAMES.index(CLASS_OF[a]) for a in AAS])

# Property scales, indexed by AAS order.
PROPS = {
    # Kyte & Doolittle 1982 hydropathy
    "hydropathy": dict(zip("ACDEFGHIKLMNPQRSTVWY",
                           [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                            1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3])),
    # Zamyatnin 1972 side-chain volume, A^3
    "volume": dict(zip("ACDEFGHIKLMNPQRSTVWY",
                       [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6, 166.7,
                        162.9, 114.1, 112.7, 143.8, 173.4, 89.0, 116.1, 140.0, 227.8, 193.6])),
    # formal side-chain charge at pH 7 (His partially protonated)
    "charge": dict(zip("ACDEFGHIKLMNPQRSTVWY",
                       [0, 0, -1, -1, 0, 0, 0.1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])),
    # aromatic ring present (His included -- imidazole)
    "aromatic": dict(zip("ACDEFGHIKLMNPQRSTVWY",
                         [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])),
    # side-chain H-bond donors + acceptors
    "hbond": dict(zip("ACDEFGHIKLMNPQRSTVWY",
                      [0, 1, 2, 2, 0, 0, 2, 0, 3, 0, 0, 3, 0, 3, 5, 2, 2, 0, 1, 2])),
    # Grantham 1974 polarity
    "polarity": dict(zip("ACDEFGHIKLMNPQRSTVWY",
                         [8.1, 5.5, 13.0, 12.3, 5.2, 9.0, 10.4, 5.2, 11.3, 4.9,
                          5.7, 11.6, 8.0, 10.5, 10.5, 9.2, 8.6, 5.9, 5.4, 6.2])),
}
PROP_VECS = {p: np.array([s[a] for a in AAS], float) for p, s in PROPS.items()}

# Tien et al. 2013 theoretical max SASA, for relative solvent accessibility
MAX_SASA = dict(zip("ACDEFGHIKLMNPQRSTVWY",
                    [129, 167, 193, 223, 240, 104, 224, 197, 236, 201,
                     224, 195, 159, 225, 274, 155, 172, 174, 285, 263]))
_T3 = dict(zip("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
               "A R N D C Q E G H I L K M F P S T W Y V".split()))


# ---- alignment loading ------------------------------------------------------------
def load_alignment():
    aln = AlignIO.read(HERE / "data" / "fp_all.aln.fasta", "fasta")
    ids = [int(r.id.split("|")[0]) for r in aln]
    A = np.array([list(str(r.seq).upper()) for r in aln])
    meta = pd.read_csv(HERE / "data" / "fp_all_meta.csv").set_index("msa_id").loc[ids].reset_index()
    return A, meta


def encode(A):
    """Alignment -> integer codes (0..19 for the 20 AAs, -1 for gap/other)."""
    code = np.full(A.shape, -1, np.int8)
    for a, i in AA_IDX.items():
        code[A == a] = i
    return code


# ---- sequence weighting ----------------------------------------------------------
def henikoff_weights(code):
    """Henikoff & Henikoff position-based weights over the given columns.

    A sequence's weight is the mean over its own non-gap columns of 1/(r * n), where r
    is the number of distinct residue types in the column and n the count of this
    sequence's residue there. Averaging over each sequence's *own* occupied columns
    (rather than all columns) keeps partial sequences from being down-weighted merely
    for being short.
    """
    n_seq, n_col = code.shape
    total = np.zeros(n_seq)
    for c in range(n_col):
        col = code[:, c]
        obs = col >= 0
        if not obs.any():
            continue
        counts = np.bincount(col[obs], minlength=20)
        types = (counts > 0).sum()
        total[obs] += 1.0 / (types * counts[col[obs]])
    occ = (code >= 0).sum(1)
    w = np.where(occ > 0, total / np.maximum(occ, 1), 0.0)
    return w / w.mean()


def identity_matrix(code):
    """Pairwise fractional identity over columns where both sequences have a residue."""
    onehot = np.zeros((code.shape[0], code.shape[1], 20), np.float32)
    obs = code >= 0
    ii, jj = np.nonzero(obs)
    onehot[ii, jj, code[ii, jj]] = 1.0
    flat = onehot.reshape(code.shape[0], -1)
    matches = flat @ flat.T
    both = obs.astype(np.float32) @ obs.astype(np.float32).T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(both > 0, matches / np.maximum(both, 1), 0.0)


def cluster_weights(code, thresh=CLUSTER_ID):
    """1/cluster_size weights from single-linkage clustering at `thresh` identity."""
    ident = identity_matrix(code)
    n = ident.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.nonzero(np.triu(ident >= thresh, 1))):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    labels = np.array([find(i) for i in range(n)])
    _, inv, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    w = 1.0 / sizes[inv]
    return w / w.mean(), sizes[inv], len(sizes)


def n_eff(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


# ---- column statistics -----------------------------------------------------------
def weighted_freqs(code, w):
    """(n_col, 20) weighted amino-acid frequencies, gaps excluded from the denominator."""
    n_col = code.shape[1]
    F = np.zeros((n_col, 20))
    for c in range(n_col):
        col = code[:, c]
        obs = col >= 0
        if obs.any():
            F[c] = np.bincount(col[obs], weights=w[obs], minlength=20)
    tot = F.sum(1, keepdims=True)
    return np.divide(F, tot, out=np.zeros_like(F), where=tot > 0)


def entropy(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(P > 0, P * np.log2(P), 0.0).sum(-1)


def class_freqs(F):
    C = np.zeros((F.shape[0], len(CLASS_NAMES)))
    for k in range(len(CLASS_NAMES)):
        C[:, k] = F[:, CLASS_VEC == k].sum(1)
    return C


def prop_stats(F, vec):
    """Weighted mean and variance of a property under each column's aa distribution."""
    mean = F @ vec
    var = F @ (vec ** 2) - mean ** 2
    return mean, np.maximum(var, 0.0)


# ---- structural context ----------------------------------------------------------
def structure_features(ref_seq):
    """Per-position RSA, chromophore distance and SSE for the reference sequence.

    1GFL residue numbering is not assumed: the modelled chain-A sequence is aligned to
    the dataset avGFP sequence and features are transferred through that mapping.
    """
    cif = pdbx.CIFFile.read(STRUCT_PATH)
    arr = pdbx.get_structure(cif, model=1)
    a = arr[(arr.chain_id == "A") & struc.filter_amino_acids(arr) & ~arr.hetero & (arr.element != "H")]
    res_ids, res_names = struc.get_residues(a)
    letters = "".join(_T3.get(n, "X") for n in res_names)

    aln = balign.align_optimal(bseq.ProteinSequence(ref_seq), bseq.ProteinSequence(letters),
                              balign.SubstitutionMatrix.std_protein_matrix(),
                              gap_penalty=(-10, -1))[0]
    trace = aln.trace
    ref2struct = {int(t[0]): int(t[1]) for t in trace if t[0] >= 0 and t[1] >= 0}

    sasa = struc.sasa(a, vdw_radii="Single", point_number=300)
    res_sasa = struc.apply_residue_wise(a, sasa, np.nansum)
    rsa = np.array([res_sasa[k] / MAX_SASA.get(letters[k], np.nan) for k in range(len(letters))])

    sse = struc.annotate_sse(a)  # one label per amino-acid residue of the chain
    sse = np.array(list(sse)) if len(sse) == len(letters) else np.full(len(letters), "")

    # chromophore = the X-[YWHF]-G tripeptide nearest canonical position 65
    cands = [i for i in range(len(letters) - 2)
             if letters[i + 1] in "YWHF" and letters[i + 2] == "G" and 50 <= i <= 85]
    chromo_k = min(cands, key=lambda i: abs(i - 64))
    chromo_ids = set(res_ids[chromo_k:chromo_k + 3].tolist())
    chromo_atoms = a[np.isin(a.res_id, list(chromo_ids))]
    dist = []
    for k in range(len(letters)):
        atoms = a[a.res_id == res_ids[k]]
        d = np.linalg.norm(atoms.coord[:, None, :] - chromo_atoms.coord[None, :, :], axis=-1)
        dist.append(float(d.min()))
    dist = np.array(dist)

    out = {}
    for ref_pos, k in ref2struct.items():
        out[ref_pos] = dict(pdb_resid=int(res_ids[k]), pdb_aa=letters[k], rsa=float(rsa[k]),
                            chromo_dist=float(dist[k]), sse=str(sse[k]))
    return out, [int(res_ids[chromo_k + j]) for j in range(3)], letters


# ---- main ------------------------------------------------------------------------
def main():
    A, meta = load_alignment()
    code_full = encode(A)
    occ_full = (code_full >= 0).mean(0)
    core = np.nonzero(occ_full >= OCC_MIN)[0]
    code = code_full[:, core]
    print(f"alignment {A.shape[0]} seq x {A.shape[1]} col   core columns (occ>={OCC_MIN:.2f}): {len(core)}")

    # per-sequence share of residues placed inside the core -> flags fusions/fragments
    res_in_core = (code >= 0).sum(1)
    res_total = (code_full >= 0).sum(1)
    meta["core_frac"] = res_in_core / res_total

    w_hen = henikoff_weights(code)
    w_clu, clu_size, n_clu = cluster_weights(code)
    print(f"sequence weights: N_eff(Henikoff) {n_eff(w_hen):.1f}, "
          f"clusters at {CLUSTER_ID:.0%} id {n_clu}, N_eff(cluster) {n_eff(w_clu):.1f}")

    F = weighted_freqs(code, w_hen)
    Fc = class_freqs(F)
    F_clu = weighted_freqs(code, w_clu)

    # background = weighted composition pooled over core columns
    bg = (F * (code >= 0).sum(0)[:, None]).sum(0)
    bg /= bg.sum()
    bg_class = np.array([bg[CLASS_VEC == k].sum() for k in range(len(CLASS_NAMES))])
    H_bg, H_bg_class = float(entropy(bg)), float(entropy(bg_class))

    H_aa, H_class = entropy(F), entropy(Fc)
    C_id = 1 - H_aa / H_bg
    C_chem = 1 - H_class / H_bg_class
    C_id_clu = 1 - entropy(F_clu) / H_bg
    C_chem_clu = 1 - entropy(class_freqs(F_clu)) / H_bg_class

    df = pd.DataFrame({
        "aln_col": core, "occupancy": occ_full[core],
        "n_obs": (code >= 0).sum(0),
        "top_aa": [AAS[i] for i in F.argmax(1)], "top_aa_freq": F.max(1),
        "top_class": [CLASS_NAMES[i] for i in Fc.argmax(1)], "top_class_freq": Fc.max(1),
        "H_aa": H_aa, "H_class": H_class,
        "C_id": C_id, "C_chem": C_chem, "C_id_cluster_w": C_id_clu, "C_chem_cluster_w": C_chem_clu,
        "chem_minus_id": C_chem - C_id,
        "n_aa_seen": (F > 0).sum(1), "n_aa_above_5pct": (F >= 0.05).sum(1),
    })

    prop_bg_var = {}
    for p, vec in PROP_VECS.items():
        mean, var = prop_stats(F, vec)
        bg_mean = float(bg @ vec)
        bg_var = float(bg @ vec ** 2 - bg_mean ** 2)
        prop_bg_var[p] = bg_var
        df[f"{p}_mean"] = mean
        df[f"{p}_rho"] = 1 - var / bg_var
    df["aa_composition"] = [
        ";".join(f"{AAS[k]}:{F[i, k]:.3f}" for k in np.argsort(-F[i])[:5] if F[i, k] > 0.01)
        for i in range(len(core))]
    # full (not truncated) per-class and per-residue weighted frequencies
    for k, cls in enumerate(CLASS_NAMES):
        df[f"f_class_{cls}"] = Fc[:, k]
    for k, a in enumerate(AAS):
        df[f"f_aa_{a}"] = F[:, k]

    # ---- reference numbering + structure ----
    ref_row = meta.index[meta.slug == REF_SLUG][0]
    ref_seq = meta.loc[ref_row, "seq"]
    ref_aln = A[ref_row]
    ref_pos = np.cumsum(ref_aln != "-") - 1              # 0-based ref index per aln column
    df["ref_pos"] = np.where(ref_aln[core] != "-", ref_pos[core] + 1, -1)   # 1-based avGFP
    df["ref_aa"] = [c if c != "-" else "" for c in ref_aln[core]]

    feats, chromo_ids, struct_letters = structure_features(ref_seq)
    for col, key, default in [("rsa", "rsa", np.nan), ("chromo_dist", "chromo_dist", np.nan),
                              ("sse", "sse", ""), ("pdb_resid", "pdb_resid", -1)]:
        df[col] = [feats.get(int(p) - 1, {}).get(key, default) if p > 0 else default
                   for p in df.ref_pos]
    df["buried"] = df.rsa < 0.20
    df["in_pocket"] = df.chromo_dist <= POCKET_CUTOFF

    HERE.joinpath("results").mkdir(exist_ok=True)
    df.to_csv(HERE / "results" / "column_conservation.csv", index=False)
    meta.to_csv(HERE / "results" / "sequence_qc.csv", index=False)

    summary = dict(
        n_seq=int(A.shape[0]), n_col_total=int(A.shape[1]), n_core=int(len(core)),
        occ_min=OCC_MIN, n_organisms=int(meta.parent_organism.nunique()),
        n_eff_henikoff=n_eff(w_hen), n_clusters_90=int(n_clu), n_eff_cluster=n_eff(w_clu),
        H_background=H_bg, H_background_class=H_bg_class,
        background_composition={a: float(bg[i]) for i, a in enumerate(AAS)},
        background_class_composition={CLASS_NAMES[k]: float(bg_class[k]) for k in range(len(CLASS_NAMES))},
        prop_background_var=prop_bg_var,
        ref_slug=REF_SLUG, ref_len=len(ref_seq), chromophore_pdb_resids=chromo_ids,
        corr_C_id_weightings=float(np.corrcoef(C_id, C_id_clu)[0, 1]),
        corr_C_chem_weightings=float(np.corrcoef(C_chem, C_chem_clu)[0, 1]),
        n_invariant_C_id_gt_0_95=int((C_id > 0.95).sum()),
        n_chem_locked_class_freq_gt_0_9=int((df.top_class_freq > 0.9).sum()),
    )
    with open(HERE / "results" / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"weighting cross-check: corr(C_id) {summary['corr_C_id_weightings']:.3f}, "
          f"corr(C_chem) {summary['corr_C_chem_weightings']:.3f}")
    print("-> results/column_conservation.csv, results/sequence_qc.csv, results/summary.json")


if __name__ == "__main__":
    main()
