#!/usr/bin/env python
"""Structure-based chromophore edit window (5 A contact shell) — experimental vs ESMFold.

Companion to ``guided_design_peak_structure_multiscaffold.ipynb``. That notebook
defines the editable window as the chromophore tripeptide plus every residue with
a heavy atom within CUTOFF (5 A) of the chromophore, taken from an *experimental*
RCSB structure and mapped onto the dataset sequence by alignment
(``_struct_pocket``). This module provides:

  * ``struct_pocket_experimental(name, seq)`` — the notebook's logic, factored out
    (fetches the RCSB structure, uses the mature hetero-chromophore CRQ/NRQ or the
    modelled tripeptide, aligns to the dataset sequence).
  * ``struct_pocket_esmfold(name, seq, pdb_path)`` — a DROP-IN replacement that
    reads a *local ESMFold prediction* of the exact dataset sequence. Because we
    fold the dataset sequence itself, the predicted-structure residue numbering
    equals the dataset numbering 1:1 — no RCSB fetch and no homolog alignment
    (the step eqFP578 currently needs against TagRFP 3M22). The chromophore
    reference is the tripeptide's heavy atoms (ESMFold outputs standard residues,
    not a fused chromophore).
  * ``compare(...)`` / CLI — quantify how the ESMFold-derived pocket differs from
    the experimental one per scaffold (Jaccard, added/dropped positions).

Drop-in use in the notebook (swap one function)::

    import pockets
    POCKET = {nm: pockets.struct_pocket_esmfold(nm, seqs[name2idx[nm]],
                                                f"structures/esmfold/{nm}.pdb")
              for nm in dict.fromkeys(s for s, _ in TASKS)}

IMPORTANT QUALITY CAVEAT — validated empirically (see repo notes): ESMFold predicts
the 11-stranded GFP-family beta-barrel POORLY (mean pLDDT ~30-44, CA-RMSD ~16-20 A
vs the experimental structures), while it folds alpha/beta control proteins well.
This matches Meta's own reference ESMFold (ESM Atlas API) and is reproducible across
torch versions, CPU/MPS, and independent implementations — i.e. it is an intrinsic
ESMFold limitation for this fold class, not an install artifact. Consequently the
ESMFold-derived pocket does NOT faithfully reproduce the experimental contact shell.
Prefer the experimental structures for defining the design window; treat the ESMFold
path as exploratory only.
"""
import os

import numpy as np
import biotite.sequence as _bseq
import biotite.sequence.align as _balign
import biotite.structure.io.pdb as _pdb
import biotite.structure.io.pdbx as _pdbx
import biotite.database.rcsb as _rcsb

# ---- constants shared with the notebook -------------------------------------------
_CANON3 = "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
_T3 = {c: o for c, o in zip(_CANON3, "A R N D C Q E G H I L K M F P S T W Y V".split())}
_MAT = _balign.SubstitutionMatrix.std_protein_matrix()
CUTOFF = 5.0
CHROMO_AA2 = set("YWHF")   # position-2 ring: Tyr/Trp/His/Phe

# scaffold -> (pdb_id, chromophore hetero-residue codes; () -> derive from tripeptide)
STRUCT = {"DsRed": ("1G7K", ("CRQ",)),
          "avGFP": ("1GFL", ()),
          "eqFP578": ("3M22", ("NRQ",))}
# buried maturation-catalytic Arg + Glu (dataset 1-based) -> PROTECTED / excluded
CATALYTIC = {"DsRed": (95, 215), "avGFP": (96, 222), "eqFP578": (92, 215)}
STRUCTDIR = "structures"
ESMFOLD_DIR = os.path.join("structures", "esmfold")


# ---- chromophore location (identical to the notebook) -----------------------------
def chromophore_index(seq, lo=50, hi=85):
    """0-based index of chromophore position 1 (the X before the pos2-[YWHF]-Gly)."""
    yg = [i for i in range(len(seq) - 2) if seq[i + 1] == "Y" and seq[i + 2] == "G" and lo <= i <= hi]
    any_ = [i for i in range(len(seq) - 2) if seq[i + 1] in CHROMO_AA2 and seq[i + 2] == "G" and lo <= i <= hi]
    cand = yg or any_
    if not cand:
        raise ValueError("no X-[YWHF]-G chromophore motif near canonical position")
    return min(cand, key=lambda i: abs(i - 65))


def _first_chain_heavy(arr):
    return arr[(arr.chain_id == sorted(set(arr.chain_id))[0]) & (arr.element != "H")]


def _resid_letters(arr):
    """First-occurrence res_id list + one-letter sequence over standard residues."""
    resids, letters, seen = [], [], set()
    for rid, rn in zip(arr.res_id.tolist(), arr.res_name.tolist()):
        if rid in seen:
            continue
        seen.add(rid)
        if rn in _T3:
            resids.append(int(rid))
            letters.append(_T3[rn])
    return resids, "".join(letters)


def _pocket_from_atoms(a, ccoord, s2d, tri, catal, cutoff):
    """Given first-chain heavy atoms `a`, chromophore coords `ccoord`, a
    structure-res_id -> dataset-index map `s2d`, the tripeptide dataset indices
    `tri` and catalytic dataset indices `catal`, return sorted 0-based pocket."""
    prot = a[np.isin(a.res_name, _CANON3)]
    pocket = []
    for rid in dict.fromkeys(prot.res_id.tolist()):
        if rid not in s2d:
            continue
        rc = prot[prot.res_id == rid].coord
        dmin = np.sqrt(((rc[:, None, :] - ccoord[None, :, :]) ** 2).sum(-1)).min()
        if dmin <= cutoff:
            dp = s2d[rid]
            if dp not in tri and dp not in catal:
                pocket.append(dp)
    return sorted(set(pocket))


# ---- experimental-structure pocket (notebook logic, factored out) -----------------
def struct_pocket_experimental(name, seq, cutoff=CUTOFF, structdir=STRUCTDIR):
    """Editable pocket (sorted 0-based dataset indices) from the RCSB structure."""
    pdb, het = STRUCT[name]
    arr = _pdbx.get_structure(_pdbx.CIFFile.read(_rcsb.fetch(pdb, "pdbx", structdir)), model=1)
    a = _first_chain_heavy(arr)
    resids, letters = _resid_letters(a)
    aln = _balign.align_optimal(_bseq.ProteinSequence(letters), _bseq.ProteinSequence(seq),
                                _MAT, gap_penalty=(-10, -1))[0]
    tr = aln.trace
    s2d = {resids[tr[r, 0]]: int(tr[r, 1]) for r in range(tr.shape[0]) if tr[r, 0] >= 0 and tr[r, 1] >= 0}
    d2s = {v: k for k, v in s2d.items()}
    c1 = chromophore_index(seq)
    tri = {c1, c1 + 1, c1 + 2}
    cmask = np.isin(a.res_name, het) if het else np.isin(a.res_id, [d2s[p] for p in tri if p in d2s])
    ccoord = a[cmask].coord
    catal = {p - 1 for p in CATALYTIC[name]}
    return _pocket_from_atoms(a[~cmask], ccoord, s2d, tri, catal, cutoff)


# ---- ESMFold-prediction pocket (DROP-IN; no fetch, no alignment) -------------------
def struct_pocket_esmfold(name, seq, pdb_path=None, cutoff=CUTOFF):
    """Editable pocket from a LOCAL ESMFold prediction of the exact dataset `seq`.

    Predicted-structure residue i (1-based) == dataset position i, so no homolog
    alignment is needed. The chromophore reference is the tripeptide's heavy atoms
    (ESMFold has no fused chromophore hetero-atoms). Signature mirrors the
    notebook's ``_struct_pocket`` so it is drop-in.
    """
    if pdb_path is None:
        pdb_path = os.path.join(ESMFOLD_DIR, f"{name}.pdb")
    arr = _pdb.PDBFile.read(pdb_path).get_structure(model=1)
    a = _first_chain_heavy(arr)
    c1 = chromophore_index(seq)
    tri = {c1, c1 + 1, c1 + 2}
    tri_1based = [c1 + 1, c1 + 2, c1 + 3]           # PDB res_id == dataset 1-based position
    # sanity: predicted PDB should cover the dataset sequence 1:1
    present = set(a.res_id.tolist())
    missing = [p for p in tri_1based if p not in present]
    if missing:
        raise ValueError(f"{name}: ESMFold PDB missing chromophore res_id(s) {missing} "
                         f"(pdb res range {min(present)}-{max(present)}, seq len {len(seq)})")
    ccoord = a[np.isin(a.res_id, tri_1based)].coord   # tripeptide heavy atoms
    s2d = {p: p - 1 for p in present}                  # identity map (1-based -> 0-based)
    cmask = np.isin(a.res_id, tri_1based)
    catal = {p - 1 for p in CATALYTIC[name]}
    return _pocket_from_atoms(a[~cmask], ccoord, s2d, tri, catal, cutoff)


# ---- comparison --------------------------------------------------------------------
def _fmt(pos):
    return ",".join(str(p + 1) for p in pos)   # 1-based for human reading


def compare(name, seq, esmfold_pdb=None, cutoff=CUTOFF, structdir=STRUCTDIR):
    """Return a dict comparing the experimental vs ESMFold 5 A pocket for `name`."""
    exp = set(struct_pocket_experimental(name, seq, cutoff, structdir))
    esm = set(struct_pocket_esmfold(name, seq, esmfold_pdb, cutoff))
    inter = exp & esm
    union = exp | esm
    return {
        "name": name,
        "n_exp": len(exp),
        "n_esm": len(esm),
        "n_shared": len(inter),
        "jaccard": (len(inter) / len(union)) if union else 1.0,
        "recall_of_exp": (len(inter) / len(exp)) if exp else float("nan"),  # frac of exp pocket recovered
        "only_exp": sorted(exp - esm),   # in experimental, missed by ESMFold
        "only_esm": sorted(esm - exp),   # spurious in ESMFold
        "exp": sorted(exp),
        "esm": sorted(esm),
    }


def main():
    import argparse
    import csv

    ap = argparse.ArgumentParser(description="Compare ESMFold vs experimental 5 A chromophore pocket.")
    ap.add_argument("--cur", default=os.path.join("..", "dataset_pipeline", "data", "peak", "curated"))
    ap.add_argument("--esmfold-dir", default=ESMFOLD_DIR)
    ap.add_argument("--cutoff", type=float, default=CUTOFF)
    ap.add_argument("--names", nargs="*", default=list(STRUCT.keys()))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.cur, "peaks_assignments.csv"))))
    n = len(rows)
    seqs = [None] * n
    h = None
    for line in open(os.path.join(args.cur, "sequences.fasta")):
        line = line.strip()
        if line.startswith(">"):
            h = int(line[1:].split("|")[0])
        elif line:
            seqs[h] = line
    name2idx = {r["name"]: i for i, r in enumerate(rows)}

    print(f"5 A chromophore-pocket comparison (experimental RCSB vs ESMFold prediction), cutoff={args.cutoff} A\n")
    print(f"{'scaffold':9}{'PDB':6}{'n_exp':>6}{'n_esm':>6}{'shared':>7}{'Jaccard':>9}{'exp recall':>11}")
    for nm in args.names:
        seq = seqs[name2idx[nm]]
        pdb_path = os.path.join(args.esmfold_dir, f"{nm}.pdb")
        c = compare(nm, seq, pdb_path, cutoff=args.cutoff, structdir=STRUCTDIR)
        print(f"{nm:9}{STRUCT[nm][0]:6}{c['n_exp']:>6}{c['n_esm']:>6}{c['n_shared']:>7}"
              f"{c['jaccard']:>9.2f}{c['recall_of_exp']:>10.0%}")
    print()
    for nm in args.names:
        seq = seqs[name2idx[nm]]
        pdb_path = os.path.join(args.esmfold_dir, f"{nm}.pdb")
        c = compare(nm, seq, pdb_path, cutoff=args.cutoff, structdir=STRUCTDIR)
        print(f"--- {nm} (1-based dataset positions) ---")
        print(f"  experimental pocket ({c['n_exp']}): {_fmt(c['exp'])}")
        print(f"  ESMFold pocket      ({c['n_esm']}): {_fmt(c['esm'])}")
        print(f"  recovered by ESMFold ({c['n_shared']}): {_fmt(c['exp'] and sorted(set(c['exp']) & set(c['esm'])))}")
        print(f"  missed by ESMFold   ({len(c['only_exp'])}): {_fmt(c['only_exp'])}")
        print(f"  spurious in ESMFold ({len(c['only_esm'])}): {_fmt(c['only_esm'])}")


if __name__ == "__main__":
    main()
