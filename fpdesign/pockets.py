#!/usr/bin/env python
"""Structure-based chromophore edit window (5 A contact shell) from EXPERIMENTAL structures.

Companion to ``guided_design_peak_structure_multiscaffold.ipynb`` and the
known-structure / conventional design campaigns. The editable window is the
chromophore tripeptide plus every residue with a heavy atom within CUTOFF (5 A)
of the chromophore, taken from an *experimental* RCSB structure and mapped onto
the dataset sequence by alignment. This module provides:

  * ``struct_pocket_experimental(name, seq)`` — the original notebook logic for
    the three hand-curated scaffolds (fetches the RCSB structure, uses the mature
    hetero-chromophore CRQ/NRQ or the modelled tripeptide, aligns to the dataset
    sequence).
  * ``experimental_window(name, seq, pdb_id)`` — the GENERALIZED version used by
    the known-structure cohort and the conventional campaign: works for any FP
    whose experimental PDB id is known, auto-detecting the chromophore (fused
    HETATM or tripeptide) and the catalytic Arg+Glu, with a local-alignment
    quality gate so structures that don't cleanly contain the FP are rejected
    rather than silently mis-mapped.
"""
import os

import numpy as np
import biotite.sequence as _bseq
import biotite.sequence.align as _balign
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


# ---- generalized experimental pocket (ANY RCSB PDB, auto chromophore + catalytic) --
_SOLVENT = {"HOH", "WAT", "DOD"}


def _chain_letters(a, chain_id):
    c = a[a.chain_id == chain_id]
    return _resid_letters(c)


def experimental_window(name, seq, pdb_id, cutoff=CUTOFF, min_id=0.90, min_cov=0.70,
                        return_quality=False, return_hbond=False, hbond_cutoff=3.5,
                        structdir=os.path.join("structures", "experimental")):
    """(c1, catal_0based, pocket_0based) from an arbitrary experimental RCSB structure.

    Generalizes ``struct_pocket_experimental`` to any FP whose PDB id is known (no
    per-scaffold hetero-code / catalytic table). Steps:

      1. fetch the mmCIF, pick the chain whose modelled sequence best matches ``seq``
         via LOCAL alignment (handles FPs embedded in larger fusion/complex chains);
      2. map that chain's residues to ``seq`` from the local-alignment trace
         (structure res_id -> dataset 0-based index);
      3. locate the chromophore reference atoms:
           * if the tripeptide (c1, c1+1) is modelled as standard residues -> their
             heavy atoms;
           * else (mature fused chromophore = a HETATM inside the residue-number gap
             between the residues flanking c1) -> that hetero-residue's heavy atoms;
      4. editable pocket = standard residues with a heavy atom within ``cutoff`` of the
         chromophore, minus the tripeptide and minus the catalytic pair;
      5. catalytic pair = the Arg and the Glu (by dataset letter) nearest the
         chromophore (generalizes GFP R96/E222).

    Return signature grows with the flags (backward compatible):
      * default                                 -> (c1, catal, pocket)
      * return_quality                          -> (c1, catal, pocket, quality)
      * return_hbond                            -> (c1, catal, pocket, hbond)
      * return_quality and return_hbond         -> (c1, catal, pocket, quality, hbond)
    where ``hbond`` (Tier-B) is the sorted 0-based list of pocket residues whose SIDE-CHAIN
    polar (N/O) atom lies within ``hbond_cutoff`` (default 3.5 A) of a chromophore polar
    (N/O) atom -- i.e. likely hydrogen-bond partners of the chromophore, excluding the
    tripeptide and the catalytic pair. Heavy-atom distance only (no explicit H / angle /
    ordered-water bridges), so treat it as an H-bond *capability* proxy, not ground truth.

    QUALITY GATE: the best chain must match the dataset sequence with local identity
    >= ``min_id`` over >= ``min_cov`` * len(seq) aligned residues. If no chain clears
    the gate the deposited structure does not cleanly contain the FP (e.g. a split-FP
    biosensor, a partial/scrambled model, or the wrong RCSB entity) and a ``ValueError``
    is raised rather than emitting a silently mis-mapped window.
    """
    cif = _pdbx.CIFFile.read(_rcsb.fetch(pdb_id, "pdbx", structdir))
    arr = _pdbx.get_structure(cif, model=1)
    arr = arr[arr.element != "H"]
    # pick the chain whose standard-residue sequence LOCAL-aligns best to the dataset seq
    best = None                                         # (matches, chain, aln, idv, cov)
    for ch in sorted(set(arr.chain_id.tolist())):
        _, letters = _chain_letters(arr, ch)
        if len(letters) < 20:
            continue
        aln = _balign.align_optimal(_bseq.ProteinSequence(letters), _bseq.ProteinSequence(seq),
                                    _MAT, gap_penalty=(-10, -1), local=True)[0]
        tr = aln.trace
        naln = matches = 0
        for r in range(tr.shape[0]):
            if tr[r, 0] >= 0 and tr[r, 1] >= 0:
                naln += 1
                if letters[tr[r, 0]] == seq[tr[r, 1]]:
                    matches += 1
        idv = matches / naln if naln else 0.0
        cov = naln / len(seq)
        if best is None or matches > best[0]:
            best = (matches, ch, aln, idv, cov)
    if best is None:
        raise ValueError(f"{name}[{pdb_id}]: no modelled protein chain to align")
    _, chain_id, aln, idv, cov = best
    if idv < min_id or cov < min_cov:
        raise ValueError(
            f"{name}[{pdb_id}]: best chain {chain_id} matches dataset seq at only "
            f"local id {idv:.0%} / coverage {cov:.0%} (need >= {min_id:.0%} id, "
            f">= {min_cov:.0%} cov) -- structure does not cleanly contain this FP")
    a = arr[arr.chain_id == chain_id]
    resids, letters = _resid_letters(a)                 # standard residues only
    tr = aln.trace
    s2d = {resids[tr[r, 0]]: int(tr[r, 1]) for r in range(tr.shape[0]) if tr[r, 0] >= 0 and tr[r, 1] >= 0}
    d2s = {v: k for k, v in s2d.items()}

    c1 = chromophore_index(seq)
    tri0 = {c1, c1 + 1, c1 + 2}

    # chromophore reference atoms
    if c1 in d2s and (c1 + 1) in d2s:                   # tripeptide modelled as standard residues
        tri_resids = [d2s[p] for p in (c1, c1 + 1, c1 + 2) if p in d2s]
        ccoord = a[np.isin(a.res_id, tri_resids)].coord
        chromo_resids = set(tri_resids)
    else:                                               # mature fused chromophore = in-gap HETATM
        std = set(resids)
        r_lo = max((d2s[p] for p in (c1 - 1, c1 - 2, c1 - 3) if p in d2s), default=None)
        r_hi = min((d2s[p] for p in (c1 + 3, c1 + 4, c1 + 5) if p in d2s), default=None)
        het = a[~np.isin(a.res_name, _CANON3) & ~np.isin(a.res_name, list(_SOLVENT))]
        cand = []
        for rid in dict.fromkeys(het.res_id.tolist()):
            if rid in std:
                continue
            if r_lo is not None and r_hi is not None and not (r_lo < rid < r_hi):
                continue
            cand.append(rid)
        if not cand:                                    # fallback: any modelled tripeptide residue
            tri_resids = [d2s[p] for p in (c1, c1 + 1, c1 + 2) if p in d2s]
            if not tri_resids:
                raise ValueError(f"{name}[{pdb_id}]: could not locate chromophore atoms")
            ccoord = a[np.isin(a.res_id, tri_resids)].coord
            chromo_resids = set(tri_resids)
        else:
            ccoord = het[np.isin(het.res_id, cand)].coord
            chromo_resids = set(cand)

    # per standard-residue min distance to the chromophore
    prot = a[np.isin(a.res_name, _CANON3) & ~np.isin(a.res_id, list(chromo_resids))]
    dmin = {}
    for rid in dict.fromkeys(prot.res_id.tolist()):
        if rid not in s2d:
            continue
        rc = prot[prot.res_id == rid].coord
        dmin[rid] = float(np.sqrt(((rc[:, None, :] - ccoord[None, :, :]) ** 2).sum(-1)).min())

    # catalytic pair: nearest Arg + nearest Glu (by dataset letter) to the chromophore
    argc = [(d, s2d[rid]) for rid, d in dmin.items() if seq[s2d[rid]] == "R"]
    gluc = [(d, s2d[rid]) for rid, d in dmin.items() if seq[s2d[rid]] == "E"]
    catal = set()
    if argc:
        catal.add(min(argc)[1])
    if gluc:
        catal.add(min(gluc)[1])

    pocket = sorted({s2d[rid] for rid, d in dmin.items() if d <= cutoff} - tri0 - catal)

    # ---- Tier-B chromophore H-bond partners (capability proxy) --------------------
    # standard residues whose SIDE-CHAIN polar (N/O) atom lies within ``hbond_cutoff`` of a
    # chromophore polar (N/O) atom. Heavy-atom distance only: no explicit H, no angle test,
    # and no water-mediated bridges, so this over/under-calls at the margins by design. The
    # tripeptide and the catalytic pair are excluded (they are handled as fixed / pos2). A
    # subset of ``pocket`` since hbond_cutoff < cutoff and polar atoms are a subset of heavy.
    chromo_atoms = a[np.isin(a.res_id, list(chromo_resids))]
    cpolar = chromo_atoms[np.isin(chromo_atoms.element, ["N", "O"])].coord
    hbond = []
    if len(cpolar):
        _BB = ["N", "CA", "C", "O", "OXT"]                # backbone: unchanged by substitution
        for rid in dict.fromkeys(prot.res_id.tolist()):
            if rid not in s2d:
                continue
            res = prot[prot.res_id == rid]
            sc = res[np.isin(res.element, ["N", "O"]) & ~np.isin(res.atom_name, _BB)]
            if len(sc) == 0:                              # no side-chain donor/acceptor (nonpolar)
                continue
            d = float(np.sqrt(((sc.coord[:, None, :] - cpolar[None, :, :]) ** 2).sum(-1)).min())
            if d <= hbond_cutoff:
                dp = s2d[rid]
                if dp not in tri0 and dp not in catal:
                    hbond.append(dp)
    hbond = sorted(set(hbond))

    quality = dict(chain=chain_id, local_id=idv, coverage=cov)
    if return_hbond and return_quality:
        return c1, sorted(catal), pocket, quality, hbond
    if return_hbond:
        return c1, sorted(catal), pocket, hbond
    if return_quality:
        return c1, sorted(catal), pocket, quality
    return c1, sorted(catal), pocket


# ---- CLI: report the experimental 5 A pocket -------------------------------------
def _fmt(pos):
    return ",".join(str(p + 1) for p in pos)   # 1-based for human reading


def main():
    import argparse
    import csv

    ap = argparse.ArgumentParser(description="Report the experimental 5 A chromophore pocket.")
    ap.add_argument("--cur", default=os.path.join("..", "dataset_pipeline", "data", "peak", "curated"))
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

    print(f"experimental 5 A chromophore pocket (RCSB), cutoff={args.cutoff} A\n")
    for nm in args.names:
        seq = seqs[name2idx[nm]]
        exp = sorted(struct_pocket_experimental(nm, seq, cutoff=args.cutoff, structdir=STRUCTDIR))
        print(f"--- {nm} [{STRUCT[nm][0]}] (1-based dataset positions) ---")
        print(f"  pocket ({len(exp)}): {_fmt(exp)}")


if __name__ == "__main__":
    main()
