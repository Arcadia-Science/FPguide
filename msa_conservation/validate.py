#!/usr/bin/env python
"""Robustness checks for the conservation analysis.

Four things could make the headline result an artifact, and each gets a test:

1. **Bad alignment.** If the barrel were misaligned, "conservation" would be noise.
   Test: the chromophore X-[YWHF]-G tripeptide must land in the same three columns
   for every sequence, independently of anything the analysis computes.
2. **Redundancy.** The set is dominated by avGFP and DsRed mutant series. Test:
   recompute on one representative per 90%-identity cluster (117 sequences,
   unweighted) and compare to the Henikoff-weighted full-set numbers.
3. **One clade carrying the signal.** Aequorea-lineage variants are the single
   largest block. Test: recompute separately on Aequorea-lineage and non-Aequorea
   sequences and require the chemistry-locked positions to hold in both halves.
4. **Threshold sensitivity.** Test: recount the identity-vs-chemistry gap at
   several occupancy and frequency cutoffs.

Run after conservation.py; prints a report and writes results/validation.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import AlignIO

from conservation import (AAS, CLASS_NAMES, CLASS_VEC, OCC_MIN, class_freqs, encode,
                          entropy, henikoff_weights, identity_matrix, n_eff,
                          weighted_freqs)

HERE = Path(__file__).resolve().parent


def core_stats(code, w, bg, bg_class):
    F = weighted_freqs(code, w)
    Fc = class_freqs(F)
    return (F, Fc,
            1 - entropy(F) / entropy(bg),
            1 - entropy(Fc) / entropy(bg_class))


def main():
    aln = AlignIO.read(HERE / "data" / "fp_all.aln.fasta", "fasta")
    A = np.array([list(str(r.seq).upper()) for r in aln])
    ids = [int(r.id.split("|")[0]) for r in aln]
    meta = pd.read_csv(HERE / "data" / "fp_all_meta.csv").set_index("msa_id").loc[ids].reset_index()
    code_full = encode(A)
    occ_full = (code_full >= 0).mean(0)
    core = np.nonzero(occ_full >= OCC_MIN)[0]
    code = code_full[:, core]
    df = pd.read_csv(HERE / "results" / "column_conservation.csv")
    out = {}

    # ---- 1. alignment integrity: chromophore tripeptide in one set of columns ----
    ref = meta.index[meta.slug == "avgfp"][0]
    refpos = np.cumsum(A[ref] != "-")
    c66, c67 = (int(np.nonzero(refpos == p)[0][0]) for p in (66, 67))
    motif = np.isin(A[:, c66], list("YWHF")) & (A[:, c67] == "G")
    gly_only = (A[:, c67] == "G")
    out["aln_chromophore_motif_frac"] = float(motif.mean())
    out["aln_gly67_frac"] = float(gly_only.mean())
    out["aln_motif_exceptions"] = meta.loc[~motif, "name"].tolist()
    print(f"[1] alignment integrity: chromophore [YWHF]-G in the reference columns for "
          f"{motif.sum()}/{len(A)} = {motif.mean():.1%}; Gly at the G67 column {gly_only.mean():.1%}")
    print(f"    exceptions: {out['aln_motif_exceptions']}")

    # background from the full weighted set (same definition as conservation.py)
    w_full = henikoff_weights(code)
    F_full = weighted_freqs(code, w_full)
    bg = (F_full * (code >= 0).sum(0)[:, None]).sum(0)
    bg /= bg.sum()
    bg_class = np.array([bg[CLASS_VEC == k].sum() for k in range(len(CLASS_NAMES))])

    # ---- 2. redundancy: one representative per 90% identity cluster ----
    ident = identity_matrix(code)
    n = len(code)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.nonzero(np.triu(ident >= 0.90, 1))):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    labels = np.array([find(i) for i in range(n)])
    reps = pd.Series(range(n)).groupby(labels).first().values   # first member of each cluster
    code_rep = code[reps]
    _, _, C_id_rep, C_chem_rep = core_stats(code_rep, np.ones(len(reps)), bg, bg_class)
    ok = df.occupancy >= 0.9
    out["n_clusters_90"] = int(len(reps))
    out["corr_C_chem_nonredundant"] = float(np.corrcoef(df.C_chem[ok], C_chem_rep[ok.values])[0, 1])
    out["corr_C_id_nonredundant"] = float(np.corrcoef(df.C_id[ok], C_id_rep[ok.values])[0, 1])
    print(f"[2] redundancy: {len(reps)} cluster representatives (unweighted) vs weighted full set -> "
          f"corr(C_chem) {out['corr_C_chem_nonredundant']:.3f}, corr(C_id) {out['corr_C_id_nonredundant']:.3f}")

    # ---- 3. clade split: Aequorea lineage vs everything else ----
    is_aeq = meta.parent_organism.fillna("").str.startswith("Aequorea").values
    halves = {}
    for lab, mask in [("aequorea", is_aeq), ("non_aequorea", ~is_aeq)]:
        sub = code[mask]
        w = henikoff_weights(sub)
        F, Fc, C_id, C_chem = core_stats(sub, w, bg, bg_class)
        halves[lab] = dict(n=int(mask.sum()), n_eff=n_eff(w), Fc=Fc, C_chem=C_chem,
                           top_class=[CLASS_NAMES[i] for i in Fc.argmax(1)],
                           top_class_freq=Fc.max(1))
        print(f"[3] {lab}: n={mask.sum()}, N_eff={n_eff(w):.1f}")
    out["clade_corr_C_chem"] = float(np.corrcoef(halves["aequorea"]["C_chem"][ok.values],
                                                 halves["non_aequorea"]["C_chem"][ok.values])[0, 1])

    locked = df[ok & (df.top_class_freq >= 0.90)]
    rows = []
    for r in locked.itertuples():
        i = int(np.nonzero(core == r.aln_col)[0][0])
        rec = dict(ref_pos=int(r.ref_pos), ref_aa=r.ref_aa, cls=r.top_class,
                   full=float(r.top_class_freq))
        for lab in halves:
            k = CLASS_NAMES.index(r.top_class)
            rec[lab] = float(halves[lab]["Fc"][i, k])
        rec["holds_both"] = bool(rec["aequorea"] >= 0.8 and rec["non_aequorea"] >= 0.8)
        rows.append(rec)
    rep = pd.DataFrame(rows).sort_values(["cls", "ref_pos"])
    out["clade_locked_holds_both"] = int(rep.holds_both.sum())
    out["clade_locked_total"] = int(len(rep))
    out["clade_locked_table"] = rep.to_dict("records")
    print(f"    corr(C_chem) between clades: {out['clade_corr_C_chem']:.3f}")
    print(f"    chemistry-locked positions holding at >=80% in BOTH clades: "
          f"{rep.holds_both.sum()}/{len(rep)}")
    print(rep[["ref_pos", "ref_aa", "cls", "full", "aequorea", "non_aequorea", "holds_both"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- 4. threshold sensitivity ----
    sens = []
    for occ_cut in (0.5, 0.7, 0.9, 0.95):
        d = df[df.occupancy >= occ_cut]
        for t in (0.9, 0.8, 0.7):
            sens.append(dict(occupancy=occ_cut, threshold=t, n_cols=len(d),
                             n_identity=int((d.top_aa_freq >= t).sum()),
                             n_chemistry=int((d.top_class_freq >= t).sum())))
    out["threshold_sensitivity"] = sens
    s = pd.DataFrame(sens)
    s["ratio"] = s.n_chemistry / s.n_identity
    print("[4] threshold sensitivity (chemistry / identity position counts)")
    print(s.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    with open(HERE / "results" / "validation.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("-> results/validation.json")


if __name__ == "__main__":
    main()
