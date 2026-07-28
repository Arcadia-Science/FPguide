#!/usr/bin/env python
"""Score the campaign design windows against the family conservation table.

Every campaign scaffold is itself a member of the aligned dataset, so a window
position can be mapped onto its alignment column exactly -- no re-alignment, no
numbering assumptions -- and read off against the per-column statistics from
conservation.py. That makes the comparison scaffold-agnostic: EGFP's 1-based 68 and
avGFP's 67 land in the same column and get the same answer.

Reports, per scaffold and pooled:
  * whether the three hard-fixed positions (chromophore Gly + catalytic Arg/Glu)
    are in fact the family's invariants -- a check on the current rule;
  * which EDITABLE positions are chemistry-locked in the family (the risk list);
  * whether the Tier-B alphabets (aromatic at pos2, H-bond set at the polar
    contacts) match what the family actually uses at those columns;
  * whether the 5 A geometric criterion enriches for unconstrained positions
    relative to the rest of the barrel (it does not -- see README).

Writes results/design_window_comparison.csv (per position) and
results/design_window_summary.csv (per scaffold); prints the report.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import AlignIO

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WINDOW_FILES = {
    "conventional24": REPO / "design-campaign-conventional" / "design_windows_24_tierB.json",
    "EGFP": REPO / "design-campaign-EGFP" / "design_windows_egfp_tierB.json",
    "avGFP": REPO / "design-campaign-avGFP" / "design_windows_avgfp_tierB.json",
}
LOCKED = 0.90        # class frequency at which we call a column chemistry-locked
AROMATIC_SET = set("YWHF")
HBOND_SET = set("STYNQDEHKRW")


def load_alignment():
    aln = AlignIO.read(HERE / "data" / "fp_all.aln.fasta", "fasta")
    ids = [int(r.id.split("|")[0]) for r in aln]
    A = np.array([list(str(r.seq).upper()) for r in aln])
    meta = pd.read_csv(HERE / "data" / "fp_all_meta.csv").set_index("msa_id").loc[ids].reset_index()
    return A, meta


def family_alphabet(cons, A, meta, mass=0.90):
    """Smallest residue set covering `mass` of the weighted family distribution per position.

    This is the empirical alternative to a hand-written alphabet: instead of asserting
    "this contact must stay H-bond capable", it reports what the family actually puts
    there, ranked, truncated at 90% of the weighted mass.
    """
    aas = [c[len("f_aa_"):] for c in cons.columns if c.startswith("f_aa_")]
    rows = []
    seq2row = {s: i for i, s in enumerate(meta.seq)}
    wins = {}
    for campaign, path in WINDOW_FILES.items():
        wins[campaign] = json.load(open(path))["windows"]
    for campaign, w_all in wins.items():
        for name, w in w_all.items():
            seq = w["scaffold_seq"]
            if seq not in seq2row:
                continue
            col_of = np.nonzero(A[seq2row[seq]] != "-")[0]
            editable = set(w["editable_0based"])
            fixed = set(w["fixed_0based"])
            constraints = {int(k): set(v) for k, v in w.get("position_constraints", {}).items()}
            for p in sorted(editable | fixed):
                c = int(col_of[p])
                if c not in cons.index:
                    continue
                s = cons.loc[c]
                freqs = sorted(((a, float(s[f"f_aa_{a}"])) for a in aas),
                               key=lambda t: -t[1])
                keep, tot = [], 0.0
                for a, f in freqs:
                    if tot >= mass:
                        break
                    keep.append(a)
                    tot += f
                allowed = constraints.get(p)
                rows.append(dict(
                    campaign=campaign, scaffold=name, pos_1based=p + 1, scaffold_aa=seq[p],
                    role="fixed" if p in fixed else "editable", avgfp_pos=int(s.ref_pos),
                    current_constraint=("aromatic" if allowed == AROMATIC_SET else
                                        "hbond" if allowed == HBOND_SET else
                                        "fixed" if p in fixed else "none"),
                    n_aa_90=len(keep), family_alphabet_90="".join(keep),
                    covered_mass=round(tot, 3),
                    scaffold_aa_in_alphabet=seq[p] in keep))
    return pd.DataFrame(rows)


def main():
    A, meta = load_alignment()
    cons = pd.read_csv(HERE / "results" / "column_conservation.csv").set_index("aln_col")
    seq2row = {s: i for i, s in enumerate(meta.seq)}

    rows, missing = [], []
    for campaign, path in WINDOW_FILES.items():
        wins = json.load(open(path))["windows"]
        for name, w in wins.items():
            seq = w["scaffold_seq"]
            if seq not in seq2row:
                missing.append((campaign, name))
                continue
            r = seq2row[seq]
            # scaffold 0-based residue index -> alignment column
            col_of = np.nonzero(A[r] != "-")[0]
            editable = set(w["editable_0based"])
            fixed = set(w["fixed_0based"])
            constraints = {int(k): set(v) for k, v in w.get("position_constraints", {}).items()}
            for p in sorted(editable | fixed):
                c = int(col_of[p])
                if c not in cons.index:
                    continue
                s = cons.loc[c]
                allowed = constraints.get(p)
                rows.append(dict(
                    campaign=campaign, scaffold=name, pos_1based=p + 1,
                    scaffold_aa=seq[p], role="fixed" if p in fixed else "editable",
                    constraint=("aromatic" if allowed == AROMATIC_SET else
                                "hbond" if allowed == HBOND_SET else
                                "none" if allowed is None else "other"),
                    avgfp_pos=int(s.ref_pos), occupancy=float(s.occupancy),
                    top_aa=s.top_aa, top_aa_freq=float(s.top_aa_freq),
                    top_class=s.top_class, top_class_freq=float(s.top_class_freq),
                    C_id=float(s.C_id), C_chem=float(s.C_chem),
                    rsa=float(s.rsa) if pd.notna(s.rsa) else np.nan,
                    chromo_dist=float(s.chromo_dist) if pd.notna(s.chromo_dist) else np.nan,
                    # how much weighted family mass the imposed alphabet would keep
                    allowed_mass=(float(sum(s[f"f_aa_{a}"] for a in allowed)) if allowed else np.nan),
                    aromatic_mass=float(sum(s[f"f_aa_{a}"] for a in AROMATIC_SET)),
                    hbond_mass=float(sum(s[f"f_aa_{a}"] for a in HBOND_SET)),
                ))
    df = pd.DataFrame(rows)
    df["chem_locked"] = df.top_class_freq >= LOCKED
    df.to_csv(HERE / "results" / "design_window_comparison.csv", index=False)
    if missing:
        print(f"scaffolds not found verbatim in the alignment (skipped): {missing}\n")

    ed = df[df.role == "editable"]
    fx = df[df.role == "fixed"]
    hi = cons[cons.occupancy >= 0.90]
    base_rate = float((hi.top_class_freq >= LOCKED).mean())

    print("=" * 84)
    print("1. ARE THE HARD-FIXED POSITIONS THE FAMILY'S INVARIANTS?")
    print("=" * 84)
    g = (fx.groupby(["scaffold_aa", "top_aa", "top_class"])
           .agg(n_scaffolds=("scaffold", "size"), class_freq=("top_class_freq", "mean"),
                aa_freq=("top_aa_freq", "mean"), C_chem=("C_chem", "mean")).round(3))
    print(g.to_string())
    print(f"\n  fixed positions that are chemistry-locked in the family: "
          f"{int(fx.chem_locked.sum())}/{len(fx)}")
    bad = fx[~fx.chem_locked]
    if len(bad):
        print("  fixed but NOT family-locked:")
        print(bad[["scaffold", "pos_1based", "scaffold_aa", "top_aa", "top_class",
                   "top_class_freq"]].to_string(index=False))

    print("\n" + "=" * 84)
    print("2. EDITABLE POSITIONS THAT THE FAMILY HOLDS CHEMICALLY LOCKED")
    print("=" * 84)
    print(f"  base rate over the whole barrel: {base_rate:.1%} of columns are chemistry-locked")
    print(f"  rate inside the edit windows   : {ed.chem_locked.mean():.1%} "
          f"({int(ed.chem_locked.sum())}/{len(ed)} scaffold-positions)")
    risk = (ed[ed.chem_locked]
            .groupby(["avgfp_pos", "top_class", "top_aa"])
            .agg(n_scaffolds=("scaffold", "size"), class_freq=("top_class_freq", "mean"),
                 aa_freq=("top_aa_freq", "mean"), rsa=("rsa", "mean"),
                 constraint=("constraint", lambda s: ",".join(sorted(set(s)))))
            .reset_index().sort_values(["class_freq"], ascending=False).round(3))
    print("\n  pooled by avGFP-equivalent position:")
    print(risk.to_string(index=False))

    print("\n" + "=" * 84)
    print("3. DO THE TIER-B ALPHABETS MATCH WHAT THE FAMILY USES?")
    print("=" * 84)
    for lab in ("aromatic", "hbond"):
        sub = ed[ed.constraint == lab]
        if not len(sub):
            continue
        print(f"\n  {lab} constraint: {len(sub)} scaffold-positions, "
              f"mean family mass kept by the alphabet {sub.allowed_mass.mean():.3f}")
        t = (sub.groupby("avgfp_pos")
                .agg(n=("scaffold", "size"), kept=("allowed_mass", "mean"),
                     top_aa=("top_aa", "first"), top_aa_freq=("top_aa_freq", "mean"),
                     top_class=("top_class", "first"), top_class_freq=("top_class_freq", "mean"))
                .round(3).sort_values("kept"))
        print(t.to_string())

    print("\n" + "=" * 84)
    print("4. UNCONSTRAINED POSITIONS THE WINDOW COULD USE (5 A shell vs barrel)")
    print("=" * 84)
    inwin = set(ed[ed.avgfp_pos > 0].avgfp_pos)
    pocket = hi[(hi.chromo_dist <= 5) & (hi.ref_pos > 0)]
    print(f"  avGFP-equivalent positions touched by any window : {len(inwin)}")
    print(f"  mean C_chem inside windows {ed.C_chem.mean():.3f}   "
          f"vs whole barrel {hi.C_chem.mean():.3f}   "
          f"vs 5 A pocket {pocket.C_chem.mean():.3f}")
    free = hi[(hi.ref_pos > 0) & (hi.C_chem < 0.35) & (~hi.ref_pos.isin(inwin))
              & (hi.chromo_dist <= 10)].sort_values("C_chem")
    print(f"\n  low-constraint columns within 10 A that no window currently edits "
          f"({len(free)}):")
    print(free[["ref_pos", "ref_aa", "top_aa", "top_aa_freq", "top_class", "top_class_freq",
                "C_chem", "rsa", "chromo_dist"]].round(3).to_string(index=False))

    print("\n" + "=" * 84)
    print("6. EVOLUTION-DERIVED ALPHABET PER WINDOW POSITION (90% of family mass)")
    print("=" * 84)
    alpha = family_alphabet(cons, A, meta)
    alpha.to_csv(HERE / "results" / "window_family_alphabet.csv", index=False)
    show = alpha[(alpha.campaign == "EGFP") & (alpha.role == "editable")]
    print(show[["pos_1based", "scaffold_aa", "avgfp_pos", "current_constraint", "n_aa_90",
                "family_alphabet_90", "scaffold_aa_in_alphabet"]].to_string(index=False))
    print(f"\n  window positions whose own scaffold residue is NOT in the family's 90% set: "
          f"{int((~alpha.scaffold_aa_in_alphabet).sum())}/{len(alpha)}")

    summ = (ed.groupby(["campaign", "scaffold"])
              .agg(n_editable=("pos_1based", "size"), n_chem_locked=("chem_locked", "sum"),
                   mean_C_chem=("C_chem", "mean"), mean_C_id=("C_id", "mean"))
              .reset_index().round(3))
    summ["frac_locked"] = (summ.n_chem_locked / summ.n_editable).round(3)
    summ.to_csv(HERE / "results" / "design_window_summary.csv", index=False)
    print("\n" + "=" * 84)
    print("5. PER-SCAFFOLD SUMMARY")
    print("=" * 84)
    print(summ.sort_values("frac_locked", ascending=False).to_string(index=False))
    print("\n-> results/design_window_comparison.csv, results/design_window_summary.csv")


if __name__ == "__main__":
    main()
