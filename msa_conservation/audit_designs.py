#!/usr/bin/env python
"""What did the EGFP campaigns actually put at each window position, and does the family
ever do the same thing?

The window comparison says which positions are *risky*; this says which risks were
actually taken. Every EGFP design CSV is diffed against its scaffold and each observed
substitution is scored against the family column it lands in:

  * ``fam_freq``      weighted frequency of the introduced residue in the family
  * ``never_seen``    the residue occurs in no aligned FP at that column
  * ``off_alphabet``  outside the residues covering 90% of family mass
  * ``class_break``   changes the side-chain class away from the family's dominant one,
                      at a column where that class holds >= 80% of the mass

Writes results/design_audit_positions.csv (per window position) and
results/design_audit_substitutions.csv (per distinct substitution).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / "design-campaign-EGFP"
AAS = "ACDEFGHIKLMNPQRSTVWY"
CLASSES = {"aliphatic": "AVLIM", "aromatic": "FWY", "polar": "STNQ",
           "acidic": "DE", "basic": "KRH", "glycine": "G", "proline": "P", "cysteine": "C"}
CLASS_OF = {a: c for c, aas in CLASSES.items() for a in aas}
DOMINANT = 0.80      # class-mass level at which a class change counts as a break


def load_designs():
    frames = []
    for p in sorted(CAMPAIGN.rglob("*.csv")):
        if "archive" in p.parts or "shortlists" in p.parts:
            continue
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if not {"designed_seq", "scaffold_seq"}.issubset(d.columns):
            continue
        d = d[d.designed_seq.notna() & d.scaffold_seq.notna()].copy()
        if not len(d):
            continue
        rel = p.relative_to(CAMPAIGN)
        d["campaign"] = rel.parts[0]
        d["source"] = str(rel)
        frames.append(d[["campaign", "source", "designed_seq", "scaffold_seq"]
                        + [c for c in ("target_name", "trial", "round") if c in d.columns]])
    return pd.concat(frames, ignore_index=True)


def main():
    d = load_designs()
    print(f"loaded {len(d)} design rows from {d.source.nunique()} files "
          f"across {d.campaign.nunique()} campaigns")
    print(d.campaign.value_counts().to_string())

    cons = pd.read_csv(HERE / "results" / "column_conservation.csv")
    cmp_ = pd.read_csv(HERE / "results" / "design_window_comparison.csv")
    cmp_ = cmp_[cmp_.campaign == "EGFP"]
    alpha = pd.read_csv(HERE / "results" / "window_family_alphabet.csv")
    alpha = alpha[alpha.campaign == "EGFP"].set_index("pos_1based")

    # EGFP 1-based position -> family row
    fam = {}
    for r in cmp_.itertuples():
        row = cons[cons.ref_pos == r.avgfp_pos].iloc[0]
        fam[r.pos_1based] = row

    subs = {}
    for r in d.itertuples():
        s, q = r.scaffold_seq, r.designed_seq
        if len(s) != len(q):
            continue
        for i, (a, b) in enumerate(zip(s, q)):
            if a != b:
                subs.setdefault((i + 1, a, b), []).append(r.campaign)

    rows = []
    for (pos, wt, mut), camps in subs.items():
        f = fam.get(pos)
        if f is None or mut not in AAS:
            continue
        freq = float(f[f"f_aa_{mut}"])
        alph = str(alpha.loc[pos, "family_alphabet_90"]) if pos in alpha.index else ""
        rows.append(dict(
            pos_1based=pos, wt=wt, mut=mut, n_designs=len(camps),
            campaigns=",".join(sorted(set(camps))),
            fam_freq=round(freq, 4), never_seen=freq == 0.0,
            off_alphabet=mut not in alph, family_alphabet_90=alph,
            fam_top_class=f.top_class, fam_top_class_freq=round(float(f.top_class_freq), 3),
            mut_class=CLASS_OF[mut],
            class_break=bool(CLASS_OF[mut] != f.top_class and f.top_class_freq >= DOMINANT),
            C_chem=round(float(f.C_chem), 3),
            rsa=round(float(f.rsa), 3) if pd.notna(f.rsa) else np.nan,
        ))
    S = pd.DataFrame(rows).sort_values(["class_break", "never_seen", "n_designs"],
                                       ascending=[False, False, False])
    S.to_csv(HERE / "results" / "design_audit_substitutions.csv", index=False)

    pos = (S.groupby("pos_1based")
             .apply(lambda g: pd.Series({
                 "wt": g.wt.iloc[0],
                 "n_designs_mutated": int(g.n_designs.sum()),
                 "n_distinct_subs": len(g),
                 "n_class_break": int((g.class_break * g.n_designs).sum()),
                 "n_never_seen": int((g.never_seen * g.n_designs).sum()),
                 "n_off_alphabet": int((g.off_alphabet * g.n_designs).sum()),
                 "mean_fam_freq": round(float((g.fam_freq * g.n_designs).sum()
                                              / g.n_designs.sum()), 4),
                 "C_chem": g.C_chem.iloc[0],
                 "fam_top_class": g.fam_top_class.iloc[0],
                 "fam_top_class_freq": g.fam_top_class_freq.iloc[0],
                 "family_alphabet_90": g.family_alphabet_90.iloc[0],
                 "subs": ",".join(f"{r.mut}x{r.n_designs}" for r in
                                  g.sort_values("n_designs", ascending=False).itertuples()),
             }), include_groups=False)
             .reset_index())
    total = pos.n_designs_mutated.sum()
    pos["frac_class_break"] = (pos.n_class_break / pos.n_designs_mutated).round(3)
    pos["frac_off_alphabet"] = (pos.n_off_alphabet / pos.n_designs_mutated).round(3)
    pos = pos.sort_values("n_class_break", ascending=False)
    pos.to_csv(HERE / "results" / "design_audit_positions.csv", index=False)

    pd.set_option("display.width", 260)
    print(f"\n{len(S)} distinct substitutions over {total} position-edits, "
          f"at {len(pos)} of 25 editable positions")
    print(f"  edits introducing a residue NEVER seen in the family at that column: "
          f"{int(pos.n_never_seen.sum())} ({pos.n_never_seen.sum()/total:.1%})")
    print(f"  edits outside the family's 90% alphabet: {int(pos.n_off_alphabet.sum())} "
          f"({pos.n_off_alphabet.sum()/total:.1%})")
    print(f"  edits breaking a >={DOMINANT:.0%} dominant class: {int(pos.n_class_break.sum())} "
          f"({pos.n_class_break.sum()/total:.1%})")

    print("\n=== per position (sorted by class-breaking edits) ===")
    print(pos[["pos_1based", "wt", "C_chem", "fam_top_class", "fam_top_class_freq",
               "family_alphabet_90", "n_designs_mutated", "n_class_break",
               "frac_off_alphabet", "subs"]].to_string(index=False))

    print("\n=== the class-breaking substitutions themselves ===")
    cb = S[S.class_break]
    print(cb[["pos_1based", "wt", "mut", "mut_class", "fam_top_class", "fam_top_class_freq",
              "fam_freq", "n_designs", "campaigns"]].to_string(index=False))
    print("\n-> results/design_audit_positions.csv, results/design_audit_substitutions.csv")


if __name__ == "__main__":
    main()
