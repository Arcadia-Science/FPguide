#!/usr/bin/env python
"""Score the EGFP campaign design window against the family conservation table.

The EGFP scaffold is itself a member of the aligned dataset, so each window position maps
onto its alignment column exactly -- no re-alignment, no numbering assumptions. Every number
below is therefore read off the same per-column statistics `conservation.py` computed, with
avGFP-equivalent positions reported alongside EGFP's own 1-based numbering.

SCOPE. This covers the EGFP window only -- the one campaign whose design window is tracked in
this repo (`design-campaign-EGFP/design_windows_egfp_tierB.json`). Earlier versions of this
script also pooled the 24 conventional scaffolds and the avGFP campaign, but both of those
campaigns were archived out of the repo (see the top-level README), and their window JSONs are
not published, so those rows could not be regenerated from a clone. Rather than report numbers
nobody can reproduce, the analysis now stops at the window it can defend. Where the pooled
version reported a mean over 26 scaffolds, this reports the per-position value for one.

Reports:
  * whether the three hard-fixed positions (chromophore Gly + catalytic Arg/Glu)
    are in fact the family's invariants -- a check on the current rule;
  * which EDITABLE positions are chemistry-locked in the family (the risk list);
  * whether the Tier-B alphabets (aromatic at pos2, H-bond set at the polar
    contacts) match what the family actually uses at those columns;
  * whether the 5 A geometric criterion enriches for unconstrained positions
    relative to the rest of the barrel (it does not -- see README).

Writes results/window_vs_family_egfp.csv (per position) and
results/window_alphabet_egfp.csv (per position); prints the report.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import AlignIO

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WINDOW_FILE = REPO / "design-campaign-EGFP" / "design_windows_egfp_tierB.json"
WINDOW_NAME = "EGFP"     # the key under "windows" in that JSON
LOCKED = 0.90            # class frequency at which we call a column chemistry-locked
AROMATIC_SET = set("YWHF")
HBOND_SET = set("STYNQDEHKRW")


def load_alignment():
    aln = AlignIO.read(HERE / "data" / "fp_all.aln.fasta", "fasta")
    ids = [int(r.id.split("|")[0]) for r in aln]
    A = np.array([list(str(r.seq).upper()) for r in aln])
    meta = pd.read_csv(HERE / "data" / "fp_all_meta.csv").set_index("msa_id").loc[ids].reset_index()
    return A, meta


def load_window():
    """The EGFP Tier-B design window, with a loud explanation if it is not on disk."""
    if not WINDOW_FILE.exists():
        raise SystemExit(
            f"{Path(__file__).name}: missing {WINDOW_FILE}\n\n"
            "This is the EGFP campaign's design window, tracked in git. If it is absent the\n"
            "clone is incomplete -- it is not a generated artifact and there is nothing to\n"
            "rebuild it from in this folder. Regenerate it from the campaign scaffold with:\n\n"
            "    python ../fpdesign/build_design_windows.py \\\n"
            "        --pairs <scaffold pairs csv> --out design-campaign-EGFP/design_windows_egfp_tierB.json\n")
    return json.load(open(WINDOW_FILE))["windows"][WINDOW_NAME]


def window_columns(w, A, meta):
    """(scaffold_seq, {0-based scaffold position -> alignment column}) for the window's scaffold."""
    seq = w["scaffold_seq"]
    seq2row = {s: i for i, s in enumerate(meta.seq)}
    if seq not in seq2row:
        raise SystemExit(
            f"{Path(__file__).name}: the {WINDOW_NAME} scaffold sequence is not in the alignment\n"
            "verbatim, so its positions cannot be mapped to alignment columns. The window JSON and\n"
            "data/fp_all.aln.fasta are out of sync -- rebuild the alignment (./run_msa.sh) or check\n"
            "that the window's scaffold_seq matches the curated dataset entry.\n")
    col_of = np.nonzero(A[seq2row[seq]] != "-")[0]
    return seq, col_of


def family_alphabet(cons, w, seq, col_of, mass=0.90):
    """Smallest residue set covering `mass` of the weighted family distribution per position.

    This is the empirical alternative to a hand-written alphabet: instead of asserting
    "this contact must stay H-bond capable", it reports what the family actually puts
    there, ranked, truncated at 90% of the weighted mass.
    """
    aas = [c[len("f_aa_"):] for c in cons.columns if c.startswith("f_aa_")]
    editable = set(w["editable_0based"])
    fixed = set(w["fixed_0based"])
    constraints = {int(k): set(v) for k, v in w.get("position_constraints", {}).items()}
    rows = []
    for p in sorted(editable | fixed):
        c = int(col_of[p])
        if c not in cons.index:
            continue
        s = cons.loc[c]
        freqs = sorted(((a, float(s[f"f_aa_{a}"])) for a in aas), key=lambda t: -t[1])
        keep, tot = [], 0.0
        for a, f in freqs:
            if tot >= mass:
                break
            keep.append(a)
            tot += f
        allowed = constraints.get(p)
        rows.append(dict(
            pos_1based=p + 1, scaffold_aa=seq[p],
            role="fixed" if p in fixed else "editable", avgfp_pos=int(s.ref_pos),
            current_constraint=("aromatic" if allowed == AROMATIC_SET else
                                "hbond" if allowed == HBOND_SET else
                                "fixed" if p in fixed else "none"),
            n_aa_90=len(keep), family_alphabet_90="".join(keep),
            covered_mass=round(tot, 3),
            scaffold_aa_in_alphabet=seq[p] in keep))
    return pd.DataFrame(rows)


def score_window(cons, w, seq, col_of):
    """One row per window position, scored against the family column statistics."""
    editable = set(w["editable_0based"])
    fixed = set(w["fixed_0based"])
    constraints = {int(k): set(v) for k, v in w.get("position_constraints", {}).items()}
    rows = []
    for p in sorted(editable | fixed):
        c = int(col_of[p])
        if c not in cons.index:
            continue
        s = cons.loc[c]
        allowed = constraints.get(p)
        rows.append(dict(
            pos_1based=p + 1, scaffold_aa=seq[p],
            role="fixed" if p in fixed else "editable",
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
            # sorted(): summing over a set iterates in an order that depends on Python's
            # per-process string hash seed, which made the last bit of these columns vary
            # between runs. Fixing the order makes the CSV byte-reproducible.
            allowed_mass=(float(sum(s[f"f_aa_{a}"] for a in sorted(allowed))) if allowed else np.nan),
            aromatic_mass=float(sum(s[f"f_aa_{a}"] for a in sorted(AROMATIC_SET))),
            hbond_mass=float(sum(s[f"f_aa_{a}"] for a in sorted(HBOND_SET))),
        ))
    df = pd.DataFrame(rows)
    df["chem_locked"] = df.top_class_freq >= LOCKED
    return df


def main():
    A, meta = load_alignment()
    cons = pd.read_csv(HERE / "results" / "column_conservation.csv").set_index("aln_col")
    w = load_window()
    seq, col_of = window_columns(w, A, meta)

    df = score_window(cons, w, seq, col_of)
    df.to_csv(HERE / "results" / "window_vs_family_egfp.csv", index=False)

    ed = df[df.role == "editable"]
    fx = df[df.role == "fixed"]
    hi = cons[cons.occupancy >= 0.90]
    base_rate = float((hi.top_class_freq >= LOCKED).mean())

    print("=" * 84)
    print(f"{WINDOW_NAME} design window vs the FP family: {len(df)} positions "
          f"({len(ed)} editable, {len(fx)} hard-fixed)")
    print("=" * 84)

    print("\n" + "=" * 84)
    print("1. ARE THE HARD-FIXED POSITIONS THE FAMILY'S INVARIANTS?")
    print("=" * 84)
    print(fx[["pos_1based", "scaffold_aa", "avgfp_pos", "top_aa", "top_aa_freq",
              "top_class", "top_class_freq", "C_chem", "chem_locked"]]
          .round(3).to_string(index=False))
    print(f"\n  fixed positions that are chemistry-locked in the family: "
          f"{int(fx.chem_locked.sum())}/{len(fx)}")
    bad = fx[~fx.chem_locked]
    if len(bad):
        print("  fixed but NOT family-locked:")
        print(bad[["pos_1based", "scaffold_aa", "top_aa", "top_class",
                   "top_class_freq"]].round(3).to_string(index=False))

    print("\n" + "=" * 84)
    print("2. EDITABLE POSITIONS THAT THE FAMILY HOLDS CHEMICALLY LOCKED")
    print("=" * 84)
    print(f"  base rate over the whole barrel: {base_rate:.1%} of columns are chemistry-locked")
    print(f"  rate inside the edit window    : {ed.chem_locked.mean():.1%} "
          f"({int(ed.chem_locked.sum())}/{len(ed)} positions)")
    risk = (ed[ed.chem_locked]
            .sort_values("top_class_freq", ascending=False)
            [["pos_1based", "scaffold_aa", "avgfp_pos", "top_class", "top_class_freq",
              "top_aa", "top_aa_freq", "rsa", "constraint"]].round(3))
    print("\n  the risk list, by how hard the family locks the position:")
    print(risk.to_string(index=False))

    print("\n" + "=" * 84)
    print("3. DO THE TIER-B ALPHABETS MATCH WHAT THE FAMILY USES?")
    print("=" * 84)
    for lab in ("aromatic", "hbond"):
        sub = ed[ed.constraint == lab]
        if not len(sub):
            continue
        print(f"\n  {lab} constraint: {len(sub)} position(s), "
              f"mean family mass kept by the alphabet {sub.allowed_mass.mean():.3f}")
        print(sub.sort_values("allowed_mass")
                 [["pos_1based", "scaffold_aa", "avgfp_pos", "allowed_mass", "top_aa",
                   "top_aa_freq", "top_class", "top_class_freq"]].round(3).to_string(index=False))

    print("\n" + "=" * 84)
    print("4. UNCONSTRAINED POSITIONS THE WINDOW COULD USE (5 A shell vs barrel)")
    print("=" * 84)
    inwin = set(ed[ed.avgfp_pos > 0].avgfp_pos)
    pocket = hi[(hi.chromo_dist <= 5) & (hi.ref_pos > 0)]
    print(f"  avGFP-equivalent positions touched by the window : {len(inwin)}")
    print(f"  mean C_chem inside the window {ed.C_chem.mean():.3f}   "
          f"vs whole barrel {hi.C_chem.mean():.3f}   "
          f"vs 5 A pocket {pocket.C_chem.mean():.3f}")
    free = hi[(hi.ref_pos > 0) & (hi.C_chem < 0.35) & (~hi.ref_pos.isin(inwin))
              & (hi.chromo_dist <= 10)].sort_values("C_chem")
    print(f"\n  low-constraint columns within 10 A that the window does not edit "
          f"({len(free)}):")
    print(free[["ref_pos", "ref_aa", "top_aa", "top_aa_freq", "top_class", "top_class_freq",
                "C_chem", "rsa", "chromo_dist"]].round(3).to_string(index=False))

    print("\n" + "=" * 84)
    print("5. WINDOW TOTALS")
    print("=" * 84)
    print(f"  editable positions        : {len(ed)}")
    print(f"  chemistry-locked of those : {int(ed.chem_locked.sum())} "
          f"({ed.chem_locked.mean():.1%})")
    print(f"  mean C_chem / C_id        : {ed.C_chem.mean():.3f} / {ed.C_id.mean():.3f}")

    print("\n" + "=" * 84)
    print("6. EVOLUTION-DERIVED ALPHABET PER WINDOW POSITION (90% of family mass)")
    print("=" * 84)
    alpha = family_alphabet(cons, w, seq, col_of)
    alpha.to_csv(HERE / "results" / "window_alphabet_egfp.csv", index=False)
    print(alpha[alpha.role == "editable"]
          [["pos_1based", "scaffold_aa", "avgfp_pos", "current_constraint", "n_aa_90",
            "family_alphabet_90", "scaffold_aa_in_alphabet"]].to_string(index=False))
    print(f"\n  window positions whose own scaffold residue is NOT in the family's 90% set: "
          f"{int((~alpha.scaffold_aa_in_alphabet).sum())}/{len(alpha)}")

    print("\n-> results/window_vs_family_egfp.csv, results/window_alphabet_egfp.csv")


if __name__ == "__main__":
    main()
