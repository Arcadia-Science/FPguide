#!/usr/bin/env python
"""Print the findings quoted in README.md, so every number there is reproducible.

Run after conservation.py. Writes results/findings.txt alongside stdout.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OCC = 0.90
PROPS = ["hydropathy", "polarity", "charge", "aromatic", "hbond", "volume"]


def strand_runs(positions, min_len=6):
    runs, cur = [], [positions[0]]
    for p in positions[1:]:
        if p == cur[-1] + 1:
            cur.append(p)
        else:
            runs.append(cur)
            cur = [p]
    runs.append(cur)
    return [r for r in runs if len(r) >= min_len]


def main():
    buf = io.StringIO()

    def out(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        buf.write(s + "\n")

    df = pd.read_csv(HERE / "results" / "column_conservation.csv")
    hi = df[df.occupancy >= OCC].copy()
    out(f"core columns at occupancy>={OCC}: {len(hi)} of {len(df)}\n")

    out("=" * 78)
    out("A. HOW MUCH IS CONSERVED: identity vs chemistry")
    out("=" * 78)
    for t in (0.95, 0.90, 0.80, 0.70):
        out(f"  >={t:.0%}: same amino acid {int((hi.top_aa_freq>=t).sum()):3d} positions | "
            f"same chemistry class {int((hi.top_class_freq>=t).sum()):3d} positions | "
            f"ratio {(hi.top_class_freq>=t).sum()/(hi.top_aa_freq>=t).sum():.2f}")

    out("\n" + "=" * 78)
    out("B. THE CHEMISTRY-LOCKED SET (class frequency >= 90%), by class")
    out("=" * 78)
    cl = hi[hi.top_class_freq >= 0.90]
    for cls, g in cl.groupby("top_class"):
        pos = ", ".join(f"{r.ref_aa}{int(r.ref_pos)}" for r in g.sort_values("ref_pos").itertuples())
        out(f"  {cls:10s} n={len(g):2d}  mean identity freq {g.top_aa_freq.mean():.2f}   {pos}")
    out(f"\n  buried (RSA<0.20): {int((cl.rsa < 0.20).sum())}/{len(cl)};  "
        f"median RSA {cl.rsa.median():.3f}")
    out(f"  in beta strand: {int((cl.sse=='b').sum())}, in loop/coil: {int((cl.sse=='c').sum())} "
        f"(all core: {int((hi.sse=='b').sum())} strand / {int((hi.sse=='c').sum())} coil)")

    out("\n  positions where chemistry is locked but identity is genuinely free "
        "(class>=90%, top aa<=60%):")
    free = cl[cl.top_aa_freq <= 0.60].sort_values("chem_minus_id", ascending=False)
    for r in free.itertuples():
        out(f"    {r.ref_aa}{int(r.ref_pos):<4d} {r.top_class:10s} class {r.top_class_freq:.2f}  "
            f"identity {r.top_aa_freq:.2f}  RSA {r.rsa:.2f}   {r.aa_composition}")

    out("\n" + "=" * 78)
    out("C. WHAT IS INVARIANT BY IDENTITY (top aa >= 90%)")
    out("=" * 78)
    inv = hi[hi.top_aa_freq >= 0.90].sort_values("top_aa_freq", ascending=False)
    for r in inv.itertuples():
        out(f"  {r.ref_aa}{int(r.ref_pos):<4d} {r.top_aa} {r.top_aa_freq:.3f}  "
            f"RSA {r.rsa:.2f}  chromophore {r.chromo_dist:5.1f} A  sse {r.sse}")

    out("\n" + "=" * 78)
    out("D. CONSERVED CHEMISTRY TRACKS BURIAL, NOT THE CHROMOPHORE")
    out("=" * 78)
    for col, lab in [("hydropathy_mean", "mean hydropathy"), ("polarity_mean", "mean polarity"),
                     ("hbond_mean", "mean H-bond capacity"), ("volume_mean", "mean volume"),
                     ("charge_rho", "charge constraint rho"), ("C_chem", "chemistry conservation"),
                     ("C_id", "identity conservation")]:
        v = hi[["rsa", col]].dropna()
        r, p = spearmanr(v.rsa, v[col])
        out(f"  {lab:24s} vs RSA              : rho {r:+.3f}  p {p:.1e}")
    v = hi[["chromo_dist", "C_chem"]].dropna()
    r, p = spearmanr(v.chromo_dist, v.C_chem)
    out(f"  {'chemistry conservation':24s} vs chromophore dist.: rho {r:+.3f}  p {p:.2f}   <- no gradient")
    v = hi[["chromo_dist", "C_id"]].dropna()
    r, p = spearmanr(v.chromo_dist, v.C_id)
    out(f"  {'identity conservation':24s} vs chromophore dist.: rho {r:+.3f}  p {p:.2f}")

    zone = np.where(hi.chromo_dist <= 5, "pocket<=5A",
                    np.where(hi.chromo_dist <= 10, "shell5-10A", "outer>10A"))
    out("\n  by zone (mean):")
    out(hi.assign(zone=zone).groupby("zone")[["C_id", "C_chem", "hydropathy_mean",
                                              "hbond_mean", "charge_rho"]].mean().round(3).to_string())
    bur = np.where(hi.rsa < 0.20, "buried<0.20", np.where(hi.rsa < 0.50, "partial", "exposed>=0.50"))
    out("\n  by burial (mean):")
    out(hi.assign(bur=bur).groupby("bur")[["C_id", "C_chem", "hydropathy_mean", "polarity_mean",
                                           "charge_rho", "volume_mean"]].mean().round(3).to_string())

    out("\n" + "=" * 78)
    out("E. BURIED CHARGE IS EXCLUDED EXCEPT AT THE CATALYTIC DYAD")
    out("=" * 78)
    b = hi[hi.rsa < 0.20]
    out(f"  buried positions: {len(b)}")
    for t in (0.90, 0.80, 0.70):
        m = b[(b.top_class.isin(["acidic", "basic"])) & (b.top_class_freq >= t)]
        out(f"    dominant class acidic/basic at >={t:.0%}: {len(m)}  "
            f"-> {[f'{r.ref_aa}{int(r.ref_pos)}({r.top_class[:4]},{r.top_class_freq:.2f})' for r in m.itertuples()]}")
    out(f"  buried positions that are charge-free with high constraint "
        f"(|mean charge|<0.1, rho>=0.7): {int(((b.charge_rho>=0.7)&(b.charge_mean.abs()<0.1)).sum())}/{len(b)}")
    e = hi[hi.rsa >= 0.50]
    out(f"  exposed positions (n={len(e)}): mean charge constraint rho {e.charge_rho.mean():+.3f} "
        f"(negative => more charge-variable than the family background)")

    out("\n" + "=" * 78)
    out("F. BETA-STRAND PERIODICITY (inside/outside alternation)")
    out("=" * 78)
    d = hi[hi.ref_pos > 0].sort_values("ref_pos").set_index("ref_pos")
    runs = strand_runs(sorted(hi.loc[hi.sse == "b", "ref_pos"].astype(int)))
    out(f"  contiguous beta-strand runs >=6 residues: {len(runs)}  lengths {[len(r) for r in runs]}")
    for col, lab in [("rsa", "RSA (structure)"), ("hydropathy_mean", "mean hydropathy"),
                     ("polarity_mean", "mean polarity")]:
        s = d[col]
        line = []
        for lag in (1, 2, 3, 4):
            vals = [np.corrcoef(s.loc[r].values[:-lag], s.loc[r].values[lag:])[0, 1]
                    for r in runs if len(r) > lag]
            line.append(f"lag{lag} {np.nanmean(vals):+.3f}")
        out(f"  autocorrelation of {lab:18s}: " + "  ".join(line))
    out("  (alternating sign with a positive lag-2 term = two-residue in/out periodicity)")

    (HERE / "results" / "findings.txt").write_text(buf.getvalue())
    print("\n-> results/findings.txt")


if __name__ == "__main__":
    main()
