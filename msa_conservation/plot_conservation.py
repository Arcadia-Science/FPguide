#!/usr/bin/env python
"""Figures for the whole-dataset FP conservation analysis.

Reads ``results/column_conservation.csv`` and writes ``figures/*.png``. Everything is
restricted to core columns with occupancy >= 0.9 (209 of 236) so no panel is driven by
a column that only half the family occupies.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import spearmanr

try:
    import arcadia_pycolor as apc

    apc.mpl.setup()
except Exception:
    pass

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OCC_MIN_FIG = 0.90

CLASS_COLOR = {"aliphatic": "#5088C5", "aromatic": "#3B3B6E", "polar": "#73B5E3",
               "acidic": "#F28360", "basic": "#7A77AB", "glycine": "#F7B846",
               "proline": "#97CD78", "cysteine": "#C85152"}
# landmarks worth naming on the figures (avGFP numbering)
LANDMARK = {67: "G67 (cyclization)", 96: "R96 (catalytic)", 222: "E222 (catalytic)",
            66: "Y66 (chromophore ring)", 65: "65 (chromophore X)", 27: "F27",
            130: "F130", 60: "L60", 148: "H148", 203: "T203", 205: "S205"}
# named on the track figure only; the rest crowd the 60-70 region illegibly
TRACK_LANDMARK = {27: "F27", 67: "G67", 96: "R96", 130: "F130", 222: "E222"}
# hand offsets where the (1,1) corner of the scatter gets crowded
LABEL_NUDGE = {96: (-26, 2), 27: (4, 10), 67: (6, -8), 12: (6, 4), 78: (2, -9),
               161: (-24, -2), 1: (-14, -4)}


def load():
    df = pd.read_csv(HERE / "results" / "column_conservation.csv")
    return df[df.occupancy >= OCC_MIN_FIG].copy()


def fig_tracks(hi):
    """Conservation, chemistry class and structural context along avGFP numbering."""
    d = hi[hi.ref_pos > 0].sort_values("ref_pos")
    x = d.ref_pos.values
    fig, ax = plt.subplots(4, 1, figsize=(15, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.1, 2.1, 0.7, 1.5], hspace=0.13))

    ax[0].bar(x, d.C_id, width=0.9, color="#BAB0A8", label="identity ($C_{id}$)")
    ax[0].plot(x, d.C_chem, lw=1.2, color="#C85152", label="chemistry ($C_{chem}$)")
    ax[0].set_ylabel("conservation\n(frac. of family\nentropy removed)")
    ax[0].legend(loc="upper right", ncol=2, fontsize=9, frameon=False)
    ax[0].set_ylim(0, 1.05)
    for p, lab in TRACK_LANDMARK.items():
        if p in set(x):
            ax[0].annotate(lab, (p, 1.06), ha="center", fontsize=8, va="bottom",
                           color="#C85152", annotation_clip=False)

    gap = d.C_chem - d.C_id
    ax[1].axhline(0, color="#999", lw=0.7)
    ax[1].bar(x, gap, width=0.9, color=[CLASS_COLOR[c] for c in d.top_class])
    ax[1].set_ylabel("$C_{chem} - C_{id}$\n(chemistry pinned,\nidentity free)")
    ax[1].legend(handles=[Patch(facecolor=v, label=k) for k, v in CLASS_COLOR.items()],
                 loc="upper right", ncol=4, fontsize=8, frameon=False)

    for i, r in enumerate(d.itertuples()):
        ax[2].add_patch(plt.Rectangle((r.ref_pos - 0.5, 0), 1, 1,
                                      color=CLASS_COLOR[r.top_class],
                                      alpha=0.25 + 0.75 * min(r.top_class_freq, 1.0)))
    ax[2].set_ylim(0, 1)
    ax[2].set_yticks([])
    ax[2].set_ylabel("dominant\nclass", rotation=0, ha="right", va="center", fontsize=9)

    ax[3].fill_between(x, d.rsa, color="#73B5E3", alpha=0.55, step="mid", label="RSA (1GFL)")
    ax[3].plot(x, d.chromo_dist / 30, color="#F28360", lw=1.0, label="dist. to chromophore / 30 Å")
    for r in d.itertuples():
        if r.sse == "b":
            ax[3].add_patch(plt.Rectangle((r.ref_pos - 0.5, 1.02), 1, 0.09,
                                          color="#3B3B6E", clip_on=False))
    ax[3].set_ylabel("structure")
    ax[3].set_xlabel("avGFP position (1GFL numbering; dark bars = β-strand)")
    ax[3].legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    ax[3].set_ylim(0, 1.0)
    ax[3].set_xlim(0, 246)

    fig.suptitle("Conservation across 763 fluorescent proteins (82 organisms): "
                 "identity vs side-chain chemistry", y=0.94, fontsize=12)
    fig.savefig(FIG / "conservation_tracks.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_identity_vs_chem(hi):
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    lim = (-0.05, 1.05)
    ax.plot(lim, lim, ls="--", color="#999", lw=0.9, zorder=0)
    for cls, g in hi.groupby("top_class"):
        ax.scatter(g.C_id, g.C_chem, s=26, color=CLASS_COLOR[cls], label=cls,
                   edgecolor="white", lw=0.4, alpha=0.9)
    for r in hi.itertuples():
        if r.ref_pos > 0 and (r.ref_pos in LANDMARK or
                              (r.chem_minus_id > 0.25 and r.top_class_freq > 0.9)):
            ax.annotate(f"{r.ref_aa}{int(r.ref_pos)}", (r.C_id, r.C_chem), fontsize=8,
                        xytext=LABEL_NUDGE.get(int(r.ref_pos), (5, 3)),
                        textcoords="offset points", color="#484B50")
    ax.set(xlim=lim, ylim=lim, xlabel="identity conservation $C_{id}$  (20 amino acids)",
           ylabel="chemistry conservation $C_{chem}$  (8 side-chain classes)")
    ax.text(0.04, 0.97, "above the diagonal:\nchemistry conserved,\nidentity substitutable",
            transform=ax.transAxes, va="top", fontsize=9, color="#C85152")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_title("Chemistry is conserved where identity is not", fontsize=12)
    fig.savefig(FIG / "identity_vs_chemistry.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_burial(hi):
    panels = [("hydropathy_mean", "mean hydropathy (Kyte–Doolittle)"),
              ("polarity_mean", "mean polarity (Grantham)"),
              ("charge_rho", "charge constraint  ρ = 1 − Var/Var$_{bg}$"),
              ("hbond_mean", "mean side-chain H-bond capacity"),
              ("volume_mean", "mean side-chain volume (Å³)"),
              ("C_chem", "chemistry conservation $C_{chem}$")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    for ax, (col, lab) in zip(axes.ravel(), panels):
        v = hi[["rsa", col, "top_class"]].dropna()
        ax.scatter(v.rsa, v[col], s=20, c=[CLASS_COLOR[c] for c in v.top_class],
                   alpha=0.85, edgecolor="white", lw=0.3)
        r, p = spearmanr(v.rsa, v[col])
        z = np.polyfit(v.rsa, v[col], 1)
        xs = np.linspace(v.rsa.min(), v.rsa.max(), 50)
        ax.plot(xs, np.polyval(z, xs), color="#484B50", lw=1.2, ls="--")
        ax.set(xlabel="relative solvent accessibility (RSA)", ylabel=lab)
        ax.set_title(f"Spearman ρ = {r:+.2f}   (p = {p:.1e})", fontsize=9.5)
        if col == "charge_rho":
            ax.axhline(0, color="#C85152", lw=0.8)
    fig.suptitle("Conserved chemistry tracks the fold: buried = greasy and charge-free, "
                 "exposed = polar and electrostatically unconstrained", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "chemistry_vs_burial.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_pocket(hi):
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.2),
                           gridspec_kw=dict(width_ratios=[1.6, 1]))
    v = hi[["chromo_dist", "C_chem", "top_class", "ref_pos", "ref_aa"]].dropna()
    ax[0].scatter(v.chromo_dist, v.C_chem, s=28, c=[CLASS_COLOR[c] for c in v.top_class],
                  alpha=0.9, edgecolor="white", lw=0.4)
    r, p = spearmanr(v.chromo_dist, v.C_chem)
    ax[0].axvline(5, color="#C85152", ls="--", lw=1.0)
    ax[0].text(5.3, 0.02, "5 Å contact shell", color="#C85152", fontsize=8.5)
    for pos in (67, 96, 222, 66, 65, 60):
        row = v[v.ref_pos == pos]
        if len(row):
            ax[0].annotate(f"{row.ref_aa.iloc[0]}{pos}", (row.chromo_dist.iloc[0], row.C_chem.iloc[0]),
                           fontsize=9, xytext=(5, 4), textcoords="offset points", weight="bold")
    ax[0].set(xlabel="distance to chromophore (Å, 1GFL)", ylabel="chemistry conservation $C_{chem}$")
    ax[0].set_title(f"No conservation gradient toward the chromophore\n"
                    f"Spearman ρ = {r:+.3f}, p = {p:.2f}", fontsize=10.5)

    hi = hi.copy()
    hi["zone"] = np.where(hi.chromo_dist <= 5, "pocket\n≤5 Å",
                          np.where(hi.chromo_dist <= 10, "shell\n5–10 Å", "outer\n>10 Å"))
    order = ["pocket\n≤5 Å", "shell\n5–10 Å", "outer\n>10 Å"]
    data = [hi.loc[hi.zone == z, "C_chem"].dropna().values for z in order]
    bp = ax[1].boxplot(data, labels=order, showmeans=True, widths=0.6, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#F28360", "#F7B846", "#73B5E3"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    for i, dd in enumerate(data):
        ax[1].scatter(np.random.normal(i + 1, 0.06, len(dd)), dd, s=9, color="#484B50", alpha=0.5)
    ax[1].set_ylabel("chemistry conservation $C_{chem}$")
    ax[1].set_title("The colour-tuning pocket is\nno more constrained than the shell", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "pocket_vs_scaffold.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_class_summary(hi):
    g = hi.groupby("top_class").agg(n=("C_id", "size"), C_id=("C_id", "mean"),
                                    C_chem=("C_chem", "mean")).sort_values("C_chem")
    locked = hi[hi.top_class_freq >= 0.9].top_class.value_counts()
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.0))
    y = np.arange(len(g))
    ax[0].barh(y - 0.2, g.C_id, height=0.4, color="#BAB0A8", label="identity $C_{id}$")
    ax[0].barh(y + 0.2, g.C_chem, height=0.4, color=[CLASS_COLOR[c] for c in g.index],
               label="chemistry $C_{chem}$")
    ax[0].set_yticks(y, [f"{c}  (n={n})" for c, n in zip(g.index, g.n)])
    ax[0].set_xlabel("mean conservation")
    ax[0].legend(fontsize=9, frameon=False, loc="lower right")
    ax[0].set_title("Glycine and aromatics are conserved by identity;\n"
                    "aliphatics only by chemistry", fontsize=10.5)

    thresholds = [0.95, 0.90, 0.80, 0.70]
    n_id = [(hi.top_aa_freq >= t).sum() for t in thresholds]
    n_ch = [(hi.top_class_freq >= t).sum() for t in thresholds]
    xx = np.arange(len(thresholds))
    ax[1].bar(xx - 0.2, n_id, 0.4, color="#BAB0A8", label="same amino acid")
    ax[1].bar(xx + 0.2, n_ch, 0.4, color="#C85152", label="same chemistry class")
    for i, (a, b) in enumerate(zip(n_id, n_ch)):
        ax[1].annotate(str(a), (i - 0.2, a), ha="center", va="bottom", fontsize=9)
        ax[1].annotate(str(b), (i + 0.2, b), ha="center", va="bottom", fontsize=9)
    ax[1].set_xticks(xx, [f"≥{int(t*100)}%" for t in thresholds])
    ax[1].set(xlabel="conservation threshold (weighted frequency)",
              ylabel="number of positions (of 209)")
    ax[1].legend(fontsize=9, frameon=False)
    ax[1].set_title("Chemistry is conserved at ~2× as many\npositions as identity", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "class_conservation_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_pocket_composition(hi):
    """Class composition of every position in the 5 A chromophore shell."""
    from conservation import CLASS_NAMES
    p = hi[hi.chromo_dist <= 5].sort_values("ref_pos")
    M = p[[f"f_class_{c}" for c in CLASS_NAMES]].values
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    bottom = np.zeros(len(p))
    for k, cls in enumerate(CLASS_NAMES):
        ax.bar(range(len(p)), M[:, k], bottom=bottom, color=CLASS_COLOR[cls],
               label=cls, width=0.86)
        bottom += M[:, k]
    ax.set_xticks(range(len(p)), [f"{a}{int(q)}" for a, q in zip(p.ref_aa, p.ref_pos)],
                  rotation=90, fontsize=8)
    ax.set(ylabel="weighted class frequency", ylim=(0, 1.0),
           xlabel="avGFP position in the 5 Å chromophore contact shell")
    ax.legend(fontsize=8, ncol=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    ax.set_title("Chemistry of the chromophore pocket: three fixed anchors "
                 "(G67, R96, E222) in an otherwise plastic cavity", fontsize=11, pad=10)
    fig.savefig(FIG / "pocket_composition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_design_window(hi):
    """The EGFP edit window, ranked by how hard the family constrains each position."""
    cmp_path = HERE / "results" / "window_vs_family_egfp.csv"
    alpha_path = HERE / "results" / "window_alphabet_egfp.csv"
    if not cmp_path.exists():
        print("skipping design-window figure (run window_vs_family.py first)")
        return
    d = pd.read_csv(cmp_path)
    a = pd.read_csv(alpha_path)[["pos_1based", "n_aa_90", "family_alphabet_90"]]
    d = d.merge(a, on="pos_1based")
    d = d.sort_values("C_chem", ascending=False).reset_index(drop=True)

    style = {"fixed": ("#484B50", "hard-fixed"), "aromatic": ("#7A77AB", "aromatic-restricted"),
             "hbond": ("#F7B846", "Tier-B H-bond alphabet"), "none": ("#73B5E3", "unrestricted")}
    key = np.where(d.role == "fixed", "fixed", d.constraint)

    fig, ax = plt.subplots(2, 1, figsize=(13.5, 7.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.4, 1], hspace=0.1))
    x = np.arange(len(d))
    ax[0].bar(x, d.C_chem, color=[style[k][0] for k in key], width=0.78)
    ax[0].axhline(0.90, ls="--", lw=0.9, color="#C85152")
    ax[0].text(len(d) - 0.4, 0.915, "chemistry-locked in the family", ha="right",
               fontsize=8.5, color="#C85152")
    ax[0].set_ylabel("family chemistry conservation $C_{chem}$")
    ax[0].set_ylim(0, 1.06)
    seen, handles = set(), []
    for k in key:
        if k not in seen:
            seen.add(k)
            handles.append(Patch(facecolor=style[k][0], label=style[k][1]))
    ax[0].legend(handles=handles, fontsize=9, frameon=False, ncol=4, loc="upper right")
    ax[0].set_title("EGFP 5 Å design window vs. what the FP family actually constrains\n"
                    "(sorted by family constraint; L61 is the only strongly constrained "
                    "position the window leaves unrestricted)", fontsize=11.5)
    for i, r in enumerate(d.itertuples()):
        if r.C_chem >= 0.62 and r.role != "fixed":
            ax[0].annotate(r.family_alphabet_90, (i, r.C_chem), ha="center", va="bottom",
                           fontsize=8, color="#484B50")

    ax[1].bar(x, d.n_aa_90, color=[style[k][0] for k in key], width=0.78)
    ax[1].set_ylabel("residues covering\n90% of family mass")
    ax[1].set_xticks(x, [f"{r.scaffold_aa}{r.pos_1based}" for r in d.itertuples()],
                     rotation=90, fontsize=8.5)
    ax[1].set_xlabel("EGFP window position (EGFP numbering)")
    ax[1].invert_yaxis()
    fig.savefig(FIG / "design_window_vs_conservation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    mpl.rcParams["savefig.facecolor"] = "white"
    hi = load()
    fig_tracks(hi)
    fig_identity_vs_chem(hi)
    fig_burial(hi)
    fig_pocket(hi)
    fig_class_summary(hi)
    fig_pocket_composition(hi)
    fig_design_window(hi)
    print(f"wrote {len(list(FIG.glob('*.png')))} figures to {FIG}")


if __name__ == "__main__":
    main()
