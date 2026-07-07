#!/usr/bin/env python
"""Visualize the dual split of the ex/em (peak) dataset as a bipartite map.

Mirrors block 3b of
`../peak_design/archive/data_processing/curate_split_visualize.ipynb`:
how surrogate roles (left) map onto oracle roles (right). Node size grows with
the number of FPs in that role; edge width/label is the number of FPs shared
between a surrogate role and an oracle role. By construction `S_test` sits inside
`O_train` and `O_test` inside `S_train`, so each side's test set is fully seen
during the other side's training.

Reads `dual_splits.csv` (columns: index, name, surrogate_role, oracle_role).
"""
import argparse
import csv
from collections import Counter

import matplotlib.pyplot as plt

try:
    import arcadia_pycolor as apc

    apc.mpl.setup()
    HAVE_APC = True
except Exception:
    HAVE_APC = False


class _C:
    """Arcadia palette with a plain-hex fallback when arcadia_pycolor is absent."""

    def __getattr__(self, k):
        if HAVE_APC:
            return getattr(apc, k)
        return {
            "aegean": "#5088C5",
            "amber": "#F28360",
            "gray": "#9a9a9a",
        }.get(k, "#333")


C = _C()


def load_roles(path):
    Srole, Orole = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            Srole.append(r["surrogate_role"])
            Orole.append(r["oracle_role"])
    return Srole, Orole


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        default="data/peak/curated/dual_splits.csv",
        help="Path to dual_splits.csv (default: %(default)s).",
    )
    ap.add_argument(
        "--out",
        default="data/peak/curated/dual_split_bipartite.png",
        help="Output image path (default: %(default)s).",
    )
    args = ap.parse_args()

    Srole, Orole = load_roles(args.csv)
    roles = ["train", "val", "test"]

    # contingency: # FPs shared between each surrogate role and each oracle role
    pair = Counter(zip(Srole, Orole))
    M = {(s, o): pair.get((s, o), 0) for s in roles for o in roles}
    Scnt = Counter(Srole)
    Ocnt = Counter(Orole)
    Stot = {s: Scnt.get(s, 0) for s in roles}
    Otot = {o: Ocnt.get(o, 0) for o in roles}

    print(f"curated N={len(Srole)}")
    print("surrogate-role -> oracle-role contingency:")
    for s in roles:
        print(f"  S_{s:5s} -> " + ", ".join(f"O_{o} {M[(s, o)]}" for o in roles))

    yp = {"train": 2, "val": 1, "test": 0}
    LX, RX = 0.0, 2.0
    cmax = max(M.values())

    fig, ax = plt.subplots(figsize=(6, 6))
    for s in roles:
        for o in roles:
            c = M[(s, o)]
            if c == 0:
                continue
            ax.plot(
                [LX, RX],
                [yp[s], yp[o]],
                lw=1.2 + 11 * c / cmax,
                color=C.gray,
                alpha=0.5,
                zorder=1,
                solid_capstyle="round",
            )
            ax.text(
                LX + 0.5 * (RX - LX),
                yp[s] + 0.5 * (yp[o] - yp[s]),
                str(c),
                ha="center",
                va="center",
                fontsize=15,
                zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            )
    for s in roles:
        ax.scatter([LX], [yp[s]], s=10 * Stot[s], color=C.aegean, ec="k", lw=1.2, zorder=2)
        ax.text(LX - 0.8, yp[s], f"{s}\n{Stot[s]}", ha="right", va="center", fontsize=15)
    for o in roles:
        ax.scatter([RX], [yp[o]], s=10 * Otot[o], color=C.amber, ec="k", lw=1.2, zorder=2)
        ax.text(RX + 0.8, yp[o], f"{o}\n{Otot[o]}", ha="left", va="center", fontsize=15)
    ax.text(LX, 2.78, "surrogate", ha="center", fontsize=15, color=C.aegean, fontweight="bold")
    ax.text(RX, 2.78, "oracle", ha="center", fontsize=15, color=C.amber, fontweight="bold")
    ax.set_xlim(-1.6, 3.6)
    ax.set_ylim(-0.7, 3.15)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"saved -> {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
