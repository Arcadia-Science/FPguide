#!/usr/bin/env python
"""Propose POPULAR fluorescent proteins that (a) are in the curated dataset and
(b) have an ACTUAL experimental structure (reusing the cached RCSB >=97% search).
Grouped by color; reports PDB id, ex/em, Stokes shift, surrogate split role.
"""
import csv
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CUR = REPO / "dataset_pipeline" / "data" / "peak" / "curated"
STRUCT_HITS = REPO / "peak_design" / "peak_designs" / "structure" / "parallel_pipeline" / "pairs" / "structure_hits.csv"

# canonical / widely-used FP names to look for (exact dataset names)
POPULAR = {
    "blue":    ["EBFP2", "mTagBFP2", "TagBFP", "Azurite", "mKalama1", "SBFP2", "mTagBFP"],
    "cyan":    ["ECFP", "Cerulean", "mCerulean3", "mCerulean", "SCFP3A", "mTurquoise", "mTurquoise2",
                "CyPet", "mTFP1", "TagCFP", "AmCyan1", "amCyan1"],
    "green":   ["avGFP", "GFP", "EGFP", "mEGFP", "superfolder GFP", "sfGFP", "Emerald", "mEmerald",
                "Clover", "mClover3", "mNeonGreen", "TagGFP2", "AcGFP1", "TurboGFP", "ZsGreen",
                "Gamillus", "mGreenLantern", "StayGold", "mStayGold", "mStayGold2", "T-Sapphire", "mBaoJin"],
    "yellow":  ["EYFP", "Citrine", "mCitrine", "Venus", "mVenus", "YPet", "SYFP2", "TagYFP", "mGold",
                "Topaz", "Ypet"],
    "orange":  ["mOrange", "mOrange2", "mKO", "mKO2", "mKOkappa", "Kusabira Orange", "tdTomato",
                "TagRFP", "TagRFP-T", "mTangerine", "mRuby", "mRuby2", "mRuby3", "CyOFP1"],
    "red":     ["DsRed", "DsRed2", "DsRed-Express", "mRFP1", "mCherry", "mScarlet", "mScarlet-I",
                "mScarlet-H", "mApple", "mStrawberry", "FusionRed", "mKate", "mKate2", "TagRFP657"],
    "far-red": ["mNeptune", "mNeptune2.5", "mCardinal", "eqFP650", "eqFP670", "mGarnet", "mMaroon1",
                "Crimson", "mPlum", "E2-Crimson", "TagRFP675", "mNeptune681"],
    "switchable/PA": ["Kaede", "EosFP", "mEos2", "mEos3.2", "mEos4b", "Dendra2", "Dendra", "Dronpa",
                      "Dronpa-2", "Dronpa-3", "rsEGFP", "PA-GFP", "PS-CFP2", "KikG", "PAmCherry",
                      "IrisFP", "NijiFP", "mMaple"],
}


def load():
    rows = list(csv.DictReader(open(CUR / "peaks_assignments.csv")))
    N = len(rows)
    peaks = np.load(CUR / "peaks.npy").astype(np.float32)
    dual = {int(r["index"]): r["surrogate_role"] for r in csv.DictReader(open(CUR / "dual_splits.csv"))}
    name2idx = {r["name"]: i for i, r in enumerate(rows)}
    pdb_of = {int(r["idx"]): r["pdb_id"] for r in csv.DictReader(open(STRUCT_HITS)) if r["pdb_id"]}
    return rows, peaks, dual, name2idx, pdb_of


def main():
    rows, peaks, dual, name2idx, pdb_of = load()
    print(f"{'color':10}{'name':22}{'PDB':7}{'ex':>5}{'em':>5}{'SS':>5}{'split':>7}")
    n_hit = 0
    missing = []
    for color, names in POPULAR.items():
        shown = False
        for nm in names:
            if nm not in name2idx:
                missing.append(nm); continue
            i = name2idx[nm]
            if i not in pdb_of:          # present but no experimental structure >=97%
                continue
            ex, em = peaks[i]; ss = em - ex
            print(f"{(color if not shown else ''):10}{nm[:21]:22}{pdb_of[i]:7}{ex:>5.0f}{em:>5.0f}{ss:>5.0f}{dual[i]:>7}")
            shown = True; n_hit += 1
    print(f"\n{n_hit} popular + structure-backed FPs found in the dataset")
    # note which requested names are absent from the dataset (context only)
    print("not in dataset (by these exact names):", ", ".join(sorted(set(missing))))


if __name__ == "__main__":
    main()
