#!/usr/bin/env python
"""Score every EGFP design (all iterations) with the avGFP-DMS brightness model and emit a shortlist.

Pipeline
--------
1. Collect every design row from both campaigns in this folder:
       guided_design/designs/design_EGFP-*.csv   (method = "guided")
       gibbs-sampling/designs/design_EGFP-*.csv   (method = "gibbs")
   Round 0 is the untouched EGFP scaffold; rounds >=1 are the design iterations. By default we keep
   ALL iterations (intermediate + final); pass --include-scaffold to also score round 0.

2. Predict log10 median brightness for each unique designed sequence with the best avGFP-DMS model,
   CNN(1)-max = ../avGFP_DMS/trained_models/full/cnn-max-d1_s0.pt (val MAE 0.21, r 0.94 in log10 units;
   Sarkisyan et al. 2016 avGFP DMS). Sequences are embedded with ESM-2 650M (per-residue, max-pooled),
   exactly the representation the model was trained on. NB: the DMS model was trained on avGFP-family
   variants; EGFP designs are close relatives, so this is a reasonable but out-of-distribution screen.

3. Filter out designs whose predicted log-brightness is below --threshold (default 2.96).

4. Build a shortlist: one row per unique sequence per (target, method), each with a stable id, the
   target vs surrogate-predicted ex/em peaks, the design coordinates (iteration, trial), and the amino
   acid + (E. coli codon-optimized) DNA sequence.

Usage
-----
    python brightness_shortlist.py                       # threshold 2.96 -> shortlist_brightness.csv
    python brightness_shortlist.py --threshold 3.0 --out my_shortlist.csv
    python brightness_shortlist.py --include-scaffold    # also score the raw EGFP scaffold (round 0)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent               # .../design-campaign-EGFP
REPO = HERE.parent                                    # .../spectrum-to-fp-design
sys.path.insert(0, str(REPO / "esm2_design"))
import peak_models as pm  # noqa: E402

DMS_CKPT = REPO / "avGFP_DMS" / "trained_models" / "full" / "cnn-max-d1_s0.pt"
DEFAULT_THRESHOLD = 2.96                               # log10 median brightness cutoff (keep >= this)

# design CSVs live one level down, split by selection method
METHOD_DIRS = {
    "guided": HERE / "guided_design" / "designs",
    "gibbs": HERE / "gibbs-sampling" / "designs",
}

# One codon per amino acid, picked from E. coli K-12 high-usage codons -- a standard choice for
# back-translating a protein into a synthesizable, well-expressed ORF. '*' -> stop (TAA).
ECOLI_CODON = {
    "A": "GCG", "R": "CGC", "N": "AAC", "D": "GAT", "C": "TGC", "Q": "CAG", "E": "GAA",
    "G": "GGC", "H": "CAT", "I": "ATT", "L": "CTG", "K": "AAA", "M": "ATG", "F": "TTT",
    "P": "CCG", "S": "AGC", "T": "ACC", "W": "TGG", "Y": "TAT", "V": "GTG", "*": "TAA",
}


def reverse_translate(aa: str, add_stop: bool = True) -> str:
    """Back-translate a protein to an E. coli codon-optimized ORF (optionally with a TAA stop)."""
    try:
        dna = "".join(ECOLI_CODON[a] for a in aa)
    except KeyError as e:  # non-standard residue (e.g. X) -- surface it rather than emit junk
        raise ValueError(f"no codon for residue {e.args[0]!r} in sequence") from e
    return dna + (ECOLI_CODON["*"] if add_stop else "")


def load_designs(include_scaffold: bool) -> pd.DataFrame:
    frames = []
    for method, d in METHOD_DIRS.items():
        files = sorted(d.glob("design_EGFP-*.csv"))
        if not files:
            print(f"  WARNING: no design CSVs found in {d}", flush=True)
        for f in files:
            df = pd.read_csv(f)
            df["method"] = method
            frames.append(df)
    if not frames:
        raise FileNotFoundError("no design CSVs found under guided_design/ or gibbs-sampling/")
    allrows = pd.concat(frames, ignore_index=True)
    if not include_scaffold:
        allrows = allrows[allrows["round"] >= 1].copy()   # drop round-0 (raw EGFP scaffold)
    return allrows


@torch.no_grad()
def score_brightness(seqs, dev, bs=16):
    """Map each unique sequence -> predicted log10 median brightness via CNN(1)-max on ESM-2 650M."""
    base, ck = pm.load_model(DMS_CKPT, dev, out=1)
    net = pm.wrap(base, ck["mean"], ck["std"], dev)       # forward returns brightness in log10 units
    net.eval()
    uniq = list(dict.fromkeys(seqs))                      # de-dupe, preserve order
    preds = {}
    for i in range(0, len(uniq), bs):
        chunk = uniq[i:i + bs]
        H, mask = pm.resid_embed(chunk, dev)
        p = net(H, mask)[:, 0].float().cpu().numpy()
        preds.update(dict(zip(chunk, p.tolist())))
        print(f"  scored {min(i + bs, len(uniq))}/{len(uniq)} unique sequences", flush=True)
    return preds, ck


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="keep designs with predicted log10 brightness >= this (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=HERE / "shortlist_brightness.csv",
                    help="output shortlist CSV (default: %(default)s)")
    ap.add_argument("--batch-size", type=int, default=16, help="ESM-2 embedding batch size")
    ap.add_argument("--include-scaffold", action="store_true",
                    help="also score round-0 (the untouched EGFP scaffold)")
    ap.add_argument("--no-stop-codon", action="store_true", help="omit the trailing TAA stop in DNA")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev} | model: {DMS_CKPT.relative_to(REPO)} (CNN(1)-max)", flush=True)

    df = load_designs(a.include_scaffold)
    print(f"loaded {len(df)} design rows | {df['designed_seq'].nunique()} unique sequences "
          f"| targets: {', '.join(sorted(df['target_name'].unique()))}", flush=True)

    preds, ck = score_brightness(df["designed_seq"].tolist(), dev, bs=a.batch_size)
    df["pred_log_brightness"] = df["designed_seq"].map(preds)

    keep = df[df["pred_log_brightness"] >= a.threshold].copy()
    n_seq_before = df["designed_seq"].nunique()
    n_seq_after = keep["designed_seq"].nunique()
    print(f"brightness filter (>= {a.threshold}): kept {n_seq_after}/{n_seq_before} unique sequences",
          flush=True)

    # one row per unique sequence within (target, method): keep the earliest (trial, round) it appeared
    keep.sort_values(["target_name", "method", "trial", "round"], inplace=True)
    keep = keep.drop_duplicates(subset=["target_name", "method", "designed_seq"], keep="first")

    # order each (target, method) block by spectral match (best first) and assign a stable index/id
    keep.sort_values(["target_name", "method", "peak_err"], inplace=True)
    keep["idx"] = keep.groupby(["target_name", "method"]).cumcount() + 1
    keep["id"] = [f"{t}-{m}-{i:03d}" for t, m, i in
                  zip(keep["target_name"], keep["method"], keep["idx"])]
    keep["dna_sequence"] = keep["designed_seq"].apply(
        lambda s: reverse_translate(s, add_stop=not a.no_stop_codon))

    out = keep.rename(columns={
        "target_ex": "target_ex_nm", "pred_ex": "pred_ex_nm",
        "target_em": "target_em_nm", "pred_em": "pred_em_nm",
        "peak_err": "peak_err_nm", "round": "iteration", "designed_seq": "aa_sequence",
    })[[
        "id", "target_name", "method",
        "target_ex_nm", "pred_ex_nm", "target_em_nm", "pred_em_nm",
        "peak_err_nm", "pred_log_brightness", "iteration", "trial",
        "aa_sequence", "dna_sequence",
    ]].reset_index(drop=True)

    out.to_csv(a.out, index=False)
    print(f"\nwrote {len(out)} shortlisted designs -> {a.out}", flush=True)
    print("\nper target x method:", flush=True)
    tab = (out.groupby(["target_name", "method"])
              .agg(n=("id", "size"),
                   best_peak_err=("peak_err_nm", "min"),
                   max_log_bright=("pred_log_brightness", "max"))
              .reset_index())
    with pd.option_context("display.width", 120, "display.max_rows", None):
        print(tab.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
