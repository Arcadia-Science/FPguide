#!/usr/bin/env python
"""Does the family MSA say anything ESM-2 does not already know?

The guided objective already carries an ESM-2 masked-LM log-probability term, so a
conservation signal is only operationally useful where it adds to what ESM-2 supplies.
This script puts the two side by side at the EGFP window positions.

The ESM-2 distribution is produced exactly the way fpdesign/campaign.py produces it at
the first iteration (``esm_logits_at``): mask one position of the unmodified scaffold,
take the logits at that position, restrict to the 20 standard amino acids, softmax. So
``p_esm`` is literally the distribution the campaign draws its top-k from before any
edit is made.

A calibration panel runs first, because the headline result is only interpretable with
it: ESM-2 650M is sharp on ordinary proteins of comparable length (ubiquitin, lysozyme,
trypsin) and close to uninformative on FP-fold sequences. Without the controls
that would look like a bug in this script rather than a property of the model on this
family.

Writes results/esm_vs_family_egfp.csv and results/esm_calibration.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import esm

HERE = Path(__file__).resolve().parent
AAS = "ACDEFGHIKLMNPQRSTVWY"
CLASSES = {"aliphatic": "AVLIM", "aromatic": "FWY", "polar": "STNQ",
           "acidic": "DE", "basic": "KRH", "glycine": "G", "proline": "P", "cysteine": "C"}
CLASS_OF = {a: c for c, aas in CLASSES.items() for a in aas}


def js_divergence(p, q, eps=1e-12):
    p, q = p + eps, q + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda x, y: float((x * np.log2(x / y)).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# matched-length controls for the calibration panel
CONTROLS = {
    "ubiquitin (76 aa)":
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "lysozyme (129 aa)":
        "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLC"
        "NIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    # trypsin, UniProt P00760 mature chain (24-246); kept identical to esm_profiles.py's set
    "trypsin (223 aa)":
        "IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSY"
        "NSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAY"
        "PGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN",
}


def load_esm(dev):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    return model.eval().to(dev), alphabet


@torch.no_grad()
def esm_marginals(model, alphabet, seq, positions, dev, bs=32):
    """(len(positions), 20) masked-marginal probabilities, campaign convention."""
    bc = alphabet.get_batch_converter()
    aa_idx = torch.tensor([alphabet.get_idx(a) for a in AAS], device=dev)
    _, _, base = bc([("s", seq)])
    base = base.to(dev)
    out = {}
    for i in range(0, len(positions), bs):
        chunk = positions[i:i + bs]
        tk = base.repeat(len(chunk), 1).clone()
        rr = torch.arange(len(chunk), device=dev)
        cols = torch.tensor([p + 1 for p in chunk], device=dev)   # +1 for the CLS token
        tk[rr, cols] = alphabet.mask_idx
        lg = model(tk)["logits"][rr, cols].float()
        p = torch.softmax(lg[:, aa_idx], -1).cpu().numpy()        # restricted to the 20 AAs
        for j, pos in enumerate(chunk):
            out[pos] = p[j]
    return out


def calibration(model, alphabet, dev, extra):
    """Masked-marginal sharpness on control proteins vs FP-fold sequences."""
    rows = []
    for name, seq in list(CONTROLS.items()) + list(extra.items()):
        P = esm_marginals(model, alphabet, seq, list(range(len(seq))), dev)
        ranks, top1, maxp = [], [], []
        for p, a in enumerate(seq):
            pr = P[p]
            order = np.argsort(-pr)
            ranks.append(int(np.where(order == AAS.index(a))[0][0]) + 1)
            top1.append(AAS[int(pr.argmax())] == a)
            maxp.append(float(pr.max()))
        rows.append(dict(protein=name, length=len(seq),
                         top1_accuracy=round(float(np.mean(top1)), 3),
                         mean_max_prob=round(float(np.mean(maxp)), 3),
                         median_rank_true=float(np.median(ranks)),
                         frac_true_in_top10=round(float(np.mean(np.array(ranks) <= 10)), 3)))
    return pd.DataFrame(rows)


def main():
    cmp_ = pd.read_csv(HERE / "results" / "design_window_comparison.csv")
    cmp_ = cmp_[cmp_.campaign == "EGFP"].copy()
    alpha = pd.read_csv(HERE / "results" / "window_family_alphabet.csv")
    alpha = alpha[alpha.campaign == "EGFP"][["pos_1based", "family_alphabet_90"]]
    cons = pd.read_csv(HERE / "results" / "column_conservation.csv")

    import json
    w = json.load(open(HERE.parent / "design-campaign-EGFP" /
                       "design_windows_egfp_tierB.json"))["windows"]["EGFP"]
    seq = w["scaffold_seq"]

    # family frequencies keyed by EGFP 1-based position, via the avGFP-equivalent column
    fam = {}
    for r in cmp_.itertuples():
        row = cons[cons.ref_pos == r.avgfp_pos].iloc[0]
        fam[r.pos_1based] = np.array([float(row[f"f_aa_{a}"]) for a in AAS])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = load_esm(dev)

    meta = pd.read_csv(HERE / "data" / "fp_all_meta.csv")
    extra = {"EGFP (campaign scaffold)": seq,
             "avGFP (wild type)": meta.loc[meta.slug == "avgfp", "seq"].iloc[0],
             "mCherry": meta.loc[meta.slug == "mcherry", "seq"].iloc[0]}
    print(f"ESM-2 650M on {dev}; calibration panel (whole-sequence masked marginals)")
    cal = calibration(model, alphabet, dev, extra)
    cal.to_csv(HERE / "results" / "esm_calibration.csv", index=False)
    print(cal.to_string(index=False))
    print("  -> ESM-2 is sharp on the controls and near-uninformative on the FP fold;\n"
          "     the window comparison below has to be read in that light.")

    positions = sorted(int(p) - 1 for p in cmp_.pos_1based)        # 0-based into seq
    print(f"\nscoring {len(positions)} EGFP window positions")
    pe = esm_marginals(model, alphabet, seq, positions, dev)

    from scipy.stats import spearmanr
    rows = []
    for r in cmp_.itertuples():
        p_f = fam[r.pos_1based]
        p_e = pe[r.pos_1based - 1]
        top_e = AAS[int(p_e.argmax())]
        esm_order = list(np.argsort(-p_e))
        # class-level mass under each distribution, for the family's dominant class
        cls = r.top_class
        members = [i for i, a in enumerate(AAS) if CLASS_OF[a] == cls]
        rows.append(dict(
            pos_1based=r.pos_1based, scaffold_aa=r.scaffold_aa, avgfp_pos=r.avgfp_pos,
            role=r.role, constraint=r.constraint,
            fam_top=r.top_aa, fam_top_freq=round(float(p_f.max()), 3),
            esm_top=top_e, esm_top_p=round(float(p_e.max()), 3),
            top1_agree=(top_e == r.top_aa),
            fam_class=cls, fam_class_mass=round(float(p_f[members].sum()), 3),
            esm_class_mass=round(float(p_e[members].sum()), 3),
            js=round(js_divergence(p_f, p_e), 3),
            esm_p_scaffold_aa=round(float(p_e[AAS.index(r.scaffold_aa)]), 3),
            fam_p_scaffold_aa=round(float(p_f[AAS.index(r.scaffold_aa)]), 3),
            esm_top5="".join(AAS[i] for i in np.argsort(-p_e)[:5]),
            fam_top5="".join(AAS[i] for i in np.argsort(-p_f)[:5]),
            spearman_fam_esm=round(float(spearmanr(p_f, p_e).statistic), 3),
            # k=10 is the campaign default: is the family's preferred residue even reachable?
            fam_top_in_esm_top10=(AAS.index(r.top_aa) in esm_order[:10]),
        ))
    df = pd.DataFrame(rows).merge(alpha, on="pos_1based").sort_values("js", ascending=False)
    df.to_csv(HERE / "results" / "esm_vs_family_egfp.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n=== EGFP window: ESM-2 masked marginal vs family frequency (sorted by disagreement) ===")
    print(df[["pos_1based", "scaffold_aa", "role", "constraint", "fam_top", "fam_top_freq",
              "esm_top", "esm_top_p", "top1_agree", "fam_class", "fam_class_mass",
              "esm_class_mass", "js", "fam_top5", "esm_top5"]].to_string(index=False))

    ed = df[df.role == "editable"]
    print(f"\ntop-1 agreement: {int(df.top1_agree.sum())}/{len(df)} overall, "
          f"{int(ed.top1_agree.sum())}/{len(ed)} editable")
    print(f"mean Jensen-Shannon divergence (bits): {df.js.mean():.3f} "
          f"[fixed {df[df.role=='fixed'].js.mean():.3f}, editable {ed.js.mean():.3f}]")
    print(f"mean family-class mass: family {df.fam_class_mass.mean():.3f} "
          f"vs ESM-2 {df.esm_class_mass.mean():.3f}")
    print(f"mean Spearman(family freq, ESM prob) over window positions: "
          f"{df.spearman_fam_esm.mean():+.3f}  "
          f"({int((df.spearman_fam_esm < 0).sum())}/{len(df)} negative)")
    print(f"family's preferred residue inside ESM-2's top-{10} (the campaign's k): "
          f"{df.fam_top_in_esm_top10.mean():.2f} of positions")
    gap = ed[(ed.fam_class_mass - ed.esm_class_mass) > 0.25].sort_values(
        "fam_class_mass", ascending=False)
    print(f"\npositions where the family is far more decided than ESM-2 "
          f"(class mass gap > 0.25): {len(gap)}")
    print(gap[["pos_1based", "scaffold_aa", "fam_class", "fam_class_mass", "esm_class_mass",
               "fam_top5", "esm_top5", "family_alphabet_90"]].to_string(index=False))
    print("\n-> results/esm_vs_family_egfp.csv")


if __name__ == "__main__":
    main()
