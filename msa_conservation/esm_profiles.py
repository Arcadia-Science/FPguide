#!/usr/bin/env python
"""Per-position ESM-2 masked marginals, cached for the visualization notebook.

``esm_vs_family.py`` answers one focused question (the 28 EGFP window positions) and
stores only summary columns. The notebook needs the underlying *distributions* — the full
(L, 20) masked-marginal matrix — for the whole barrel and for more than one protein, plus
a family-wide sweep establishing that what EGFP shows is a property of the fold rather
than of one engineered scaffold.

Distributions are produced exactly as ``fpdesign/campaign.py`` produces them
(``esm_logits_at``): mask one position of the unmodified sequence, take the logits there,
restrict to the 20 standard amino acids, softmax. So these are literally the distributions
the design search samples from.

Writes (all under ``results/``):

``esm_profiles.npz``       (L, 20) masked marginals for the reference set — three FPs,
                           three matched-length ordinary proteins.
``esm_profiles_meta.json`` sequences and group labels for that set.
``esm_family_sweep.csv``   one row per protein for a clade-stratified sample of the
                           aligned family: masked-marginal sharpness, and the mean
                           Jensen-Shannon divergence from the family's own column
                           distribution over the shared core columns.
``esm_sweep_profiles.npz`` the sweep's raw (L, 20) matrices, so anything derived from
                           them can be recomputed without a GPU.

Run once (needs a GPU; ~9 min):  python esm_profiles.py
Cached matrices are reused on re-runs; delete the .npz files to force recomputation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
AAS = "ACDEFGHIKLMNPQRSTVWY"
OCC_MIN = 0.90          # the 208 near-complete columns every other analysis uses
N_SWEEP = 96            # proteins in the family sweep, stratified over source organisms

# Matched-length ordinary proteins. Without them the FP numbers look like a broken harness
# rather than a property of the family; identical code path produces both.
CONTROLS = {
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "lysozyme":
        "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLC"
        "NIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    # trypsin, UniProt P00760 mature chain (24-246); 223 aa is within a dozen residues of the
    # FPs' 236-239, so the FP/control gap below cannot be a sequence-length effect
    "trypsin":
        "IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSY"
        "NSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAY"
        "PGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN",
}


# --------------------------------------------------------------------------- ESM-2 ------
def load_esm(dev):
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    return model.eval().to(dev), alphabet


def esm_marginals(model, alphabet, seq, dev, bs=48):
    """(len(seq), 20) masked-marginal probabilities over AAS, campaign convention."""
    import torch

    with torch.no_grad():
        bc = alphabet.get_batch_converter()
        aa_idx = torch.tensor([alphabet.get_idx(a) for a in AAS], device=dev)
        _, _, base = bc([("s", seq)])
        base = base.to(dev)
        out = np.zeros((len(seq), 20), dtype=np.float32)
        for i in range(0, len(seq), bs):
            chunk = list(range(i, min(i + bs, len(seq))))
            tk = base.repeat(len(chunk), 1).clone()
            rr = torch.arange(len(chunk), device=dev)
            cols = torch.tensor([p + 1 for p in chunk], device=dev)   # +1 for the CLS token
            tk[rr, cols] = alphabet.mask_idx
            lg = model(tk)["logits"][rr, cols].float()
            out[chunk] = torch.softmax(lg[:, aa_idx], -1).cpu().numpy()
    return out


# ---------------------------------------------------------------------- family side -----
def read_alignment(path):
    """{header: aligned_seq} in file order."""
    seqs, name, buf = {}, None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf)
            name, buf = line[1:].strip(), []
        else:
            buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def js_divergence(p, q, eps=1e-12):
    p, q = p + eps, q + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda x, y: float((x * np.log2(x / y)).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def entropy_bits(p, eps=1e-12):
    p = np.asarray(p, dtype=float)
    return float(-(p * np.log2(p + eps)).sum())


def family_matrix(cons):
    """(n_col, 20) weighted family frequencies, indexed by alignment column."""
    F = cons[[f"f_aa_{a}" for a in AAS]].to_numpy(dtype=np.float32)
    return dict(zip(cons.aln_col.to_numpy(), F))


def sample_family(meta, qc, n):
    """Clade-stratified sample: round-robin over source organisms, so the 251 Aequorea
    entries cannot swamp the 82-organism family the way an unweighted draw would."""
    df = meta.merge(qc[["slug", "core_frac"]], on="slug")
    df = df[(df.core_frac > 0.8) & (df.seq_len.between(200, 300))]    # drop tandem fusions
    df = df.sort_values(["parent_organism", "slug"])
    picked, groups = [], {o: list(g.slug) for o, g in df.groupby("parent_organism")}
    order = sorted(groups, key=lambda o: (-len(groups[o]), o))
    while len(picked) < n and any(groups.values()):
        for o in order:
            if groups[o] and len(picked) < n:
                picked.append(groups[o].pop(0))
    return df[df.slug.isin(picked)].reset_index(drop=True)


CLADE = {"Aequorea victoria": "Aequorea", "Aequorea macrodactyla": "Aequorea"}
ANTHOZOA = {"Entacmaea quadricolor", "Discosoma sp.", "Lobophyllia hemprichii",
            "Clavularia sp.", "Echinophyllia sp. SC22", "Verrillofungia concinna",
            "Montastraea cavernosa", "Acropora millepora", "Zoanthus sp.",
            "Corynactis californica", "Anemonia sulcata", "Anemonia majano",
            "Ricordea florida", "Scolymia cubensis", "Montipora sp.", "Fungia concinna",
            "Heteractis crispa", "Actinia equina", "Condylactis gigantea"}


def clade_of(org):
    if org in CLADE:
        return "Aequorea"
    if org in ANTHOZOA:
        return "Anthozoa"
    return "Other FP"


# ------------------------------------------------------------------------------ main ----
def cached(path):
    return dict(np.load(path)) if Path(path).exists() else {}


def main():
    import torch

    cons = pd.read_csv(RES / "column_conservation.csv")
    core = cons[cons.occupancy >= OCC_MIN].copy()
    FAM = family_matrix(core)
    core_cols = set(core.aln_col)

    meta = pd.read_csv(HERE / "data" / "fp_all_meta.csv")
    qc = pd.read_csv(RES / "sequence_qc.csv")
    aln = read_alignment(HERE / "data" / "fp_all.aln.fasta")
    aln_by_slug = {h.split("|")[-1]: s for h, s in aln.items()}     # headers are "id|slug"

    ref_cache = cached(RES / "esm_profiles.npz")
    sweep_cache = cached(RES / "esm_sweep_profiles.npz")
    model = alphabet = None
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def marginals(key, seq, cache):
        nonlocal model, alphabet
        if key in cache and cache[key].shape[0] == len(seq):
            return cache[key]
        if model is None:
            model, alphabet = load_esm(dev)
            print(f"ESM-2 650M on {dev}")
        return esm_marginals(model, alphabet, seq, dev)

    # --- reference set: full (L, 20) profiles kept for the per-position figures ----------
    egfp = json.load(open(HERE.parent / "design-campaign-EGFP" /
                          "design_windows_egfp_tierB.json"))["windows"]["EGFP"]["scaffold_seq"]
    ref = {"avGFP": meta.loc[meta.slug == "avgfp", "seq"].iloc[0],
           "EGFP": egfp,
           "mCherry": meta.loc[meta.slug == "mcherry", "seq"].iloc[0],
           **CONTROLS}
    group = {"avGFP": "FP", "EGFP": "FP", "mCherry": "FP",
             **{k: "control" for k in CONTROLS}}

    profiles = {}
    for name, seq in ref.items():
        profiles[name] = marginals(name, seq, ref_cache)
        top1 = np.mean([AAS[int(p.argmax())] == a for p, a in zip(profiles[name], seq)])
        print(f"  {name:<18} L={len(seq):>3}  top-1 {top1:.3f}  "
              f"mean max p {profiles[name].max(1).mean():.3f}  "
              f"mean H {np.mean([entropy_bits(p) for p in profiles[name]]):.2f} bits")
    np.savez_compressed(RES / "esm_profiles.npz", **profiles)
    json.dump({"sequences": ref, "group": group, "aas": AAS},
              open(RES / "esm_profiles_meta.json", "w"), indent=1)

    # --- family sweep: is the flatness a property of the fold or of one scaffold? --------
    samp = sample_family(meta, qc, N_SWEEP)
    print(f"\nfamily sweep: {len(samp)} proteins, "
          f"{samp.parent_organism.nunique()} organisms")
    rows, sweep_profiles = [], {}
    for k, r in enumerate(samp.itertuples(), 1):
        P = marginals(r.slug, r.seq, sweep_cache)
        sweep_profiles[r.slug] = P
        ranks = np.array([int(np.where(np.argsort(-P[i]) == AAS.index(a))[0][0]) + 1
                          for i, a in enumerate(r.seq)])
        # map residues onto alignment columns to compare against the family's own columns
        a_seq = aln_by_slug.get(r.slug)
        js, fam_H, esm_H, top1_fam = [], [], [], []
        if a_seq is not None:
            res_i = -1
            for col, ch in enumerate(a_seq):
                if ch == "-":
                    continue
                res_i += 1
                if col not in core_cols or res_i >= len(P):
                    continue
                f = FAM[col]
                js.append(js_divergence(f, P[res_i]))
                fam_H.append(entropy_bits(f))
                esm_H.append(entropy_bits(P[res_i]))
                top1_fam.append(int(f.argmax()) == int(P[res_i].argmax()))
        rows.append(dict(
            slug=r.slug, name=r.name, organism=r.parent_organism,
            clade=clade_of(r.parent_organism), length=r.seq_len,
            em_max=r.em_max,
            top1_accuracy=round(float(np.mean([AAS[int(p.argmax())] == a
                                               for p, a in zip(P, r.seq)])), 4),
            mean_max_prob=round(float(P.max(1).mean()), 4),
            mean_entropy_bits=round(float(np.mean([entropy_bits(p) for p in P])), 4),
            median_rank_true=float(np.median(ranks)),
            frac_true_in_top10=round(float(np.mean(ranks <= 10)), 4),
            n_core_cols=len(js),
            mean_js_family=round(float(np.mean(js)), 4) if js else np.nan,
            mean_family_entropy_bits=round(float(np.mean(fam_H)), 4) if js else np.nan,
            mean_esm_entropy_core_bits=round(float(np.mean(esm_H)), 4) if js else np.nan,
            top1_agree_family=round(float(np.mean(top1_fam)), 4) if js else np.nan,
        ))
        if k % 16 == 0:
            print(f"  {k}/{len(samp)}")
    np.savez_compressed(RES / "esm_sweep_profiles.npz", **sweep_profiles)
    sweep = pd.DataFrame(rows)

    # the controls get the same summary row so the sweep figure can show both populations
    for name, seq in CONTROLS.items():
        P = profiles[name]
        ranks = np.array([int(np.where(np.argsort(-P[i]) == AAS.index(a))[0][0]) + 1
                          for i, a in enumerate(seq)])
        sweep.loc[len(sweep)] = dict(
            slug=name.replace(" ", "_"), name=name, organism="control", clade="Control",
            length=len(seq), em_max=np.nan,
            top1_accuracy=round(float(np.mean([AAS[int(p.argmax())] == a
                                               for p, a in zip(P, seq)])), 4),
            mean_max_prob=round(float(P.max(1).mean()), 4),
            mean_entropy_bits=round(float(np.mean([entropy_bits(p) for p in P])), 4),
            median_rank_true=float(np.median(ranks)),
            frac_true_in_top10=round(float(np.mean(ranks <= 10)), 4),
            n_core_cols=0, mean_js_family=np.nan, mean_family_entropy_bits=np.nan,
            mean_esm_entropy_core_bits=np.nan, top1_agree_family=np.nan)

    sweep.to_csv(RES / "esm_family_sweep.csv", index=False)
    print(sweep.groupby("clade")[["top1_accuracy", "mean_max_prob", "mean_entropy_bits",
                                  "mean_js_family", "top1_agree_family"]].mean().round(3))
    print(f"\n-> {RES/'esm_profiles.npz'}, {RES/'esm_family_sweep.csv'}")


if __name__ == "__main__":
    main()
