#!/usr/bin/env python
"""Build ONE per-strategy wet-lab shortlist xlsx for the consolidated EGFP campaign.

    python make_shortlist_case.py <case>

where <case> is one of:  mOrange_gibbs     mOrange_spectra  mOrange_constr  mOrange_DMS  EBFP_DMS
                         mOrange_MSA       EBFP_MSA
                         mOrange_MSAgibbs  EBFP_MSAgibbs    EBFP_gibbs

Each file lists the two references (EGFP scaffold + the case's target, with their TRUE dataset ex/em)
followed by the top-10 DIVERSE designs (greedy, >= 5 residues apart in the edit window, ranked by
surrogate peak error). Every case pools ALL iteration rounds (>= 1) of ALL trials before selecting.
For the DMS-guide and MSA-guide cases the pool is first restricted to designs that are both
IN-DISTRIBUTION (ESM max-pool NN-distance to the 40k GFP-DMS reference <= its 99th pct) AND PREDICTED
BRIGHT (classifier logit > 0); the other strategies take the plain closest-10. Every design row is
annotated with `is_id` and `is_bright` (+ the raw brightness logit), the run it came from, and an
E. coli codon-optimized DNA sequence.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.neighbors import NearestNeighbors

REPO = Path("/home/ubuntu/spectrum-to-fp-design")
CAMP = REPO / "design-campaign-EGFP"
sys.path.insert(0, str(REPO / "esm2_design"))
sys.path.insert(0, str(CAMP))
import peak_models as pm
from embed_cache import MaxPoolCache

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N, MIN_HD = 10, 5

GIBBS = CAMP / "gibbs-sampling" / "designs" / "design_EGFP.csv"
SPEC  = CAMP / "guided-design" / "designs"
CONS  = CAMP / "guided-design-constraint" / "designs_lam-edit10"
DMS   = CAMP / "brightness-guided" / "guided_design" / "designs_lam-bright60_lam-edit10"
# MSA-guided: the family profile replaces ESM-2 as the proposal. This strategy was run as a 125-cell
# lambda sweep (12 trials x 3 rounds each) rather than at one setting, so its pool is the UNION of
# every cell -- i.e. every design the strategy produced anywhere in the sweep. That is a far deeper
# pool than the single-setting strategies (~2.6-3.1k unique designs vs 288), so its top-10 is
# selected from more candidates; see the note in visualize_campaign.ipynb.
MSA   = CAMP / "msa-guided" / "designs"
def _msa_pool(target):
    return sorted(MSA.glob(f"*/design_EGFP-{target}.csv"))
# MSA-gibbs: the same family profile with the surrogate removed from selection -- the unguided
# control for MSA-guided, and target-free like ESM gibbs, so ONE run serves both targets and
# peak_err is computed post-hoc against each ("gibbs" mode). It produces no predicted-bright designs
# at all (0/288), so like ESM gibbs it takes the plain closest-10 rather than the ID&bright filter,
# which would return an empty pool. See msa-gibbs/README.md.
MGIB  = CAMP / "msa-gibbs" / "designs" / "design_EGFP.csv"

# (target, alias, code, csv-or-list-of-csvs, mode, out-filename)
CASES = {
    "mOrange_gibbs":   ("mOrange", "gibbs",                     "gibbs",   GIBBS,                         "gibbs",     "shortlist_mOrange_gibbs.xlsx"),
    "EBFP_gibbs":      ("EBFP",    "gibbs",                     "gibbs",   GIBBS,                         "gibbs",     "shortlist_EBFP_gibbs.xlsx"),
    "mOrange_MSAgibbs":("mOrange", "MSA gibbs",                 "MSAgib",  MGIB,                          "gibbs",     "shortlist_mOrange_MSA-gibbs.xlsx"),
    "EBFP_MSAgibbs":   ("EBFP",    "MSA gibbs",                 "MSAgib",  MGIB,                          "gibbs",     "shortlist_EBFP_MSA-gibbs.xlsx"),
    "mOrange_spectra": ("mOrange", "spectra guide",             "spectra", SPEC / "design_EGFP-mOrange.csv", "plain",  "shortlist_mOrange_spectra-guide.xlsx"),
    "mOrange_constr":  ("mOrange", "constrained spectra guide", "constr",  CONS / "design_EGFP-mOrange.csv", "plain",  "shortlist_mOrange_constrained-spectra-guide.xlsx"),
    "mOrange_DMS":     ("mOrange", "DMS guide - bright",        "DMS",     DMS  / "design_EGFP-mOrange.csv", "id_bright", "shortlist_mOrange_DMS-guide.xlsx"),
    "EBFP_DMS":        ("EBFP",    "DMS guide - bright",        "DMS",     DMS  / "design_EGFP-EBFP.csv",    "id_bright", "shortlist_EBFP_DMS-guide.xlsx"),
    "mOrange_MSA":     ("mOrange", "MSA guide - bright",        "MSA",     _msa_pool("mOrange"),          "id_bright", "shortlist_mOrange_MSA-guide.xlsx"),
    "EBFP_MSA":        ("EBFP",    "MSA guide - bright",        "MSA",     _msa_pool("EBFP"),             "id_bright", "shortlist_EBFP_MSA-guide.xlsx"),
}
# reference (scaffold_seq/target_seq/true peaks) source per target: a guided CSV that carries them
REF_CSV = {"mOrange": SPEC / "design_EGFP-mOrange.csv", "EBFP": DMS / "design_EGFP-EBFP.csv"}
OUTDIR = CAMP / "shortlists"

# ---- E. coli codon-optimized back-translation ----
ECOLI_CODON = {"A":"GCG","R":"CGC","N":"AAC","D":"GAT","C":"TGC","Q":"CAG","E":"GAA","G":"GGC",
               "H":"CAT","I":"ATT","L":"CTG","K":"AAA","M":"ATG","F":"TTT","P":"CCG","S":"AGC",
               "T":"ACC","W":"TGG","Y":"TAT","V":"GTG","*":"TAA"}
def reverse_translate(aa, add_stop=True):
    return "".join(ECOLI_CODON[a] for a in aa) + (ECOLI_CODON["*"] if add_stop else "")

# ---- ID (OOD) machinery: NN-distance to the 40k (10k/scaffold) GFP-DMS cloud <= its 99th pct ----
z = np.load(REPO / "GFP_DMS" / "DMS_data" / "esm_maxpool_4scaffold_10k.npz", allow_pickle=True)
mp = z["mp"]; mu, sd = mp.mean(0), mp.std(0) + 1e-6; Z = (mp - mu) / sd
nn = NearestNeighbors(n_neighbors=1).fit(Z)
p99 = float(np.percentile(NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[0][:, 1], 99))

@torch.no_grad()
def _max_embed_gpu(seqs, bs=64):
    out = []
    for i in range(0, len(seqs), bs):
        H, m = pm.resid_embed(list(seqs[i:i + bs]), dev); out.append(pm.masked_pool(H, m, "max").float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, mp.shape[1]), np.float32)

# Same on-disk cache the notebook uses, keyed on the DMS-window sequence: the MSA-guided cases pool
# 125 sweep cells (~2.7k designs each) and overlap heavily with what the notebook embeds, so without
# this every run re-embeds the same sequences from scratch.
max_embed = MaxPoolCache(CAMP / ".embed_cache" / "maxpool_esm2_650M.npz", _max_embed_gpu,
                         dim=mp.shape[1])

def id_dist(full_seqs):
    if not len(full_seqs):
        return np.array([])
    return nn.kneighbors((max_embed([s[3:238] for s in full_seqs]) - mu) / sd)[0].ravel()

# ---- brightness classifier (cnn-max-d2): logit > 0 == predicted bright ----
_bb, _bm = pm.load_model(str(REPO / "fpdesign" / "models" / "brightness_cnn-max-d2_40k.pt"), dev, out=1)
bnet = pm.wrap(_bb, _bm.get("mean", 0.0) or 0.0, _bm.get("std", 1.0) or 1.0, dev)
@torch.no_grad()
def bright_logit(full_seqs, bs=32):
    out = []
    for i in range(0, len(full_seqs), bs):
        H, m = pm.resid_embed(list(full_seqs[i:i + bs]), dev); out.append(bnet(H, m).float().cpu().numpy().ravel())
    return np.concatenate(out) if out else np.array([])

def diverse_topk(d, n=N, min_hd=MIN_HD):
    d = d.sort_values("peak_err"); keep, kept = [], []
    for i, s in zip(d.index, d["designed_seq"]):
        if all(sum(a != b for a, b in zip(s, p)) >= min_hd for p in kept):
            keep.append(i); kept.append(s)
            if len(keep) >= n: break
    if len(keep) < n:
        for i in d.index:
            if i not in set(keep):
                keep.append(i)
                if len(keep) >= n: break
    return d.loc[keep]

def build(case):
    target, alias, code, csv, mode, fname = CASES[case]
    ref = pd.read_csv(REF_CSV[target]).iloc[0]
    tex, tem = float(ref["target_ex"]), float(ref["target_em"])
    # `csv` is one file, or (MSA-guided) the whole sweep: pool every cell, then dedupe on sequence.
    # `source` keeps the run/lambda-cell each surviving design came from so a pick is reproducible.
    paths = list(csv) if isinstance(csv, list) else [csv]
    d = pd.concat([pd.read_csv(p).assign(source=Path(p).parent.name) for p in paths],
                  ignore_index=True)
    d = d[d["round"] >= 1].drop_duplicates("designed_seq").copy()
    if mode == "gibbs":
        d["peak_err"] = 0.5 * ((d["pred_ex"] - tex).abs() + (d["pred_em"] - tem).abs())
    ntot = len(d)
    # brightness: prefer the campaign's own logged pred_bright (DMS runs); else score with the same net
    blog_all = d["pred_bright"].to_numpy() if "pred_bright" in d.columns else bright_logit(d["designed_seq"].tolist())
    d["_blog"] = blog_all
    d["_idist"] = id_dist(d["designed_seq"].tolist())
    d["_is_id"] = d["_idist"] <= p99
    d["_is_bright"] = d["_blog"] > 0
    if mode == "id_bright":
        pool = d[d["_is_id"] & d["_is_bright"]]
        note = f"{len(pool)} ID & bright of {ntot}"
    else:
        pool = d
        note = f"all {ntot} (unfiltered)"
    sel = diverse_topk(pool, N, MIN_HD).reset_index(drop=True)

    rows = [dict(name="EGFP", role="reference", target="", strategy="",
                 true_ex_nm=int(ref["scaffold_ex"]), true_em_nm=int(ref["scaffold_em"]),
                 pred_ex_nm="", pred_em_nm="", is_id="", is_bright="", bright_logit="", source="",
                 aa_sequence=ref["scaffold_seq"], dna_sequence=reverse_translate(ref["scaffold_seq"])),
            dict(name=target, role="reference", target="", strategy="",
                 true_ex_nm=int(tex), true_em_nm=int(tem), pred_ex_nm="", pred_em_nm="",
                 is_id="", is_bright="", bright_logit="", source="",
                 aa_sequence=ref["target_seq"], dna_sequence=reverse_translate(ref["target_seq"]))]
    for i, (_, r) in enumerate(sel.iterrows(), 1):
        rows.append(dict(
            name=f"{target}_{code}_{i:02d}", role="design", target=target, strategy=alias,
            true_ex_nm="", true_em_nm="",
            pred_ex_nm=round(float(r["pred_ex"]), 1), pred_em_nm=round(float(r["pred_em"]), 1),
            is_id="yes" if bool(r["_is_id"]) else "no",
            is_bright="yes" if bool(r["_is_bright"]) else "no",
            bright_logit=round(float(r["_blog"]), 2), source=r["source"],
            aa_sequence=r["designed_seq"], dna_sequence=reverse_translate(r["designed_seq"]),
        ))
    COLS = ["name", "role", "target", "strategy", "true_ex_nm", "true_em_nm", "pred_ex_nm",
            "pred_em_nm", "is_id", "is_bright", "bright_logit", "source", "aa_sequence",
            "dna_sequence"]
    OUTDIR.mkdir(exist_ok=True)
    out = pd.DataFrame(rows)[COLS]
    dest = OUTDIR / fname
    out.to_excel(dest, index=False, sheet_name="shortlist")
    print(f"[{case}] pooled {len(paths)} run(s) | {note} | selected {len(sel)} designs "
          f"-> {dest.name}", flush=True)
    return dest

if __name__ == "__main__":
    build(sys.argv[1])
