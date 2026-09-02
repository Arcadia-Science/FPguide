#!/usr/bin/env python
"""Build ONE per-strategy wet-lab shortlist xlsx for the consolidated EGFP campaign.

    python make_shortlist_case.py <case>     # one case
    python make_shortlist_case.py --all      # every case in CASES order
    python make_shortlist_case.py --verify-refs

where <case> is one of:  mOrange_gibbs     EBFP_gibbs
                         mOrange_MSA       EBFP_MSA
                         mOrange_MSAgibbs  EBFP_MSAgibbs

Each file lists the two references (EGFP scaffold + the case's target, with their TRUE dataset ex/em)
followed by the top-10 DIVERSE designs (greedy, >= 5 residues apart in the edit window, ranked by
surrogate peak error). Every case pools ALL iteration rounds (>= 1) of ALL trials before selecting,
and the MSA guide additionally pools EVERY LAMBDA CELL of its 125-cell sweep, so it is judged on
everything it produced rather than on one setting.
For the MSA-guide cases the pool is first restricted to designs that are both IN-DISTRIBUTION (ESM
max-pool NN-distance to the 40k GFP-DMS reference <= its 99th pct) AND CONFIDENTLY BRIGHT
(classifier logit > BRIGHT_T = 0.5, not merely the model's > 0 decision boundary); the other
strategies take the plain closest-10. Every design row is annotated with `is_id` and `is_bright`
(+ the raw brightness logit), `n_mut_vs_EGFP` (substitutions from the scaffold), the run it came
from, and an E. coli codon-optimized DNA sequence.

RETIRED STRATEGIES. The ESM-2 arm at the unmatched T=10 scale -- `spectra guide`,
`constrained spectra guide` and `DMS guide`, cases `mOrange_spectra` / `mOrange_constr` /
`mOrange_DMS` / `EBFP_DMS` -- was superseded by the matched-lambda grid in `esm2_guided/`
and no longer has shortlists. Its runs are in `archive/superseded-unmatched-runs/` and its six
xlsx in `archive/superseded-shortlists/`; nothing in active code reads either. The surviving
cross-strategy comparison lives in `benchmark_report.py` (equal budget) and
`esm2_guided/analyze.py`.

FROZEN SHORTLISTS. The four MSA files below back wet-lab batch 1 and cannot be silently rebuilt --
see the FROZEN block further down.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.neighbors import NearestNeighbors

CAMP = Path(__file__).resolve().parent        # .../design-campaign-EGFP
REPO = CAMP.parent                           # .../spectrum-to-fp-design
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CAMP))
from fpdesign import peak_models as pm
from embed_cache import MaxPoolCache
from xlsx_io import write_xlsx

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N, MIN_HD = 10, 5
# Selection bar for the brightness-guided strategies. The classifier's own decision boundary is
# logit > 0, but that is a 0.51-probability call at the margin and several designs were clearing it
# by hundredths of a logit. Shortlisting instead requires logit > 0.5 (p ~ 0.62), so a wet-lab slot
# is only spent on a design that is clear of the boundary. `is_bright` in the output still reports
# the model's own > 0 verdict; this constant only gates what may enter the pool.
BRIGHT_T = 0.5

GIBBS = CAMP / "gibbs-sampling" / "designs" / "design_EGFP.csv"
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
    "mOrange_gibbs":   ("mOrange", "gibbs",              "gibbs",  GIBBS,                "gibbs",     "shortlist_mOrange_gibbs.xlsx"),
    "EBFP_gibbs":      ("EBFP",    "gibbs",              "gibbs",  GIBBS,                "gibbs",     "shortlist_EBFP_gibbs.xlsx"),
    "mOrange_MSAgibbs":("mOrange", "MSA gibbs",          "MSAgib", MGIB,                 "gibbs",     "shortlist_mOrange_MSA-gibbs.xlsx"),
    "EBFP_MSAgibbs":   ("EBFP",    "MSA gibbs",          "MSAgib", MGIB,                 "gibbs",     "shortlist_EBFP_MSA-gibbs.xlsx"),
    "mOrange_MSA":     ("mOrange", "MSA guide - bright", "MSA",    _msa_pool("mOrange"), "id_bright", "shortlist_mOrange_MSA-guide.xlsx"),
    "EBFP_MSA":        ("EBFP",    "MSA guide - bright", "MSA",    _msa_pool("EBFP"),    "id_bright", "shortlist_EBFP_MSA-guide.xlsx"),
}

# These four shortlists are FROZEN: wet-lab batch 1 was chosen off them, and make_batch.py's PICKS
# pin design NAMES that are rank-derived (`f"{target}_{code}_{i:02d}"`, assigned in peak-error
# order) plus a hard-coded n_mut_vs_EGFP per pick. So a name does not pin a sequence -- rebuilding
# under any changed filter, threshold, model or input silently repoints a wet-lab construct at a
# different design. Frozen cases still rebuild in full; the result is compared against the file on
# disk and a mismatch is fatal (see build()). make_batch.py:80-84 catches the same drift one step
# later, where the diff is much harder to read.
#
# KNOWN spurious-failure mode, deliberately NOT pre-emptively fixed: diverse_topk sorts with
# pandas' default (non-stable) quicksort, so designs tied on peak_err may reorder across pandas
# versions. If the assert ever fires with "RE-RANK ONLY", pin kind="mergesort" there and re-verify
# against the frozen file -- do NOT overwrite the baseline. Changing the sort now could itself
# break the freeze.
FROZEN = frozenset({"mOrange_MSA", "EBFP_MSA", "mOrange_MSAgibbs", "EBFP_MSAgibbs"})
assert FROZEN <= set(CASES), f"FROZEN names not in CASES: {sorted(FROZEN - set(CASES))}"

# Reference (scaffold_seq/target_seq/true peaks) source per target. These six fields are
# run-invariant metadata, so they live in a tiny checked-in CSV rather than being lifted out of
# whichever design run happens to be around -- which is what broke every case when the unmatched
# runs were retired to the gitignored archive/. See references/README.md.
REFDIR  = CAMP / "references"
REF_CSV = {"mOrange": REFDIR / "reference_EGFP-mOrange.csv",
           "EBFP":    REFDIR / "reference_EGFP-EBFP.csv"}
REF_FIELDS = ("scaffold_seq", "target_seq", "scaffold_ex", "scaffold_em", "target_ex", "target_em")
OUTDIR = CAMP / "shortlists"


def verify_references():
    """Re-assert the checked-in reference rows against every live design CSV that carries them.

    Cheap insurance that references/ has not drifted from the runs the shortlists are built out
    of -- e.g. if a design run is ever regenerated against a different pairs CSV.
    """
    n_ok = 0
    for target, refp in REF_CSV.items():
        ref = pd.read_csv(refp).iloc[0]
        live = (sorted((CAMP / "msa-guided" / "designs").glob(f"*/design_EGFP-{target}.csv"))
                + sorted((CAMP / "esm2_guided" / "designs").glob(f"*/design_EGFP-{target}.csv")))
        if not live:
            raise SystemExit(f"--verify-refs: no live design CSV found for {target}")
        for p in live:
            row = pd.read_csv(p, nrows=1).iloc[0]
            for f in REF_FIELDS:
                if str(row[f]) != str(ref[f]):
                    raise SystemExit(f"--verify-refs: {p} {f}={row[f]!r} but {refp.name} "
                                     f"says {ref[f]!r}")
            n_ok += 1
        print(f"[verify-refs] {target}: {refp.name} matches all {len(live)} live design CSVs")
    print(f"[verify-refs] OK -- {n_ok} files checked on {len(REF_FIELDS)} fields each")

# ---- E. coli codon-optimized back-translation ----
ECOLI_CODON = {"A":"GCG","R":"CGC","N":"AAC","D":"GAT","C":"TGC","Q":"CAG","E":"GAA","G":"GGC",
               "H":"CAT","I":"ATT","L":"CTG","K":"AAA","M":"ATG","F":"TTT","P":"CCG","S":"AGC",
               "T":"ACC","W":"TGG","Y":"TAT","V":"GTG","*":"TAA"}
def reverse_translate(aa, add_stop=True):
    return "".join(ECOLI_CODON[a] for a in aa) + (ECOLI_CODON["*"] if add_stop else "")

# ---- ID (OOD) machinery: NN-distance to the 40k (10k/scaffold) GFP-DMS cloud <= its 99th pct ----
# The cloud is sub40k -- the very rows the deployed brightness classifier was fitted and selected on
# -- so "in distribution" and "inside the training distribution" are one statement. Built by
# GFP_DMS/build_maxpool_cache.py.
_CLOUD = REPO / "GFP_DMS" / "DMS_data" / "sub40k_maxpool.npz"
if not _CLOUD.exists():
    raise SystemExit(f"{__file__}: missing {_CLOUD}\n\n"
                     "The in-distribution reference cloud is not on disk (gitignored, 197 MiB). Fetch the\n"
                     "published copy:\n\n"
                     "    gh release download reference-cloud-v1 -p sub40k_maxpool.npz -D GFP_DMS/DMS_data/\n\n"
                     "Or rebuild it from the two source DMS studies -- see the Reproduce block in\n"
                     "GFP_DMS/README.md -- ending in:\n\n"
                     "    python GFP_DMS/build_maxpool_cache.py\n")
z = np.load(_CLOUD, allow_pickle=True)
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

COLS = ["name", "role", "target", "strategy", "true_ex_nm", "true_em_nm", "pred_ex_nm",
        "pred_em_nm", "n_mut_vs_EGFP", "is_id", "is_bright", "bright_logit", "source",
        "aa_sequence", "dna_sequence"]
# Columns written as numbers-or-blank. Everything else is compared as plain text.
_NUM_COLS = frozenset({"true_ex_nm", "true_em_nm", "pred_ex_nm", "pred_em_nm",
                       "n_mut_vs_EGFP", "bright_logit"})


class FrozenShortlistChanged(SystemExit):
    """A frozen shortlist's rebuild disagrees with the file on disk."""


def _canon(df):
    """Both frames to the same all-string form so an xlsx round-trip cannot fake a difference.

    Round-tripping through xlsx is lossy in ways that make a raw DataFrame.equals useless: blanks
    come back as NaN, ints widen to float64 (0 -> 0.0, 488 -> 488.0), and NaN != NaN. Fixing all
    of that on one side only would still leave float repr drift, so both sides are canonicalized
    identically instead. 4 dp sits far below the 1 dp / 2 dp rounding build() applies, so a real
    change always survives and float noise never shows.
    """
    out = {}
    for c in COLS:
        s = df[c]
        if c in _NUM_COLS:
            v = pd.to_numeric(s, errors="coerce")
            out[c] = ["" if pd.isna(x) else f"{float(x):.4f}" for x in v]
        else:
            out[c] = ["" if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x)
                      for x in s]
    return pd.DataFrame(out, columns=COLS)


def _int(canon_val):
    """Canonical '9.0000' back to '9' for display; non-numeric/blank passes through."""
    try:
        return f"{float(canon_val):g}"
    except (TypeError, ValueError):
        return str(canon_val) or "-"


def _seq_delta(a, b, show=4):
    """Render two sequences by what differs between them, not by their (shared) prefix."""
    if len(a) != len(b):
        return f"len {len(a)}", f"len {len(b)}"
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    head = " ".join(f"{a[i]}{i + 1}{b[i]}" for i in diff[:show])
    tail = f" +{len(diff) - show} more" if len(diff) > show else ""
    return f"{len(diff)} subs", f"{head}{tail}"


def _frozen_report(case, dest, new, old, max_cells=40):
    """Human-readable diff of a frozen rebuild, or None if the two are identical.

    Written so the first thing a reader learns is WHICH KIND of drift happened: a pure re-rank
    (same designs, new names -- the case make_batch.py's pinned names care about) reads very
    differently from a changed selection or a changed value.
    """
    if new.equals(old):
        return None
    L = [f"FROZEN SHORTLIST CHANGED: {case}",
         f"  file: {dest}",
         f"  rows: {len(old)} on disk -> {len(new)} rebuilt"]

    if list(new.columns) != list(old.columns):
        L += [f"  COLUMNS differ; row-level diff suppressed.",
              f"    added:   {[c for c in new.columns if c not in set(old.columns)]}",
              f"    removed: {[c for c in old.columns if c not in set(new.columns)]}"]
        return "\n".join(L)

    n_des, o_des = new[new.role == "design"], old[old.role == "design"]
    n_seq, o_seq = set(n_des.aa_sequence), set(o_des.aa_sequence)
    if n_seq == o_seq:
        L.append("  SELECTION: RE-RANK ONLY -- the same designs, reordered:")
        o_name = dict(zip(o_des.aa_sequence, o_des.name))
        for s, nm in zip(n_des.aa_sequence, n_des.name):
            if o_name[s] != nm:
                L.append(f"    {o_name[s]} -> {nm}")
    else:
        # Names are rank-derived, so the SAME name can appear in both lists pointing at different
        # sequences -- which is exactly the hazard this whole check exists for. Label the side
        # each name came from so that reads as information rather than as a contradiction.
        L.append("  SELECTION CHANGED (names are rank-derived; a name may appear on both sides):")
        for s in sorted(o_seq - n_seq):
            r = o_des[o_des.aa_sequence == s].iloc[0]
            L.append(f"    dropped  was {r['name']:<20s} n_mut={_int(r['n_mut_vs_EGFP']):<4s} src={r['source']}")
        for s in sorted(n_seq - o_seq):
            r = n_des[n_des.aa_sequence == s].iloc[0]
            L.append(f"    added    now {r['name']:<20s} n_mut={_int(r['n_mut_vs_EGFP']):<4s} src={r['source']}")

    # Positional, because rank position IS the identity make_batch.py pins.
    shown, more = 0, 0
    cells = []
    for i in range(min(len(new), len(old))):
        for c in COLS:
            if new.at[i, c] == old.at[i, c]:
                continue
            if shown < max_cells:
                a, b = old.at[i, c], new.at[i, c]
                if c in ("aa_sequence", "dna_sequence"):
                    # Every design shares the EGFP scaffold, so a prefix would look identical on
                    # both sides. Report the substitutions instead -- that IS the difference.
                    a, b = _seq_delta(a, b)
                cells.append(f"    row {i:2d} [{old.at[i, 'name']}] {c}: {a} -> {b}")
                shown += 1
            else:
                more += 1
    if cells:
        L.append("  CELL DIFFS (by row position):")
        L += cells
        if more:
            L.append(f"    ... ({more} more)")

    L += ["",
          "  make_batch.py PICKS pin these NAMES and assert n_mut_vs_EGFP (make_batch.py:50-84),",
          "  so an accepted change here can silently repoint a wet-lab construct. The file was",
          "  NOT overwritten. To deliberately accept a new selection:",
          f"    1. rm {dest}",
          f"    2. python make_shortlist_case.py {case}      # writes a new frozen baseline",
          "    3. python make_batch.py                      # re-verify every pick, then git add"]
    return "\n".join(L)


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
        pool = d[d["_is_id"] & (d["_blog"] > BRIGHT_T)]
        note = f"{len(pool)} ID & bright(logit>{BRIGHT_T:g}) of {ntot}"
    else:
        pool = d
        note = f"all {ntot} (unfiltered)"
    sel = diverse_topk(pool, N, MIN_HD).reset_index(drop=True)

    scaffold_seq = ref["scaffold_seq"]
    def n_mut(seq):
        """Substitutions vs the EGFP scaffold. Designs only ever substitute inside the Tier-B
        window, so this is a plain Hamming distance; the target reference can be a different
        length (mOrange is 236 aa vs EGFP's 239), and is reported as blank rather than a
        misleading truncated count."""
        return sum(a != b for a, b in zip(seq, scaffold_seq)) if len(seq) == len(scaffold_seq) else ""

    rows = [dict(name="EGFP", role="reference", target="", strategy="",
                 true_ex_nm=int(ref["scaffold_ex"]), true_em_nm=int(ref["scaffold_em"]),
                 pred_ex_nm="", pred_em_nm="", n_mut_vs_EGFP=0,
                 is_id="", is_bright="", bright_logit="", source="",
                 aa_sequence=scaffold_seq, dna_sequence=reverse_translate(scaffold_seq)),
            dict(name=target, role="reference", target="", strategy="",
                 true_ex_nm=int(tex), true_em_nm=int(tem), pred_ex_nm="", pred_em_nm="",
                 n_mut_vs_EGFP=n_mut(ref["target_seq"]),
                 is_id="", is_bright="", bright_logit="", source="",
                 aa_sequence=ref["target_seq"], dna_sequence=reverse_translate(ref["target_seq"]))]
    for i, (_, r) in enumerate(sel.iterrows(), 1):
        rows.append(dict(
            name=f"{target}_{code}_{i:02d}", role="design", target=target, strategy=alias,
            true_ex_nm="", true_em_nm="",
            pred_ex_nm=round(float(r["pred_ex"]), 1), pred_em_nm=round(float(r["pred_em"]), 1),
            n_mut_vs_EGFP=n_mut(r["designed_seq"]),
            is_id="yes" if bool(r["_is_id"]) else "no",
            is_bright="yes" if bool(r["_is_bright"]) else "no",
            bright_logit=round(float(r["_blog"]), 2), source=r["source"],
            aa_sequence=r["designed_seq"], dna_sequence=reverse_translate(r["designed_seq"]),
        ))
    OUTDIR.mkdir(exist_ok=True)
    out = pd.DataFrame(rows)[COLS]
    dest = OUTDIR / fname
    print(f"[{case}] pooled {len(paths)} run(s) | {note} | selected {len(sel)} designs", flush=True)

    if case in FROZEN and dest.exists():
        # The whole rebuild above had to run for this check to mean anything: it is the SELECTION
        # (pool -> dedupe -> ID -> brightness -> diverse_topk) being re-derived, not the file being
        # looked up. Short-circuiting on dest.exists() would reduce it to a file-existence test.
        diff = _frozen_report(case, dest, _canon(out), _canon(pd.read_excel(dest)))
        if diff:
            raise FrozenShortlistChanged(diff)
        # Deliberately NOT rewritten. xlsx bytes are not reproducible (docProps/core.xml carries
        # dcterms:created/modified, and every zip entry a wall-clock mtime), so rewriting would
        # dirty git status on every passing run.
        print(f"[{case}] FROZEN: rebuild matches {dest.name} -- not rewritten", flush=True)
        return dest

    write_xlsx(out, dest, sheet_name="shortlist")
    if case in FROZEN:
        print(f"[{case}] *** NEW FROZEN BASELINE *** wrote {dest.name}. This file is now the "
              f"contract for make_batch.py -- re-verify every PICK and `git add` it before "
              f"ordering anything.", flush=True)
    else:
        print(f"[{case}] -> {dest.name}", flush=True)
    return dest


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--verify-refs":
        verify_references()
    elif arg == "--all":
        for c in CASES:
            build(c)
    elif arg in CASES:
        build(arg)
    else:
        raise SystemExit(f"usage: make_shortlist_case.py <case> | --all | --verify-refs\n"
                         f"cases: {' '.join(CASES)}")
