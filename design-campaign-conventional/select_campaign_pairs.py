#!/usr/bin/env python
"""Select 24 scaffold->target FP pairs for the conventional design campaign.

Constraints (as specified):
  * scaffold has an ACTUAL experimental structure (RCSB experimental hit >=97%);
  * target is 70-90% sequence-identical to the scaffold;
  * BOTH scaffold and target have Stokes shift (em_max - ex_max) < 80 nm.

STRUCTURE VALIDATION: a scaffold is only accepted if its experimental structure yields a
clean design window -- i.e. pockets.experimental_window succeeds (a deposited chain matches
the dataset sequence at >=90% local identity over >=70% coverage). The cached top RCSB hit
(structure_hits.csv) is tried first; if it fails the gate we query further experimental hits
and keep the first PDB that passes. This avoids scaffolds whose "hit" is a split-FP biosensor,
fusion, or partial model that can't be mapped to a real chromophore pocket.

No train/val/test split is used (the surrogate was trained on all data), so targets are drawn
from the WHOLE curated set (any FP). Per scaffold the target is the in-band partner closest to
0.80 identity. A set of named scaffolds is force-INCLUDED (default: mTagBFP2, EGFP, mVenus,
mCherry, mKate2) and the rest are filled with RANDOM structure-known scaffolds (seeded).

Output: pairs/campaign_pairs_24.csv  (scaffold_idx, scaffold_name, scaffold_SS, scaffold_pdb,
target_idx, target_name, target_SS, identity, selection[forced|random]).
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import biotite.sequence as bseq
import biotite.sequence.align as balign

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CUR = REPO / "dataset_pipeline" / "data" / "peak" / "curated"
STRUCT_HITS = REPO / "peak_design" / "peak_designs" / "structure" / "parallel_pipeline" / "pairs" / "structure_hits.csv"
STRUCTDIR = str(REPO / "peak_design" / "structures" / "experimental")
sys.path.insert(0, str(REPO / "peak_design"))
import pockets                     # noqa: E402

SS_MAX = 80.0
ID_LO, ID_HI, ID_TARGET = 0.70, 0.90, 0.80
STRUCT_ID = 0.97                   # RCSB experimental identity cutoff
CHROMO_AA2 = set("YWHF")
_MAT = balign.SubstitutionMatrix.std_protein_matrix()
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def has_chromo(seq, lo=50, hi=85):
    return any(seq[i + 1] in CHROMO_AA2 and seq[i + 2] == "G" and lo <= i <= hi
               for i in range(len(seq) - 2))


def seq_identity(a, b):
    aln = balign.align_optimal(bseq.ProteinSequence(a), bseq.ProteinSequence(b),
                               _MAT, gap_penalty=(-10, -1))[0]
    return float(balign.get_sequence_identity(aln))


def load_dataset():
    rows = list(csv.DictReader(open(CUR / "peaks_assignments.csv")))
    N = len(rows)
    peaks = np.load(CUR / "peaks.npy").astype(np.float32)
    seqs = [None] * N
    h = None
    for line in open(CUR / "sequences.fasta"):
        line = line.strip()
        if line.startswith(">"):
            h = int(line[1:].split("|")[0])
        elif line:
            seqs[h] = line
    return rows, seqs, peaks[:, 1] - peaks[:, 0], N


def rcsb_experimental_pdbs(seq, identity=STRUCT_ID, evalue=0.1, rows_ret=15, retries=3):
    """Ordered list of distinct experimental PDB ids >= identity (best first)."""
    q = {
        "query": {"type": "terminal", "service": "sequence",
                  "parameters": {"evalue_cutoff": evalue, "identity_cutoff": identity,
                                 "sequence_type": "protein", "value": seq}},
        "request_options": {"results_content_type": ["experimental"],
                            "paginate": {"start": 0, "rows": rows_ret}},
        "return_type": "polymer_entity",
    }
    data = json.dumps(q).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(SEARCH_URL, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status == 204:
                    return []
                res = json.load(r)
            out = []
            for hit in res.get("result_set", []):
                pdb = hit["identifier"].partition("_")[0]
                if pdb not in out:
                    out.append(pdb)
            return out
        except urllib.error.HTTPError as e:
            if e.code == 204:
                return []
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return []


def resolve_pdb(name, seq, cached_pdb):
    """Return (pdb, quality) for the first structure yielding a gate-passing window,
    trying the cached hit first then further RCSB experimental hits. (None, None) if none."""
    order = []
    if cached_pdb:
        order.append(cached_pdb)
    for pdb in rcsb_experimental_pdbs(seq):
        if pdb not in order:
            order.append(pdb)
    for pdb in order:
        try:
            _, _, _, q = pockets.experimental_window(name, seq, pdb, return_quality=True, structdir=STRUCTDIR)
            return pdb, q
        except Exception:
            continue
    return None, None


DEFAULT_INCLUDE = ["mTagBFP2", "EGFP", "mVenus", "mCherry", "mKate2"]


def best_target(si, seqs, SS, N, len_tol):
    """In-band target for scaffold si: any SS<80 FP with 70-90% identity, closest to 0.80."""
    Ls = len(seqs[si]); best = None
    for tj in range(N):
        if tj == si or SS[tj] >= SS_MAX or abs(len(seqs[tj]) - Ls) > len_tol:
            continue
        idv = seq_identity(seqs[si], seqs[tj])
        if ID_LO <= idv <= ID_HI and (best is None or abs(idv - ID_TARGET) < abs(best[1] - ID_TARGET)):
            best = (tj, idv)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--include", nargs="*", default=DEFAULT_INCLUDE,
                    help="scaffold names to force-include; the rest are filled at random")
    ap.add_argument("--len-tol", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "pairs" / "campaign_pairs_24.csv"))
    args = ap.parse_args()

    rows, seqs, SS, N = load_dataset()
    name2idx = {r["name"]: i for i, r in enumerate(rows)}
    if not STRUCT_HITS.exists():
        raise SystemExit(f"missing cached structure search: {STRUCT_HITS}\n"
                         "run peak_design/parallel_pipeline/curate_knownstruct.py first")
    pdb_of = {int(r["idx"]): r["pdb_id"] for r in csv.DictReader(open(STRUCT_HITS)) if r["pdb_id"]}
    print(f"N={N} | structure-known scaffolds (cached RCSB >=97% experimental): {len(pdb_of)}")

    picks = []               # (scaf_i, tgt_j, id, selection, pdb, quality)
    used_scaf = set()

    # ---- forced scaffolds (must resolve to a gate-passing structure + valid target) ----
    t0 = time.time()
    for nm in args.include:
        if nm not in name2idx:
            raise SystemExit(f"requested scaffold '{nm}' not in dataset")
        si = name2idx[nm]
        if SS[si] >= SS_MAX:
            raise SystemExit(f"requested scaffold '{nm}' has SS={SS[si]:.0f} >= {SS_MAX:.0f}")
        pdb, q = resolve_pdb(nm, seqs[si], pdb_of.get(si))
        if pdb is None:
            raise SystemExit(f"requested scaffold '{nm}' has no experimental structure that maps "
                             f"cleanly to a chromophore pocket (>=90% id / >=70% cov)")
        bt = best_target(si, seqs, SS, N, args.len_tol)
        if bt is None:
            raise SystemExit(f"requested scaffold '{nm}' has no 70-90% SS<80 target")
        picks.append((si, bt[0], bt[1], "forced", pdb, q)); used_scaf.add(si)
        print(f"  forced {nm:20} [{pdb}] chain {q['chain']} id {q['local_id']:.0%}/cov {q['coverage']:.0%}", flush=True)
    print(f"forced {len(picks)} scaffolds resolved | {time.time()-t0:.0f}s")

    # ---- random fill (validated structure + valid target) --------------------------
    rng = np.random.default_rng(args.seed)
    rest = [i for i in pdb_of if i not in used_scaf and SS[i] < SS_MAX and has_chromo(seqs[i])]
    rng.shuffle(rest)
    need = args.n - len(picks)
    for si in rest:
        if need <= 0:
            break
        bt = best_target(si, seqs, SS, N, args.len_tol)
        if bt is None:
            continue
        pdb, q = resolve_pdb(rows[si]["name"], seqs[si], pdb_of.get(si))
        if pdb is None:
            continue
        picks.append((si, bt[0], bt[1], "random", pdb, q)); used_scaf.add(si); need -= 1
        print(f"  random {rows[si]['name']:20} [{pdb}] id {q['local_id']:.0%}/cov {q['coverage']:.0%}", flush=True)
    print(f"filled {sum(1 for p in picks if p[3]=='random')} random scaffolds | total scan {time.time()-t0:.0f}s")
    if len(picks) < args.n:
        print(f"WARNING: only {len(picks)}/{args.n} scaffolds could be validated")

    # ---- write --------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scaffold_idx", "scaffold_name", "scaffold_SS", "scaffold_pdb",
                    "target_idx", "target_name", "target_SS", "identity", "selection"])
        for si, tj, idv, sel, pdb, q in picks:
            w.writerow([si, rows[si]["name"], f"{SS[si]:.0f}", pdb,
                        tj, rows[tj]["name"], f"{SS[tj]:.0f}", f"{idv:.4f}", sel])
    ids = np.array([p[2] for p in picks])
    print(f"\nselected {len(picks)} pairs | identity min {ids.min():.0%} med {np.median(ids):.0%} max {ids.max():.0%} "
          f"-> {args.out}")
    print(f"{'sel':7}{'scaffold':24}{'PDB':6}{'SS':>4}  ->  {'target':24}{'SS':>4}{'id':>6}")
    for si, tj, idv, sel, pdb, q in picks:
        print(f"{sel:7}{rows[si]['name'][:23]:24}{pdb:6}{SS[si]:>4.0f}  ->  "
              f"{rows[tj]['name'][:23]:24}{SS[tj]:>4.0f}{idv:>6.0%}")
    print("\nselect_campaign_pairs done.")


if __name__ == "__main__":
    main()
