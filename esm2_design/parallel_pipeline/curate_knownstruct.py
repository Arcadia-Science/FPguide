#!/usr/bin/env python
"""Curate the KNOWN-STRUCTURE cohort: scaffolds = dataset FPs that have an
experimental PDB structure (RCSB sequence search >= identity cutoff), targets =
O_train FPs at ~80% identity. Keeps SS<80 on scaffold and target.

Two artifacts (both cached / resumable) under
``peak_designs/structure/parallel_pipeline/pairs/``:

  * ``structure_hits.csv``  - one row per queried candidate scaffold:
        idx,name,SS,surr_role,orac_role,queried,pdb_id,pdb_entity
    (pdb_id empty => no experimental structure >= cutoff). RCSB is only queried
    for rows not already present, so re-running resumes.
  * ``pairs_knownstruct_Otrain.csv`` - the final scaffold->target manifest
    (identical schema to the other cohorts + a ``scaffold_pdb`` column), written
    for ALL available pairs; cap to N at design time.

Usage
-----
    python curate_knownstruct.py                      # >=97%, target band [0.70,0.90]
    python curate_knownstruct.py --struct-id 0.97 --id-lo 0.70 --id-hi 0.90
    python curate_knownstruct.py --report-only        # just print counts, no re-query
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ensure sibling modules importable

import biotite.sequence as bseq
import biotite.sequence.align as balign

import common as C

_MAT = balign.SubstitutionMatrix.std_protein_matrix()
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
HITS_CSV = os.path.join(C.PAIRS_DIR, "structure_hits.csv")
KNOWN_COHORT = "knownstruct_Otrain"


def seq_identity(a, b):
    aln = balign.align_optimal(bseq.ProteinSequence(a), bseq.ProteinSequence(b),
                               _MAT, gap_penalty=(-10, -1))[0]
    return float(balign.get_sequence_identity(aln))


def rcsb_top_experimental(seq, identity=0.97, evalue=0.1, retries=3):
    """Return (pdb_id, entity) of the top EXPERIMENTAL structure >= identity, or (None, None)."""
    q = {
        "query": {"type": "terminal", "service": "sequence",
                  "parameters": {"evalue_cutoff": evalue, "identity_cutoff": identity,
                                 "sequence_type": "protein", "value": seq}},
        "request_options": {"return_all_hits": False,
                            "results_content_type": ["experimental"],
                            "paginate": {"start": 0, "rows": 1}},
        "return_type": "polymer_entity",
    }
    data = json.dumps(q).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(SEARCH_URL, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status == 204:
                    return None, None
                res = json.load(r)
            rs = res.get("result_set", [])
            if not rs:
                return None, None
            ident = rs[0]["identifier"]           # e.g. "9DZE_3"
            pdb, _, ent = ident.partition("_")
            return pdb, ent
        except urllib.error.HTTPError as e:
            if e.code == 204:
                return None, None
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return None, None


def load_hits():
    if not os.path.exists(HITS_CSV):
        return {}
    out = {}
    for r in csv.DictReader(open(HITS_CSV)):
        out[int(r["idx"])] = r
    return out


def save_hits(hits):
    os.makedirs(C.PAIRS_DIR, exist_ok=True)
    cols = ["idx", "name", "SS", "surr_role", "orac_role", "queried", "pdb_id", "pdb_entity"]
    with open(HITS_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for idx in sorted(hits):
            w.writerow(hits[idx])


def main():
    ap = argparse.ArgumentParser(description="Curate the known-structure cohort.")
    ap.add_argument("--struct-id", type=float, default=0.97, help="RCSB experimental identity cutoff")
    ap.add_argument("--id-lo", type=float, default=0.70)
    ap.add_argument("--id-hi", type=float, default=0.90)
    ap.add_argument("--id-target", type=float, default=0.80)
    ap.add_argument("--ss-max", type=float, default=C.SS_MAX)
    ap.add_argument("--len-tol", type=int, default=30)
    ap.add_argument("--report-only", action="store_true", help="no RCSB queries; use cache only")
    args = ap.parse_args()

    data = C.load_dataset()
    rows, seqs, SS = data["rows"], data["seqs"], data["SS"]
    Srole, Orole, Nn = data["Srole"], data["Orole"], data["N"]

    # candidate scaffolds: SS<80 + chromophore motif (any split)
    cand = [i for i in range(Nn) if SS[i] < args.ss_max and C.has_chromo(seqs[i])]
    print(f"candidate scaffolds (SS<{args.ss_max:.0f} + chromophore motif): {len(cand)}")

    hits = load_hits()
    to_query = [i for i in cand if i not in hits]
    if args.report_only:
        print(f"report-only: using {len(hits)} cached hits ({len(to_query)} uncached, skipped)")
    else:
        print(f"RCSB experimental search >= {args.struct_id:.0%} for {len(to_query)} uncached candidates "
              f"({len(hits)} cached) ...")
        t0 = time.time()
        for c, i in enumerate(to_query):
            pdb, ent = rcsb_top_experimental(seqs[i], identity=args.struct_id)
            hits[i] = dict(idx=i, name=rows[i]["name"], SS=f"{SS[i]:.0f}",
                           surr_role=Srole[i], orac_role=Orole[i], queried=1,
                           pdb_id=(pdb or ""), pdb_entity=(ent or ""))
            if (c + 1) % 25 == 0:
                save_hits(hits)
                nk = sum(1 for h in hits.values() if h["pdb_id"])
                print(f"  queried {c+1}/{len(to_query)} | structure-known so far {nk} | {time.time()-t0:.0f}s",
                      flush=True)
            time.sleep(0.12)
        save_hits(hits)
        print(f"RCSB search done in {time.time()-t0:.0f}s -> {HITS_CSV}")

    # structure-known scaffolds among candidates
    known = [i for i in cand if i in hits and hits[i]["pdb_id"]]
    print(f"\nstructure-known scaffolds (>= {args.struct_id:.0%} experimental): {len(known)} / {len(cand)} candidates")

    # target pool: O_train, SS<80, chromophore motif
    pool = [i for i in range(Nn) if SS[i] < args.ss_max and Orole[i] == "train" and C.has_chromo(seqs[i])]
    print(f"target pool (O_train, SS<{args.ss_max:.0f}, motif): {len(pool)}")

    # pair each structure-known scaffold with its closest-to-id_target O_train partner in band
    pairs = []
    t0 = time.time()
    for si in known:
        Ls = len(seqs[si])
        best = None
        for ti in pool:
            if ti == si or abs(len(seqs[ti]) - Ls) > args.len_tol:
                continue
            idv = seq_identity(seqs[si], seqs[ti])
            if args.id_lo <= idv <= args.id_hi and (best is None or abs(idv - args.id_target) < abs(best[1] - args.id_target)):
                best = (ti, idv)
        if best is not None:
            pairs.append((si, best[0], best[1]))
    print(f"structure-known scaffolds WITH a valid target: {len(pairs)} (pairing scan {time.time()-t0:.0f}s)")

    # write full manifest (all available pairs)
    os.makedirs(C.PAIRS_DIR, exist_ok=True)
    fn = C.pairs_csv_path(KNOWN_COHORT)
    with open(fn, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(C.PAIRS_COLS + ["scaffold_pdb"])
        for si, ti, idv in sorted(pairs, key=lambda p: -p[2]):
            w.writerow([si, rows[si]["name"], f"{SS[si]:.0f}", Srole[si],
                        ti, rows[ti]["name"], f"{SS[ti]:.0f}", Orole[ti], f"{idv:.4f}",
                        hits[si]["pdb_id"]])
    print(f"wrote {len(pairs)} available pairs -> {fn}")

    # quick distributions
    if pairs:
        import numpy as np
        idv = np.array([p[2] for p in pairs])
        sroles = [Srole[p[0]] for p in pairs]
        print(f"\ntarget identity: min {idv.min():.0%} med {np.median(idv):.0%} max {idv.max():.0%}")
        print("scaffold surrogate-role split:",
              {b: sum(1 for s in sroles if s == b) for b in ("train", "val", "test")})
        print("\nexamples:")
        for si, ti, v in sorted(pairs, key=lambda p: abs(p[2] - args.id_target))[:10]:
            print(f"  {rows[si]['name']:22}[{hits[si]['pdb_id']}] (SS{SS[si]:.0f},S:{Srole[si]}) -> "
                  f"{rows[ti]['name']:22}(SS{SS[ti]:.0f},O:train) id {v:.0%}")
    print("\ncurate_knownstruct done.")


if __name__ == "__main__":
    main()
