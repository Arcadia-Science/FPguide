#!/usr/bin/env python
"""Known-structure cohort design: guide structure-backed scaffolds toward ~80%-identity
O_train targets, using the 5 A pocket read off each scaffold's EXPERIMENTAL RCSB
structure (no ESMFold). Same batched, CUDA-fp16, dual-model guided-refinement as
``parallel_design_normal.ipynb`` (surrogate ESM-2 cnn-max-d1 guides on (ex,em); oracle
ProstT5 cnn-max-d2 judges), settings 3 iters / T=5 / k=10 / lam=20.

Cohorts (from select_knownstruct.py):
  * knownstruct_Strain_Otrain - scaffold in surrogate TRAIN split
  * knownstruct_Stest_Otrain  - scaffold in surrogate TEST  split (generalization)

Windows come from pockets.experimental_window (auto chromophore + catalytic). All tasks
across the requested cohorts advance the same position-slot together (one big GPU batch
per masked-LM / surrogate step). Resumable: one CSV per task, existing ones skipped.

Usage
-----
    python design_knownstruct.py                       # both cohorts, all 40 tasks
    python design_knownstruct.py --cohorts knownstruct_Stest_Otrain
    python design_knownstruct.py --smoke 2             # first 2 tasks only (wiring test)
"""
import argparse
import csv
import glob
import os
import sys
import time
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import common as C

sys.path.insert(0, str(C.PEAK_DIR))
import peak_models as pm
import prostt5_embed as pe
import pockets

# ---- design hyper-parameters (as requested) ----------------------------------------
LAM_EX = 20.0
LAM_EM = 20.0
K_TOP = 10
TEMP = 5.0
N_ITERS = 3
CUTOFF = 5.0
DEFAULT_COHORTS = ["knownstruct_Strain_Otrain", "knownstruct_Stest_Otrain"]

ESM_BS = 48
ORAC_BS = 32

COLS = ["example", "cohort", "phase", "lam_ex", "lam_em", "round", "n_editable",
        "chromo_pos1_1based", "scaffold_name", "scaffold_pdb", "scaffold_idx",
        "scaffold_ex", "scaffold_em", "target_name", "target_idx", "target_ex", "target_em",
        "seq_id_scaf_target", "pred_ex", "pred_em", "peak_err", "ppl", "ident_to_scaffold",
        "designed_seq", "scaffold_seq", "target_seq"]


def _san(s):
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="*", default=DEFAULT_COHORTS)
    ap.add_argument("--smoke", type=int, default=0, help="run only the first N tasks (wiring test)")
    args = ap.parse_args()

    dev = (torch.device("cuda") if torch.cuda.is_available()
           else torch.device("mps") if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
           else torch.device("cpu"))
    use_fp16 = (dev.type == "cuda")
    AMP = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_fp16 else (lambda: nullcontext())
    print(f"device {dev} | fp16 {use_fp16}")

    d = C.load_dataset()
    rows, seqs, N = d["rows"], d["seqs"], d["N"]
    EXM, EMM, peaks = d["EXM"], d["EMM"], d["peaks"]

    # ---- tasks from the requested cohorts ------------------------------------------
    tasks = []
    for coh in args.cohorts:
        fn = C.pairs_csv_path(coh)
        if not os.path.exists(fn):
            raise SystemExit(f"missing manifest {fn}; run curate_knownstruct.py + select_knownstruct.py first")
        outdir = os.path.join(C.PIPE_OUT, coh)
        os.makedirs(outdir, exist_ok=True)
        for r in csv.DictReader(open(fn)):
            tasks.append(dict(cohort=coh, outdir=outdir,
                              si=int(r["scaffold_idx"]), ti=int(r["target_idx"]),
                              idv=float(r["identity"]), pdb=r["scaffold_pdb"]))
    if args.smoke:
        tasks = tasks[:args.smoke]
    print(f"{len(tasks)} tasks across {len(args.cohorts)} cohort(s): "
          + ", ".join(f"{c}={sum(1 for t in tasks if t['cohort']==c)}" for c in args.cohorts))

    # ---- experimental 5 A windows (fetch cached in structures/experimental) --------
    print("building experimental windows ...")
    WIN = {}
    t0 = time.time()
    for t in tasks:
        si = t["si"]
        if si in WIN:
            continue
        nm = rows[si]["name"]
        c1, catal, pocket = pockets.experimental_window(nm, seqs[si], t["pdb"], cutoff=CUTOFF)
        WIN[si] = dict(c1=c1, p2=c1 + 1, p3=c1 + 2, catal=catal, pocket=pocket)
    ns = [len(WIN[si]["pocket"]) for si in WIN]
    print(f"  {len(WIN)} windows in {time.time()-t0:.0f}s | pocket size min/med/max = "
          f"{min(ns)}/{int(np.median(ns))}/{max(ns)}")

    # ---- models + batched helpers (recovered from parallel_design_normal.ipynb) ----
    _sb, s_meta = pm.load_model(C.SURR_CKPT, dev); surrogate_net = pm.wrap(_sb, s_meta["mean"], s_meta["std"], dev)
    _ob, o_meta = pm.load_model(C.ORAC_CKPT, dev); oracle_net = pm.wrap(_ob, o_meta["mean"], o_meta["std"], dev)
    print(f"surrogate val MAE {s_meta['val_mae']:.1f} nm | oracle val MAE {o_meta['val_mae']:.1f} nm")
    esm_model, alphabet, bc = pm.get_esm(dev)

    AA_MASK = torch.zeros(len(alphabet.all_toks), dtype=torch.bool)
    for a in "ACDEFGHIKLMNPQRSTVWY":
        AA_MASK[alphabet.get_idx(a)] = True
    AA_MASK = AA_MASK.to(dev)
    SPECIAL = {alphabet.cls_idx, alphabet.eos_idx, alphabet.padding_idx, alphabet.mask_idx}
    AROMATIC = torch.zeros(len(alphabet.all_toks), dtype=torch.bool)
    for a in "YWHF":
        AROMATIC[alphabet.get_idx(a)] = True
    AROMATIC = AROMATIC.to(dev)

    @torch.no_grad()
    def _esm_embed(seqlist):
        _, _, tk = bc([(f"s{i}", s) for i, s in enumerate(seqlist)]); tk = tk.to(dev)
        with AMP():
            reps = esm_model(tk, repr_layers=[pm.ESM_LAYER])["representations"][pm.ESM_LAYER]
        Lmax = max(len(s) for s in seqlist)
        H = torch.zeros(len(seqlist), Lmax, pm.D_IN, device=dev)
        mask = torch.zeros(len(seqlist), Lmax, dtype=torch.bool, device=dev)
        for j, s in enumerate(seqlist):
            n = len(s); H[j, :n] = reps[j, 1:1 + n].float(); mask[j, :n] = True
        return H, mask

    @torch.no_grad()
    def surrogate_peaks_batched(seqlist, bs=ESM_BS):
        outs = []
        for i in range(0, len(seqlist), bs):
            H, mask = _esm_embed(seqlist[i:i + bs]); outs.append(surrogate_net(H, mask))
        return torch.cat(outs, 0)

    @torch.no_grad()
    def oracle_peaks_batched(seqlist, bs=ORAC_BS):
        outs = []
        for i in range(0, len(seqlist), bs):
            ch = seqlist[i:i + bs]; H, mask = pe.resid_embed_prostt5(ch, dev, bs=len(ch)); outs.append(oracle_net(H, mask))
        return torch.cat(outs, 0)

    @torch.no_grad()
    def esm_logits_at(seqlist, positions):
        _, _, tk = bc([(f"s{i}", s) for i, s in enumerate(seqlist)]); tk = tk.to(dev)
        rr = torch.arange(len(seqlist), device=dev)
        cols = torch.tensor([p + 1 for p in positions], device=dev)
        tk[rr, cols] = alphabet.mask_idx
        with AMP():
            lg = esm_model(tk)["logits"]
        return lg[rr, cols].float()

    @torch.no_grad()
    def ppl_batched(seqlist, bs=96):
        jobs = []
        for s in seqlist:
            _, _, base = bc([("s", s.strip().upper())]); base = base[0]
            respos = [i for i, tok in enumerate(base.tolist()) if tok not in SPECIAL]
            jobs.append((base, respos))
        rows_meta = [(si, base, p) for si, (base, respos) in enumerate(jobs) for p in respos]
        tot = torch.zeros(len(seqlist), device=dev)
        for b in range(0, len(rows_meta), bs):
            chunk = rows_meta[b:b + bs]
            Lmax = max(base.numel() for _, base, _ in chunk)
            bt = torch.full((len(chunk), Lmax), alphabet.padding_idx, dtype=torch.long)
            cols = torch.zeros(len(chunk), dtype=torch.long)
            truth = torch.zeros(len(chunk), dtype=torch.long)
            sidx = torch.zeros(len(chunk), dtype=torch.long)
            for r, (si, base, p) in enumerate(chunk):
                L = base.numel(); bt[r, :L] = base; bt[r, p] = alphabet.mask_idx
                cols[r] = p; truth[r] = base[p]; sidx[r] = si
            bt = bt.to(dev)
            with AMP():
                lp = torch.log_softmax(esm_model(bt)["logits"], -1)
            rr = torch.arange(len(chunk), device=dev)
            vals = lp[rr, cols.to(dev), truth.to(dev)].float()
            tot.index_add_(0, sidx.to(dev), vals)
        nsn = torch.tensor([len(rp) for _, rp in jobs], device=dev, dtype=torch.float).clamp(min=1)
        return torch.exp(-tot / nsn).cpu().tolist()

    def _zc(t):
        return (t - t.mean()) / (t.std() + 1e-6)

    def resolve_positions(chromo_nums, use_pocket, w):
        sel = [{1: w["c1"], 2: w["p2"], 3: w["p3"]}[n] for n in sorted(chromo_nums)]
        if use_pocket:
            sel += w["pocket"]
        return sel

    PHASE_PLAN = [("chromo+pocket", [({1, 2}, False), (set(), True)], N_ITERS)]

    # ---- resumable: drop tasks whose CSV already exists -----------------------------
    todo = []
    for t in tasks:
        name = f"{rows[t['si']]['name']}-{rows[t['ti']]['name']}"
        fn = os.path.join(t["outdir"], f"design_{_san(name)}.csv")
        if os.path.exists(fn):
            print(f"[{name}] cached -> skip"); continue
        t["name"] = name; t["fn"] = fn
        todo.append(t)
    if not todo:
        print("all tasks cached; nothing to run")
        return

    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    T = []
    for t in todo:
        si = t["si"]; w = WIN[si]
        T.append(dict(**t, w=w, scaffold=seqs[si], seq=seqs[si],
                      tgt=torch.tensor(peaks[t["ti"]], device=dev),
                      pos_allowed={w["p2"]: AROMATIC}, c1=w["c1"], hist=[]))
    print(f"running {len(T)} tasks | {N_ITERS} iters x (chromo -> pocket) | k={K_TOP} T={TEMP} "
          f"lam_ex={LAM_EX} lam_em={LAM_EM}")

    def evaluate_round(phase, r, touched):
        seqlist = [t["seq"] for t in T]
        P = oracle_peaks_batched(seqlist)
        ppls = ppl_batched(seqlist)
        for i, t in enumerate(T):
            ex, em = float(P[i, 0]), float(P[i, 1])
            err = 0.5 * (abs(ex - float(t["tgt"][0])) + abs(em - float(t["tgt"][1])))
            ident = sum(x == y for x, y in zip(t["seq"], t["scaffold"])) / len(t["scaffold"])
            t["hist"].append(dict(phase=phase, round=r, nedit=touched.get(id(t), 0),
                                  pred_ex=ex, pred_em=em, peak_err=err, ppl=ppls[i], ident=ident, seq=t["seq"]))

    t0 = time.time()
    evaluate_round("scaffold", 0, {})
    print(f"  round 0 (scaffold): mean oracle err {np.mean([t['hist'][-1]['peak_err'] for t in T]):.1f} nm "
          f"| {time.time()-t0:.0f}s")
    r = 0
    for lab, steps, n in PHASE_PLAN:
        for _ in range(n):
            r += 1; touched = {id(t): set() for t in T}
            for nums, use_pocket in steps:
                for t in T:
                    t["_ed"] = resolve_positions(nums, use_pocket, t["w"])
                maxlen = max((len(t["_ed"]) for t in T), default=0)
                for j in range(maxlen):
                    sub = [t for t in T if j < len(t["_ed"])]
                    positions = [t["_ed"][j] for t in sub]
                    logits = esm_logits_at([t["seq"] for t in sub], positions)
                    cand_all = []; meta = []
                    for i, t in enumerate(sub):
                        pos = positions[i]
                        mh = t["pos_allowed"].get(pos, AA_MASK)
                        lg = logits[i].masked_fill(~mh, float("-inf"))
                        logp = torch.log_softmax(lg, -1)
                        k_eff = min(K_TOP, int(mh.sum().item()))
                        topv, topi = torch.topk(logp, k_eff)
                        aas = [alphabet.get_tok(int(x)) for x in topi.tolist()]
                        for aa in aas:
                            cand_all.append(t["seq"][:pos] + aa + t["seq"][pos + 1:])
                        meta.append((t, pos, topv, aas, k_eff))
                    Pk = surrogate_peaks_batched(cand_all)
                    off = 0
                    for (t, pos, topv, aas, k_eff) in meta:
                        Pc = Pk[off:off + k_eff]; off += k_eff
                        ex_err = (Pc[:, 0] - t["tgt"][0]).abs(); em_err = (Pc[:, 1] - t["tgt"][1]).abs()
                        scores = _zc(topv) - LAM_EX * _zc(ex_err) - LAM_EM * _zc(em_err)
                        ch = (int(torch.multinomial(torch.softmax(scores / TEMP, -1), 1).item())
                              if TEMP > 0 else int(torch.argmax(scores)))
                        t["seq"] = t["seq"][:pos] + aas[ch] + t["seq"][pos + 1:]
                        touched[id(t)].add(pos)
            touched_ct = {k: len(v) for k, v in touched.items()}
            evaluate_round(lab, r, touched_ct)
            print(f"  it{r} [{lab}]: mean oracle err {np.mean([t['hist'][-1]['peak_err'] for t in T]):.1f} nm "
                  f"| mean id {np.mean([t['hist'][-1]['ident'] for t in T]):.2f} | {time.time()-t0:.0f}s", flush=True)

    for t in T:
        si, ti = t["si"], t["ti"]
        with open(t["fn"], "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(COLS)
            for hh in t["hist"]:
                w.writerow([t["name"], t["cohort"], hh["phase"], LAM_EX, LAM_EM, hh["round"], hh["nedit"],
                            t["c1"] + 1, rows[si]["name"], t["pdb"], si, f"{EXM[si]:.0f}", f"{EMM[si]:.0f}",
                            rows[ti]["name"], ti, f"{EXM[ti]:.0f}", f"{EMM[ti]:.0f}", f"{t['idv']:.3f}",
                            f"{hh['pred_ex']:.1f}", f"{hh['pred_em']:.1f}", f"{hh['peak_err']:.2f}", f"{hh['ppl']:.2f}",
                            f"{hh['ident']:.3f}", hh["seq"], t["scaffold"], seqs[ti]])
    print(f"wrote {len(T)} design CSVs | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
