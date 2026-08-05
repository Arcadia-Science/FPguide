#!/usr/bin/env python
"""Known-structure cohort design, Tier-B window + MSA-guided proposal, scored by THIS
experiment's own CV-selected models (oracle-scored).

Tier-B window + family-PSSM guided proposal, scored by this experiment's own models. The
window rule, PSSM-guided proposal and guided-score formula are the same design the Tier-B +
MSA pipeline established; what is specific to this experiment is its inputs:

  SURROGATE: ``cnn-max-d1``, refit by ``train_final_surrogate.py`` on this experiment's own
  nested-split surrogate train+val (515 rows) -- the architecture the 3-fold CV
  (``cv_all_surrogate.py``, see sweep_results.ipynb) confirmed as the best of the 48-config
  sweep, not just the single-split sweep's (different) pick.

  ORACLE: ``cnn-max-d1``, this experiment's own oracle-sweep winner, trained on the 80/10/10
  oracle split (single split, not CV'd -- CV here was scoped to the surrogate only).

  TASKS: ``pairs/`` from ``curate_pairs.py`` -- cohorts labeled by this experiment's own nested
  split and selected for scaffold->target SPECTRAL DISTANCE, so each task genuinely asks the
  search to move somewhere new in ex/em space. All three role cohorts are run; note that they
  form only TWO reporting conditions, since the deployed surrogate was refit on train+val
  (``design_common.COHORT_CONDITION``: S-train + S-val = ``seen``, S-test = ``held-out``).
  That grouping is applied in analysis, so it never affects what this script runs.

  WINDOWS: ``design_windows.json`` from ``build_windows.py``, built from scratch against this
  folder's own structure cache and family alignment.

TRIALS AND VISITING ORDER. Each task is run ``--trials`` times independently (default 3). A trial
is one full search from the scaffold, differing from its siblings only in its random stream, so
the spread across trials is the search's own variance on that task -- the pipeline is stochastic
(top-k multinomial at T=1), and a single trial cannot tell a good task from a lucky draw. Within
a cycle the editable positions are visited in a RANDOM order (a fresh permutation per trial per
cycle), matching ``design-campaign-EGFP``: selection is sequential and greedy-ish, so a fixed
N->C walk always picks early positions against an unedited C-terminal tail and biases which
substitutions look good. The permutation is drawn inside each phase step, so the phase plan's
chromophore-before-pocket structure is preserved -- it is the order WITHIN a step that is
randomized. ``--visit-order sequence`` restores the old fixed walk.

Each trial gets its own seeded RNG pair (numpy for the permutation, torch for the multinomial),
derived from (SEED, trial, scaffold_idx, target_idx), so a trial reproduces independently of how
many other tasks or trials ran alongside it.

Usage
-----
    python design_knownstruct.py                        # all cohorts, TRIALS x N_ITERS cycles
    python design_knownstruct.py --cohorts knownstruct_Stest
    python design_knownstruct.py --trials 1 --visit-order sequence   # the original first pass
    python design_knownstruct.py --iters 3              # more design cycles per task
    python design_knownstruct.py --smoke 2              # first 2 tasks only (wiring/timing probe)

Writes one CSV per task, all trials in it, keyed by the ``trial`` column. Resumable: a task whose
CSV already covers the requested --trials and --iters is skipped; one that does not (including
any written by an earlier config) is recomputed and overwritten.
"""
import argparse
import csv
import json
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

# --- stage-folder bootstrap: put the experiment root (design_common), lib/ (vendored
# --- modules) and msa/ (family alignment code) on the import path.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, "lib"), _os.path.join(_ROOT, "msa")]

import design_common as C

import peak_models as pm      # local copy -- this folder is self-contained
import prostt5_embed as pe

# ---- design hyper-parameters (msa-guided z-scored scale, matching design-campaign-EGFP) -----
LAM_EX = 1.0
LAM_EM = 1.0
K_TOP = 10
TEMP = 1.0
N_ITERS = 2          # design cycles per task; override with --iters
N_TRIALS = 3         # independent searches per task; override with --trials
ESM_BS = 48
ORAC_BS = 32

COLS = ["example", "trial", "cohort", "phase", "lam_ex", "lam_em", "round", "n_editable",
        "chromo_pos1_1based", "scaffold_name", "scaffold_pdb", "scaffold_idx",
        "scaffold_ex", "scaffold_em", "target_name", "target_idx", "target_ex", "target_em",
        "seq_id_scaf_target", "pred_ex", "pred_em", "peak_err", "ppl", "fam_logp",
        "ident_to_scaffold", "designed_seq", "scaffold_seq", "target_seq"]


def _san(s):
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in s)


def _complete(fn, trials, iters):
    """Does this CSV already hold every requested trial, each run to `iters` cycles?

    Stricter than "the file exists", because the trial and cycle counts are configuration: a CSV
    from the single-trial / fixed-order first pass, or from a shorter --iters, must be recomputed
    rather than silently accepted as cached.
    """
    if not os.path.exists(fn):
        return False
    try:
        with open(fn, newline="") as fh:
            rr = list(csv.DictReader(fh))
    except Exception:
        return False
    if not rr or "trial" not in rr[0]:
        return False
    have = {(int(r["trial"]), int(r["round"])) for r in rr}
    return all((tr, rd) in have for tr in range(trials) for rd in range(iters + 1))


def _rng_pair(dev, trial, si, ti):
    """Independent (numpy, torch) streams for one trial, reproducible from its identity alone."""
    ss = np.random.SeedSequence([C.SEED, trial, si, ti])
    rng = np.random.default_rng(ss)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(rng.integers(2 ** 62)))
    return rng, gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="*", default=C.DEFAULT_COHORTS)
    ap.add_argument("--smoke", type=int, default=0, help="run only the first N tasks (wiring/timing probe)")
    ap.add_argument("--iters", type=int, default=N_ITERS, help=f"design cycles per task (default {N_ITERS})")
    ap.add_argument("--trials", type=int, default=N_TRIALS,
                    help=f"independent searches per task (default {N_TRIALS})")
    ap.add_argument("--visit-order", choices=["random", "sequence"], default="random",
                    help="order of editable positions within a phase step (default random)")
    ap.add_argument("--outdir", default=str(C.PIPE_OUT_R3),
                    help=f"output root (default {C.PIPE_OUT_R3.name})")
    args = ap.parse_args()

    if not os.path.exists(C.WINDOWS_JSON):
        raise SystemExit(f"missing {C.WINDOWS_JSON}; run build_windows.py first")
    WINJ = json.load(open(C.WINDOWS_JSON))["windows"]

    dev = (torch.device("cuda") if torch.cuda.is_available()
           else torch.device("mps") if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
           else torch.device("cpu"))
    use_fp16 = (dev.type == "cuda")
    AMP = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_fp16 else (lambda: nullcontext())
    print(f"device {dev} | fp16 {use_fp16}")

    d = C.load_dataset()
    rows, seqs, N = d["rows"], d["seqs"], d["N"]
    EXM, EMM, peaks = d["EXM"], d["EMM"], d["peaks"]

    # ---- tasks from the requested cohorts, restricted to scaffolds with a built window ----
    tasks, n_no_window = [], 0
    for coh in args.cohorts:
        fn = C.pairs_csv_path(coh)
        if not os.path.exists(fn):
            raise SystemExit(f"missing manifest {fn}; run curate_pairs.py first")
        outdir = os.path.join(args.outdir, coh)
        os.makedirs(outdir, exist_ok=True)
        for r in csv.DictReader(open(fn)):
            nm = rows[int(r["scaffold_idx"])]["name"]
            if nm not in WINJ:
                n_no_window += 1
                continue
            tasks.append(dict(cohort=coh, outdir=outdir,
                              si=int(r["scaffold_idx"]), ti=int(r["target_idx"]),
                              idv=float(r["identity"]), pdb=r["scaffold_pdb"]))
    if n_no_window:
        print(f"{n_no_window} task(s) skipped: scaffold has no Tier-B+MSA window "
              f"(failed structure quality gate or not in family alignment)")
    if args.smoke:
        tasks = tasks[:args.smoke]
    print(f"{len(tasks)} tasks across {len(args.cohorts)} cohort(s): "
          + ", ".join(f"{c}={sum(1 for t in tasks if t['cohort']==c)}" for c in args.cohorts))

    # ---- models: this experiment's CV-selected surrogate + oracle-sweep winner ----
    _sb, s_meta = pm.load_model(C.SURR_CKPT, dev); surrogate_net = pm.wrap(_sb, s_meta["mean"], s_meta["std"], dev)
    _ob, o_meta = pm.load_model(C.ORAC_CKPT, dev); oracle_net = pm.wrap(_ob, o_meta["mean"], o_meta["std"], dev)
    print(f"surrogate ({s_meta.get('role','?')}) test MAE {s_meta.get('test_mae', float('nan')):.1f} nm "
          f"(n_train={s_meta.get('n_train','?')}) | oracle val MAE {o_meta['val_mae']:.1f} nm")
    esm_model, alphabet, bc = pm.get_esm(dev)   # still needed: surrogate's ESM-2 embedding
    V = len(alphabet.all_toks)

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

    def fam_logp_batch(T_list):
        """Design's log-likelihood under its OWN scaffold's family PSSM. Pure function of the
        sequence + the (already-loaded) PSSM -- no model forward, replaces ESM-2 ppl."""
        out = []
        for t in T_list:
            lp = 0.0
            for p in t["editable"]:
                row = t["pssm_logp"][p]
                aa = t["seq"][p]
                lp += float(row.get(aa, row["_min"]))
            out.append(lp)
        return out

    def _zc(t):
        return (t - t.mean()) / (t.std() + 1e-6)

    PHASE_PLAN = [("chromo+pocket", [({1, 2}, False), (set(), True)], args.iters)]

    # ---- resumable: drop tasks whose CSV already covers --trials x --iters -----------
    todo = []
    for t in tasks:
        name = f"{rows[t['si']]['name']}-{rows[t['ti']]['name']}"
        fn = os.path.join(t["outdir"], f"design_{_san(name)}.csv")
        if _complete(fn, args.trials, args.iters):
            print(f"[{name}] cached ({args.trials} trials x {args.iters} iters) -> skip"); continue
        if os.path.exists(fn):
            print(f"[{name}] present but not {args.trials} trials x {args.iters} iters -> recompute")
        t["name"] = name; t["fn"] = fn
        todo.append(t)
    if not todo:
        print("all tasks cached; nothing to run")
        return

    # ---- per-task window + PSSM (from JSON; no pockets.py / RCSB call here), then one
    # ---- INSTANCE per (task, trial). The window tensors are read-only, so all trials of a
    # ---- task share them; only seq/hist/rng are per trial.
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    T = []
    for t in todo:
        si = t["si"]; nm = rows[si]["name"]; w = WINJ[nm]
        c1 = w["chromophore"]["pos1_0based"]; p2 = w["chromophore"]["pos2_0based"]
        editable = list(w["editable_0based"])
        pos_allowed, pssm_vec, pssm_logp = {}, {}, {}
        for p_str, ent in w["pssm"].items():
            p = int(p_str)
            mask = torch.zeros(V, dtype=torch.bool)
            vec = torch.full((V,), float("-inf"))
            row = {}
            for aa, pr in zip(ent["alphabet"], ent["probs"]):
                idx = alphabet.get_idx(aa)
                mask[idx] = True
                vec[idx] = float(np.log(max(pr, 1e-12)))
                row[aa] = vec[idx].item()
            pos_allowed[p] = mask.to(dev)
            pssm_vec[p] = vec.to(dev)
            row["_min"] = min(row.values())
            pssm_logp[p] = row
        tgt = torch.tensor(peaks[t["ti"]], device=dev)
        for trial in range(args.trials):
            rng, gen = _rng_pair(dev, trial, si, t["ti"])
            T.append(dict(**t, trial=trial, w=w, scaffold=seqs[si], seq=seqs[si], tgt=tgt,
                          editable=editable, pos_allowed=pos_allowed, pssm_vec=pssm_vec,
                          pssm_logp=pssm_logp, c1=c1, p2=p2, hist=[], rng=rng, gen=gen))
    ns = [len(t["editable"]) for t in T]
    print(f"running {len(todo)} tasks x {args.trials} trials = {len(T)} searches | {args.iters} "
          f"iters x (chromo -> pocket) | k={K_TOP} T={TEMP} lam_ex={LAM_EX} lam_em={LAM_EM} | "
          f"editable min/med/max = {min(ns)}/{int(np.median(ns))}/{max(ns)} (Tier-B) | "
          f"visit order = {args.visit_order} | proposal = family MSA PSSM "
          f"(ESM-2 masked-LM not called for selection)")

    def resolve_positions(chromo_nums, use_pocket, t):
        """Positions to visit in this phase step, in the order they will be visited.

        With --visit-order random the order is a fresh per-trial permutation drawn here, i.e.
        once per (trial, cycle, step); with `sequence` it is the fixed N->C walk the first pass
        used. Either way the step's position SET is unchanged.
        """
        sel = [{1: t["c1"], 2: t["p2"]}[n] for n in sorted(chromo_nums)]
        if use_pocket:
            sel += [p for p in t["editable"] if p not in (t["c1"], t["p2"])]
        if args.visit_order == "random" and len(sel) > 1:
            sel = [int(p) for p in t["rng"].permutation(sel)]
        return sel

    def evaluate_round(phase, r, touched):
        seqlist = [t["seq"] for t in T]
        P = oracle_peaks_batched(seqlist)
        fam = fam_logp_batch(T)
        for i, t in enumerate(T):
            ex, em = float(P[i, 0]), float(P[i, 1])
            err = 0.5 * (abs(ex - float(t["tgt"][0])) + abs(em - float(t["tgt"][1])))
            ident = sum(x == y for x, y in zip(t["seq"], t["scaffold"])) / len(t["scaffold"])
            t["hist"].append(dict(phase=phase, round=r, nedit=touched.get(id(t), 0),
                                  pred_ex=ex, pred_em=em, peak_err=err, fam_logp=fam[i],
                                  ident=ident, seq=t["seq"]))

    t0 = time.time()
    evaluate_round("scaffold", 0, {})
    print(f"  round 0 (scaffold): mean oracle err {np.mean([t['hist'][-1]['peak_err'] for t in T]):.1f} nm "
          f"| {time.time()-t0:.0f}s", flush=True)
    r = 0
    for lab, steps, n in PHASE_PLAN:
        for _ in range(n):
            r += 1; touched = {id(t): set() for t in T}
            for nums, use_pocket in steps:
                for t in T:
                    t["_ed"] = resolve_positions(nums, use_pocket, t)
                maxlen = max((len(t["_ed"]) for t in T), default=0)
                for j in range(maxlen):
                    sub = [t for t in T if j < len(t["_ed"])]
                    positions = [t["_ed"][j] for t in sub]
                    # PSSM logits: static per-position family log-frequency, no ESM-2 forward
                    logits = torch.stack([t["pssm_vec"][pos] for t, pos in zip(sub, positions)])
                    cand_all = []; meta = []
                    for i, t in enumerate(sub):
                        pos = positions[i]
                        mh = t["pos_allowed"][pos]
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
                        # each trial draws from its own generator, so a trial is reproducible
                        # regardless of what ran beside it in the batch
                        ch = (int(torch.multinomial(torch.softmax(scores / TEMP, -1), 1,
                                                    generator=t["gen"]).item())
                              if TEMP > 0 else int(torch.argmax(scores)))
                        t["seq"] = t["seq"][:pos] + aas[ch] + t["seq"][pos + 1:]
                        touched[id(t)].add(pos)
            touched_ct = {k: len(v) for k, v in touched.items()}
            evaluate_round(lab, r, touched_ct)
            print(f"  it{r} [{lab}]: mean oracle err {np.mean([t['hist'][-1]['peak_err'] for t in T]):.1f} nm "
                  f"| mean id {np.mean([t['hist'][-1]['ident'] for t in T]):.2f} | {time.time()-t0:.0f}s", flush=True)

    # ---- one CSV per task, every trial in it (keyed by the `trial` column) -----------
    by_task = {}
    for t in T:
        by_task.setdefault(t["fn"], []).append(t)
    for fn, insts in by_task.items():
        si, ti = insts[0]["si"], insts[0]["ti"]
        with open(fn, "w", newline="") as fh:
            wtr = csv.writer(fh); wtr.writerow(COLS)
            for t in sorted(insts, key=lambda x: x["trial"]):
                for hh in t["hist"]:
                    wtr.writerow([t["name"], t["trial"], t["cohort"], hh["phase"], LAM_EX, LAM_EM,
                                 hh["round"], hh["nedit"],
                                 t["c1"] + 1, rows[si]["name"], t["pdb"], si, f"{EXM[si]:.0f}", f"{EMM[si]:.0f}",
                                 rows[ti]["name"], ti, f"{EXM[ti]:.0f}", f"{EMM[ti]:.0f}", f"{t['idv']:.3f}",
                                 f"{hh['pred_ex']:.1f}", f"{hh['pred_em']:.1f}", f"{hh['peak_err']:.2f}", "",
                                 f"{hh['fam_logp']:.2f}", f"{hh['ident']:.3f}", hh["seq"], t["scaffold"], seqs[ti]])

    # what the extra trials bought: the per-task spread is the search's own variance, which a
    # single trial cannot separate from a task being easy or hard
    if args.trials > 1:
        spread = [max(v) - min(v) for v in
                  ([t["hist"][-1]["peak_err"] for t in insts] for insts in by_task.values())]
        for tr in range(args.trials):
            errs = [t["hist"][-1]["peak_err"] for t in T if t["trial"] == tr]
            print(f"  trial {tr}: mean final oracle err {np.mean(errs):.1f} nm")
        print(f"  within-task spread across trials: mean {np.mean(spread):.1f} nm, "
              f"median {np.median(spread):.1f} nm, max {max(spread):.1f} nm")
    print(f"wrote {len(by_task)} design CSVs ({len(T)} trajectories) -> {args.outdir} "
          f"| total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
