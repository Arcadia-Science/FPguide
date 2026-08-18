#!/usr/bin/env python
"""Unguided control -- ESM-2 Gibbs sampling in the same design window, no surrogate.

This is ``3.1_design_run_guided/design_knownstruct_guided.py`` with **lam_ex = lam_em = 0**. Same
tasks, same Tier-B windows, same ESM-2 650M masked-LM proposal, same k = 10 / T = 1.0 / 2 cycles,
same per-trial random visiting order, same seeding scheme. The only change is that the two
guidance terms are switched off, which leaves::

    GUIDED      score = z(logp_proposal) - 1.0 * z(|ex_err|) - 1.0 * z(|em_err|)
    THIS ARM    score = z(logp_esm)                                                 (lam_ex = lam_em = 0)

so the target spectrum never enters the search. Each editable position is resampled from ESM-2's
conditional at that position given the design's current sequence -- a Gibbs sweep over the pocket
under the window's structural constraints -- and the oracle simply watches where that wanders.

THE DESIGN ENGINE. ``run_task2_gibbs.py`` beside it is the named stage entry point, a thin runner
passing the same defaults explicitly plus ``--no-ppl``. The defaults below are stage 3.2's live
task-2 configuration (``pairs_task2/``, S-pool + S-test, ``C.PIPE_OUT_GIBBS_T2_R12``); pointing
``--pairs-dir``/``--cohorts``/``--outdir`` at task 1 reproduces the ARCHIVED task-1 null, which
this same code produced before task set 1 was archived. See ``archive/README.md`` -- its stage
numbering is task 1's own and does not line up with the live stages.

WHY. Every number a guided arm reports is an improvement *over the scaffold*, and a scaffold is
not a null: mutating 26 pocket positions moves ex/em whether or not anything is steering. This
arm is the null those results need. Read the arms as::

    scaffold error  ->  Gibbs (this arm)  ->  guided (3.1 ESM-2; archived task-1 MSA / ESM-2)

If Gibbs closes most of the gap the guided arms report, the guidance is doing little; if it
drifts or worsens, the gap between it and the guided arm is what the surrogate is really buying.

COHORTS AND TRIALS. S-train (36) + S-test (36) only, 12 trials each = 864 searches. S-val is
skipped: it is the same reporting condition as S-train (``seen``, see ``design_common``), so a
balanced 36 seen / 36 held-out pair of cohorts costs a third of the compute and says the same
thing. **Comparisons against the archived task-1 arms must be restricted to those 72 tasks**, not
run against their published 108-task means. 12 trials rather than 3 because this arm's whole
output is a distribution -- with a 24-28 nm within-task trial spread in the guided arms, a null
estimated from 3 draws is not a null.

NO SURROGATE IN THE LOOP. With both lambdas at 0 the k = 10 candidates' surrogate predictions are
multiplied by zero, so this arm skips the surrogate forward pass entirely (that pass is 10 seqs
per position per search; dropping it is where the ~10x speedup over 3.1 comes from). It is an
exact short-circuit, not an approximation: the RNG streams and the sampled candidate are
unchanged. ``--lam-ex``/``--lam-em`` restore the guided path (and the surrogate load) if given.
The surrogate can still be applied post-hoc for analysis --
``python score_traj_surrogate.py --arm gibbs_r12``.

IS THIS *EXACTLY* GIBBS? Not quite, by design: the step keeps 3.1's machinery so the arms differ
in one place only. Two deviations from sampling p(x_p | x_-p):

  * candidates are the **top k = 10** tokens of the allowed alphabet, not all of it (a truncated
    conditional -- it drops tail mass ESM-2 assigns to residues it considers implausible);
  * their log-probs are **z-scored** before the softmax, exactly as 3.1 z-scores them against the
    error terms, which rescales the conditional by its own spread rather than preserving it.

``--proposal raw`` removes the z-scoring and samples ``softmax(logp / T)`` over the same top-k,
i.e. the truncated conditional itself. Default is ``zscore``, the controlled guided-minus-guidance
setting, because that is what makes the difference between this arm and 3.1 attributable to the
guidance alone.

Usage
-----
    python 3.2_design_run_gibbs/run_task2_gibbs.py                          # task 2 -- the live null
    python 3.2_design_run_gibbs/design_knownstruct_gibbs.py --no-ppl        # bare: the same, minus the runner
    python 3.2_design_run_gibbs/design_knownstruct_gibbs.py --proposal raw  # untruncated-scale Gibbs
    python 3.2_design_run_gibbs/design_knownstruct_gibbs.py --smoke 2 --trials 2   # wiring probe
    python 3.2_design_run_gibbs/design_knownstruct_gibbs.py --lam-ex 1 --lam-em 1  # reproduces 3.1

Writes one CSV per task with every trial in it, identical columns to 3.1 (``lam_ex``/``lam_em``
record the 0/0 used). Resumable: a task whose CSV already covers the requested --trials and
--iters is skipped.
"""
import argparse
import csv
import json
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

# --- stage-folder bootstrap: put the experiment root (design_common) and lib/ (vendored
# --- modules) on the import path. msa/ is NOT needed -- this arm reads no alignment.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, "lib")]

import design_common as C

import peak_models as pm      # vendored copy -- this experiment folder is self-contained
import prostt5_embed as pe

# ---- design hyper-parameters -- 3.1's, with the two guidance weights zeroed -----------------
LAM_PRIOR = 1.0      # weight on the ESM-2 masked-LM log-prob (now the ONLY term in the score)
LAM_EX = 0.0         # <- 3.1 uses 1.0; this is the whole point of the arm
LAM_EM = 0.0         # <- 3.1 uses 1.0
K_TOP = 10
TEMP = 1.0
N_ITERS = 2          # design cycles per task; override with --iters
N_TRIALS = 12        # independent searches per task -- this arm reports a distribution
ORAC_BS = 32         # oracle ProstT5 batch
ESM_BS = 48          # surrogate ESM-2 embedding batch (only used if --lam-ex/--lam-em are given)
LOGIT_BS = 64        # masked-LM proposal batch (all searches share one position slot per step)
PPL_BS = 96          # pseudo-perplexity: masked single-residue rows per forward

# The archived task-1 control's cohorts: S-val was dropped there because it is the same
# reporting condition as S-train, so 36 seen + 36 held-out is balanced and costs a third
# less. Task 2 (the default, C.TASK2_COHORTS) merges train+val up front and needs no such
# choice. Kept for reproducing the archived run: --cohorts knownstruct_Strain knownstruct_Stest
GIBBS_COHORTS_TASK1 = ["knownstruct_Strain", "knownstruct_Stest"]

STD_AA = "ACDEFGHIKLMNPQRSTVWY"

COLS = ["example", "trial", "cohort", "phase", "lam_ex", "lam_em", "round", "n_editable",
        "chromo_pos1_1based", "scaffold_name", "scaffold_pdb", "scaffold_idx",
        "scaffold_ex", "scaffold_em", "target_name", "target_idx", "target_ex", "target_em",
        "seq_id_scaf_target", "pred_ex", "pred_em", "peak_err", "ppl",
        "ident_to_scaffold", "designed_seq", "scaffold_seq", "target_seq"]


def _san(s):
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in s)


def _complete(fn, trials, iters):
    """Does this CSV already hold every requested trial, each run to `iters` cycles?

    Stricter than "the file exists", because the trial and cycle counts are configuration: a CSV
    from a shorter --trials or --iters must be recomputed rather than silently accepted as cached.
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
    """Independent (numpy, torch) streams for one trial, reproducible from its identity alone.

    Same construction as the guided arm, so trial `t` of a task starts from the same state it would
    have there -- the arms differ in the score, not in the randomness fed to it.
    """
    ss = np.random.SeedSequence([C.SEED, trial, si, ti])
    rng = np.random.default_rng(ss)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(rng.integers(2 ** 62)))
    return rng, gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="*", default=C.TASK2_COHORTS)
    ap.add_argument("--pairs-dir", default=str(C.PAIRS_DIR_T2),
                    help=f"manifest directory, i.e. which task set to run (default "
                         f"{C.PAIRS_DIR_T2.name}, stage 2's random-target set; archive/pairs "
                         f"is the archived furthest-target set)")
    ap.add_argument("--smoke", type=int, default=0, help="run only the first N tasks (wiring/timing probe)")
    ap.add_argument("--iters", type=int, default=N_ITERS, help=f"design cycles per task (default {N_ITERS})")
    ap.add_argument("--no-ppl", action="store_true", help="skip the ESM-2 pseudo-perplexity diagnostic")
    ap.add_argument("--trials", type=int, default=N_TRIALS,
                    help=f"independent searches per task (default {N_TRIALS})")
    ap.add_argument("--visit-order", choices=["random", "sequence"], default="random",
                    help="order of editable positions within a phase step (default random)")
    ap.add_argument("--proposal", choices=["zscore", "raw"], default="zscore",
                    help="zscore: 3.1's z-scored top-k log-probs (default, the controlled setting); "
                         "raw: softmax of the top-k log-probs themselves (the truncated conditional)")
    ap.add_argument("--lam-ex", type=float, default=LAM_EX,
                    help=f"weight on z(|ex_err|); default {LAM_EX} -- nonzero restores 3.1's guided search")
    ap.add_argument("--lam-em", type=float, default=LAM_EM,
                    help=f"weight on z(|em_err|); default {LAM_EM}")
    ap.add_argument("--outdir", default=str(C.PIPE_OUT_GIBBS_T2_R12),
                    help=f"output root (default {C.PIPE_OUT_GIBBS_T2_R12.name})")
    args = ap.parse_args()

    # with both lambdas at 0 the surrogate's predictions are multiplied by zero, so it is not
    # loaded and its forward pass is skipped -- an exact short-circuit, see the module docstring
    guided = (args.lam_ex != 0.0) or (args.lam_em != 0.0)

    if not os.path.exists(C.WINDOWS_JSON):
        raise SystemExit(f"missing {C.WINDOWS_JSON}; run 2_design_task_specification/build_windows.py first")
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
        fn = C.pairs_csv_path(coh, args.pairs_dir)
        if not os.path.exists(fn):
            raise SystemExit(f"missing manifest {fn}; run the curation stage for this task set first")
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
        print(f"{n_no_window} task(s) skipped: scaffold has no Tier-B window "
              f"(failed structure quality gate or not in family alignment)")
    if args.smoke:
        tasks = tasks[:args.smoke]
    print(f"{len(tasks)} tasks across {len(args.cohorts)} cohort(s): "
          + ", ".join(f"{c}={sum(1 for t in tasks if t['cohort']==c)}" for c in args.cohorts))

    # ---- models: the oracle always (it scores every round); the surrogate only if guided ----
    _ob, o_meta = pm.load_model(C.ORAC_CKPT, dev); oracle_net = pm.wrap(_ob, o_meta["mean"], o_meta["std"], dev)
    print(f"oracle val MAE {o_meta['val_mae']:.1f} nm"
          + ("" if guided else " | surrogate NOT loaded (lam_ex = lam_em = 0)"))
    surrogate_net = None
    if guided:
        _sb, s_meta = pm.load_model(C.SURR_CKPT, dev); surrogate_net = pm.wrap(_sb, s_meta["mean"], s_meta["std"], dev)
        print(f"surrogate ({s_meta.get('role','?')}) test MAE {s_meta.get('test_mae', float('nan')):.1f} nm "
              f"(n_train={s_meta.get('n_train','?')})")
    esm_model, alphabet, bc = pm.get_esm(dev)   # the proposal (and the surrogate embedding, if guided)
    V = len(alphabet.all_toks)

    AA_MASK = torch.zeros(V, dtype=torch.bool)
    for a in STD_AA:
        AA_MASK[alphabet.get_idx(a)] = True
    AA_MASK = AA_MASK.to(dev)
    SPECIAL = {alphabet.cls_idx, alphabet.eos_idx, alphabet.padding_idx, alphabet.mask_idx}

    def aa_mask_from(letters):
        """Boolean token mask over an explicit amino-acid alphabet (the window's structural
        constraint at a position), intersected with the 20 standard residues."""
        m = torch.zeros(V, dtype=torch.bool)
        for a in letters:
            if a in STD_AA:
                m[alphabet.get_idx(a)] = True
        return m.to(dev)

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
    def esm_logits_at(seqlist, positions, bs=LOGIT_BS):
        """Masked-LM logits at one 0-based position per sequence -- the Gibbs conditional
        p(x_p | x_-p). Chunked: all searches advance the same slot together, so seqlist is as
        long as the live search list."""
        outs = []
        for i in range(0, len(seqlist), bs):
            sl, ps = seqlist[i:i + bs], positions[i:i + bs]
            _, _, tk = bc([(f"s{j}", s) for j, s in enumerate(sl)]); tk = tk.to(dev)
            rr = torch.arange(len(sl), device=dev)
            cols = torch.tensor([p + 1 for p in ps], device=dev)   # +1 for the BOS token
            tk[rr, cols] = alphabet.mask_idx
            with AMP():
                lg = esm_model(tk)["logits"]
            outs.append(lg[rr, cols].float())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def ppl_batched(seqlist, bs=PPL_BS):
        """ESM-2 pseudo-perplexity: mask each residue in turn, average its log-prob."""
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

    # ---- per-task window: editable set + structural alphabets. Identical to 3.1. ----------
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    T = []
    n_constrained = 0
    for t in todo:
        si = t["si"]; nm = rows[si]["name"]; w = WINJ[nm]
        c1 = w["chromophore"]["pos1_0based"]; p2 = w["chromophore"]["pos2_0based"]
        editable = list(w["editable_0based"])

        # structural per-position alphabets (aromatic at chromophore pos 2, H-bond-capable at
        # H-bond partners); every other editable position gets the full 20 AAs
        pos_allowed = {}
        for p_str, letters in w.get("position_constraints", {}).items():
            pos_allowed[int(p_str)] = aa_mask_from(letters)
        n_constrained += len(pos_allowed)


        # one INSTANCE per (task, trial); the window tensors are read-only so trials share them
        tgt = torch.tensor(peaks[t["ti"]], device=dev)
        for trial in range(args.trials):
            rng, gen = _rng_pair(dev, trial, si, t["ti"])
            T.append(dict(**t, trial=trial, w=w, scaffold=seqs[si], seq=seqs[si], tgt=tgt,
                          editable=editable, pos_allowed=pos_allowed,
                          c1=c1, p2=p2, hist=[], rng=rng, gen=gen))
    ns = [len(t["editable"]) for t in T]
    print(f"running {len(todo)} tasks x {args.trials} trials = {len(T)} searches | {args.iters} "
          f"iters x (chromo -> pocket) | k={K_TOP} T={TEMP} lam_prior={LAM_PRIOR} "
          f"lam_ex={args.lam_ex} lam_em={args.lam_em} | editable min/med/max = "
          f"{min(ns)}/{int(np.median(ns))}/{max(ns)} (Tier-B, same windows as the guided arm) | "
          f"visit order = {args.visit_order} | proposal = ESM-2 masked-LM ({args.proposal}) | "
          f"ppl = {'off' if args.no_ppl else 'on'} | "
          + ("guided (surrogate in the loop)" if guided
             else "UNGUIDED: target spectrum not used; surrogate pass skipped")
          + f" | {n_constrained} structurally constrained positions across tasks, rest = 20 AAs")

    def resolve_positions(chromo_nums, use_pocket, t):
        """Positions to visit in this phase step, in the order they will be visited.

        With --visit-order random the order is a fresh per-trial permutation drawn here, i.e. once
        per (trial, cycle, step); with `sequence` it is a fixed N->C walk. Either way the step's
        position SET is unchanged. Because the ESM-2 conditional depends on the current sequence,
        the order also changes what gets proposed downstream -- which is exactly what makes a
        random sweep order the right choice for a Gibbs sampler.
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
        ppls = [float("nan")] * len(T) if args.no_ppl else ppl_batched(seqlist)
        for i, t in enumerate(T):
            ex, em = float(P[i, 0]), float(P[i, 1])
            err = 0.5 * (abs(ex - float(t["tgt"][0])) + abs(em - float(t["tgt"][1])))
            ident = sum(x == y for x, y in zip(t["seq"], t["scaffold"])) / len(t["scaffold"])
            t["hist"].append(dict(phase=phase, round=r, nedit=touched.get(id(t), 0),
                                  pred_ex=ex, pred_em=em, peak_err=err, ppl=ppls[i],
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
                    # ESM-2 masked-LM logits, conditioned on each design's CURRENT sequence
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
                        if guided:
                            for aa in aas:
                                cand_all.append(t["seq"][:pos] + aa + t["seq"][pos + 1:])
                        meta.append((t, pos, topv, aas, k_eff))
                    # unguided: no surrogate call at all -- its term is multiplied by zero
                    Pk = surrogate_peaks_batched(cand_all) if guided else None
                    off = 0
                    for (t, pos, topv, aas, k_eff) in meta:
                        # the proposal term: 3.1's z-scored log-probs, or the truncated
                        # conditional itself under --proposal raw
                        scores = LAM_PRIOR * (_zc(topv) if args.proposal == "zscore" else topv)
                        if guided:
                            Pc = Pk[off:off + k_eff]; off += k_eff
                            ex_err = (Pc[:, 0] - t["tgt"][0]).abs(); em_err = (Pc[:, 1] - t["tgt"][1]).abs()
                            scores = scores - args.lam_ex * _zc(ex_err) - args.lam_em * _zc(em_err)
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
                    wtr.writerow([t["name"], t["trial"], t["cohort"], hh["phase"], args.lam_ex, args.lam_em,
                                 hh["round"], hh["nedit"],
                                 t["c1"] + 1, rows[si]["name"], t["pdb"], si, f"{EXM[si]:.0f}", f"{EMM[si]:.0f}",
                                 rows[ti]["name"], ti, f"{EXM[ti]:.0f}", f"{EMM[ti]:.0f}", f"{t['idv']:.3f}",
                                 f"{hh['pred_ex']:.1f}", f"{hh['pred_em']:.1f}", f"{hh['peak_err']:.2f}",
                                 "" if args.no_ppl else f"{hh['ppl']:.2f}",
                                 f"{hh['ident']:.3f}", hh["seq"], t["scaffold"], seqs[ti]])

    # the trial spread IS the result in this arm: it is the width of the null a guided run has to
    # beat, not a nuisance to average away
    if args.trials > 1:
        spread = [max(v) - min(v) for v in
                  ([t["hist"][-1]["peak_err"] for t in insts] for insts in by_task.values())]
        finals = [t["hist"][-1]["peak_err"] for t in T]
        scaff = [t["hist"][0]["peak_err"] for t in T]
        print(f"  final oracle err over all {len(T)} searches: mean {np.mean(finals):.1f} nm "
              f"(scaffold {np.mean(scaff):.1f} nm) | best-of-{args.trials} per task "
              f"{np.mean([min(v) for v in ([t['hist'][-1]['peak_err'] for t in insts] for insts in by_task.values())]):.1f} nm")
        print(f"  within-task spread across trials: mean {np.mean(spread):.1f} nm, "
              f"median {np.median(spread):.1f} nm, max {max(spread):.1f} nm")
    print(f"wrote {len(by_task)} design CSVs ({len(T)} trajectories) -> {args.outdir} "
          f"| total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
