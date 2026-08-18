#!/usr/bin/env python
"""Known-structure cohort design -- the ESM-2-proposal guided search. THE DESIGN ENGINE.

This is the search itself; ``run_task2_guided.py`` beside it is the named stage entry point, a thin
runner that passes the same defaults explicitly plus ``--no-ppl``. Nothing in the search is
task-specific -- the defaults below are simply stage 3.1's live task-2 configuration
(``pairs_task2/``, the two merged cohorts, ``C.PIPE_OUT_ESM2_T2_R3``). The same file, run as
::

    python 3.1_design_run_guided/design_knownstruct_guided.py --pairs-dir pairs \\
        --cohorts knownstruct_Strain knownstruct_Sval knownstruct_Stest \\
        --outdir peak_designs/structure/knownstruct_esm2_rand3 --no-ppl

reproduces the ARCHIVED task-1 ESM-2 arm -- it is the same code that produced it, moved here
when task set 1 was archived. (``archive/`` keeps task 1's own stage numbering, which does not
line up with the live stages; archived paths are always written with the ``archive/`` prefix.)

Per-position proposal distributions are the one component this arm swaps out. The archived
family-PSSM arm (``archive/3.1_design_run_MSA/design_knownstruct.py``) is identical in every other
respect -- same tasks, same Tier-B windows, same surrogate/oracle, same 1/1/1 guided-score
weights, same seed -- so any difference in outcome is attributable to the proposal:

  archived MSA arm  proposal = family-MSA PSSM (static per-position Henikoff-weighted
                    log-frequencies read out of ``design_windows.json``)
  THIS FILE         proposal = ESM-2 650M masked-LM logits at the position being edited,
                    conditioned on the design's CURRENT sequence -- the mechanism
                    ``archive/esm2_design/parallel_pipeline/design_knownstruct.py`` uses

Cohorts follow whichever manifests are passed. Task 1's three role manifests report as two
conditions (``seen`` = S-train + S-val, ``held-out`` = S-test) since the deployed surrogate was
refit on train+val; task 2 merges those pools up front into ``knownstruct_Spool`` /
``knownstruct_Stest``. Either way see ``design_common.COHORT_CONDITION`` -- it is an
analysis-time grouping and does not change what runs.

What "same window" means here, precisely. ``design_windows.json`` carries two separable things,
and only the second is MSA-derived:

  * STRUCTURAL -- the editable set (chromophore positions 1-2 + the 5 A Tier-B pocket) and the
    per-position alphabet constraints (aromatics at chromophore position 2, H-bond-capable
    residues at positions H-bonding the chromophore). These come from the scaffold's
    experimental structure and are KEPT verbatim.
  * FAMILY -- the PSSM: those alphabets intersected with what the 763-sequence family alignment
    supports at that column, plus the frequencies themselves. This is what ESM-2 REPLACES, so a
    position's candidate set here is its structural constraint (or all 20 AAs) and the ranking
    over it is ESM-2's, un-intersected with family support.

Weights are 1/1/1 on the z-scored guided score, unchanged from both arms:

    score = z(logp_proposal) - 1.0 * z(|ex_err|) - 1.0 * z(|em_err|)

with logp_proposal now the ESM-2 masked-LM log-probability rather than the PSSM's.

Both naturalness diagnostics are written per round so the two arms compare on the same axes:
``ppl`` (ESM-2 pseudo-perplexity over the whole sequence, the metric the MSA arm leaves blank)
and ``fam_logp`` (the design's log-likelihood under its own scaffold's family PSSM, the metric
the MSA arm optimizes). Neither enters the search.

TRIALS AND VISITING ORDER (kept identical to the MSA arm -- see its module docstring for the
reasoning). Each task is run ``--trials`` times independently, default 3, so the spread across
trials measures the search's own variance rather than the task's difficulty; within a cycle the
editable positions of each phase step are visited in a fresh per-trial random order, as
``design-campaign-EGFP`` does, instead of the fixed N->C walk of the first pass
(``--visit-order sequence`` restores it). Each trial carries its own seeded numpy/torch stream
derived from (SEED, trial, scaffold_idx, target_idx).

Note the visiting order matters MORE here than in the MSA arm: the ESM-2 proposal is conditioned
on the design's current sequence, so the order in which positions are edited changes the
proposal distribution at every later position, not just which candidates get scored first.

Usage
-----
    python 3.1_design_run_guided/run_task2_guided.py                            # task 2 -- the live arm
    python 3.1_design_run_guided/design_knownstruct_guided.py --no-ppl          # bare: the same, minus the runner
    python 3.1_design_run_guided/design_knownstruct_guided.py --cohorts knownstruct_Stest
    python 3.1_design_run_guided/design_knownstruct_guided.py --iters 3
    python 3.1_design_run_guided/design_knownstruct_guided.py --trials 1 --visit-order sequence
    python 3.1_design_run_guided/design_knownstruct_guided.py --smoke 2        # wiring/timing probe

``--no-ppl`` leaves the ``ppl`` column blank and is worth using whenever the pseudo-perplexity
diagnostic is not needed: it masks every residue of every design in turn, so it costs about as
much as a whole design cycle and dominates the runtime at 3 trials. ``fam_logp`` (free) is still
written, so the naturalness axis is not lost.

Writes one CSV per task, all trials in it, keyed by the ``trial`` column. Resumable: a task whose
CSV already covers the requested --trials and --iters is skipped; one that does not (including
any written by an earlier config) is recomputed and overwritten.
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

# ---- design hyper-parameters (z-scored guided-score scale; 1/1/1 as in the MSA arm) --------
LAM_PRIOR = 1.0      # weight on the ESM-2 masked-LM log-prob (the proposal/naturalness term)
LAM_EX = 1.0
LAM_EM = 1.0
K_TOP = 10
TEMP = 1.0
N_ITERS = 2          # design cycles per task; override with --iters
N_TRIALS = 3         # independent searches per task; override with --trials
ESM_BS = 48          # surrogate ESM-2 embedding batch
ORAC_BS = 32         # oracle ProstT5 batch
LOGIT_BS = 64        # masked-LM proposal batch (all tasks share one position slot per step)
PPL_BS = 96          # pseudo-perplexity: masked single-residue rows per forward

STD_AA = "ACDEFGHIKLMNPQRSTVWY"

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
    ap.add_argument("--cohorts", nargs="*", default=C.TASK2_COHORTS)
    ap.add_argument("--pairs-dir", default=str(C.PAIRS_DIR_T2),
                    help=f"manifest directory, i.e. which task set to run (default "
                         f"{C.PAIRS_DIR_T2.name}, stage 2's random-target set; {C.PAIRS_DIR.name} "
                         f"is the archived furthest-target set)")
    ap.add_argument("--smoke", type=int, default=0, help="run only the first N tasks (wiring/timing probe)")
    ap.add_argument("--iters", type=int, default=N_ITERS, help=f"design cycles per task (default {N_ITERS})")
    ap.add_argument("--no-ppl", action="store_true", help="skip the ESM-2 pseudo-perplexity diagnostic")
    ap.add_argument("--trials", type=int, default=N_TRIALS,
                    help=f"independent searches per task (default {N_TRIALS})")
    ap.add_argument("--visit-order", choices=["random", "sequence"], default="random",
                    help="order of editable positions within a phase step (default random)")
    ap.add_argument("--outdir", default=str(C.PIPE_OUT_ESM2_T2_R3),
                    help=f"output root (default {C.PIPE_OUT_ESM2_T2_R3.name})")
    args = ap.parse_args()

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

    # ---- models: this experiment's CV-selected surrogate + oracle-sweep winner ------------
    _sb, s_meta = pm.load_model(C.SURR_CKPT, dev); surrogate_net = pm.wrap(_sb, s_meta["mean"], s_meta["std"], dev)
    _ob, o_meta = pm.load_model(C.ORAC_CKPT, dev); oracle_net = pm.wrap(_ob, o_meta["mean"], o_meta["std"], dev)
    print(f"surrogate ({s_meta.get('role','?')}) test MAE {s_meta.get('test_mae', float('nan')):.1f} nm "
          f"(n_train={s_meta.get('n_train','?')}) | oracle val MAE {o_meta['val_mae']:.1f} nm")
    esm_model, alphabet, bc = pm.get_esm(dev)   # now serves BOTH the surrogate embedding and the proposal
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
        """Masked-LM logits at one 0-based position per sequence -- THE substitution for the
        MSA arm's static PSSM row. Chunked: all tasks advance the same slot together, so
        seqlist is as long as the task list."""
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

    def fam_logp_batch(T_list):
        """DIAGNOSTIC ONLY (the MSA arm's objective, reported here for comparability): the
        design's log-likelihood over its editable positions under its own scaffold's family
        PSSM. Never enters the guided score in this arm."""
        out = []
        for t in T_list:
            lp = 0.0
            for p in t["editable"]:
                row = t["pssm_logp"][p]
                lp += float(row.get(t["seq"][p], row["_min"]))
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

    # ---- per-task window: editable set + structural alphabets (PSSM kept for the
    # ---- fam_logp diagnostic only -- it does NOT gate candidates or rank them here) --
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

        # family PSSM -> per-AA log-prob rows, for the fam_logp diagnostic
        pssm_logp = {}
        for p_str, ent in w["pssm"].items():
            row = {aa: float(np.log(max(pr, 1e-12))) for aa, pr in zip(ent["alphabet"], ent["probs"])}
            row["_min"] = min(row.values())
            pssm_logp[int(p_str)] = row

        # one INSTANCE per (task, trial); the window tensors are read-only so trials share them
        tgt = torch.tensor(peaks[t["ti"]], device=dev)
        for trial in range(args.trials):
            rng, gen = _rng_pair(dev, trial, si, t["ti"])
            T.append(dict(**t, trial=trial, w=w, scaffold=seqs[si], seq=seqs[si], tgt=tgt,
                          editable=editable, pos_allowed=pos_allowed,
                          pssm_logp=pssm_logp, c1=c1, p2=p2, hist=[], rng=rng, gen=gen))
    ns = [len(t["editable"]) for t in T]
    print(f"running {len(todo)} tasks x {args.trials} trials = {len(T)} searches | {args.iters} "
          f"iters x (chromo -> pocket) | k={K_TOP} T={TEMP} lam_prior={LAM_PRIOR} "
          f"lam_ex={LAM_EX} lam_em={LAM_EM} | editable min/med/max = "
          f"{min(ns)}/{int(np.median(ns))}/{max(ns)} (Tier-B, same windows as the archived MSA arm) | "
          f"visit order = {args.visit_order} | ppl = {'off' if args.no_ppl else 'on'} | "
          f"proposal = ESM-2 masked-LM (family PSSM not used for selection; "
          f"{n_constrained} structurally constrained positions across tasks, rest = 20 AAs)")

    def resolve_positions(chromo_nums, use_pocket, t):
        """Positions to visit in this phase step, in the order they will be visited.

        With --visit-order random the order is a fresh per-trial permutation drawn here, i.e. once
        per (trial, cycle, step); with `sequence` it is the fixed N->C walk the first pass used.
        Either way the step's position SET is unchanged. Because the ESM-2 proposal is conditioned
        on the current sequence, this order also changes what gets PROPOSED downstream.
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
        fam = fam_logp_batch(T)
        for i, t in enumerate(T):
            ex, em = float(P[i, 0]), float(P[i, 1])
            err = 0.5 * (abs(ex - float(t["tgt"][0])) + abs(em - float(t["tgt"][1])))
            ident = sum(x == y for x, y in zip(t["seq"], t["scaffold"])) / len(t["scaffold"])
            t["hist"].append(dict(phase=phase, round=r, nedit=touched.get(id(t), 0),
                                  pred_ex=ex, pred_em=em, peak_err=err, ppl=ppls[i],
                                  fam_logp=fam[i], ident=ident, seq=t["seq"]))

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
                        for aa in aas:
                            cand_all.append(t["seq"][:pos] + aa + t["seq"][pos + 1:])
                        meta.append((t, pos, topv, aas, k_eff))
                    Pk = surrogate_peaks_batched(cand_all)
                    off = 0
                    for (t, pos, topv, aas, k_eff) in meta:
                        Pc = Pk[off:off + k_eff]; off += k_eff
                        ex_err = (Pc[:, 0] - t["tgt"][0]).abs(); em_err = (Pc[:, 1] - t["tgt"][1]).abs()
                        scores = LAM_PRIOR * _zc(topv) - LAM_EX * _zc(ex_err) - LAM_EM * _zc(em_err)
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
                                 f"{hh['pred_ex']:.1f}", f"{hh['pred_em']:.1f}", f"{hh['peak_err']:.2f}",
                                 "" if args.no_ppl else f"{hh['ppl']:.2f}",
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
