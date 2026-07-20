#!/usr/bin/env python
"""Gibbs-sampling design campaign: pure ESM-2 masked-LM design, ONE scaffold->target pair at a time.

This is the ESM-2-only sibling of ../guided_design. It runs the IDENTICAL design procedure
(same 24 pairs, same 5 A windows/constraints, any-order masking, --trials independent trials,
--iters passes, batched forwards, resumable, same CSV schema) but changes ONLY the per-position
scoring: candidates are chosen from ESM-2's masked-LM likelihood alone. There is NO surrogate
peak-error (MAE) term in the search -- we do NOT bias the choice toward the target (ex, em).

We iterate over the 24 pairs (../pairs/campaign_pairs_24.csv) sequentially. For each pair we run
``--trials`` independent design trials (default 6) starting from the scaffold sequence. All trials
of a pair share the IDENTICAL design window (../design_windows_24.json: chromophore pos1 & pos2 +
the 5 A pocket; pos2 restricted to aromatics {Y,W,H,F}; Gly + catalytic Arg/Glu fixed), but each
trial masks/edits those window positions in its OWN random order (any-order masking; a fresh random
permutation is drawn per trial per iteration). Sweeping random-scan positions and resampling each
from its ESM-2 conditional p(x_i | x_{-i}) is masked-LM Gibbs sampling over the editable window.

Each iteration (default 3) is a masked-LM proposal + ESM-2-only selection over the window:
  at each visited position, ESM-2 gives the top-k allowed residues with log-probs logp; we sample
  DIRECTLY from that conditional -- softmax(logp / T) at T=1 -- so the residue is a true Gibbs draw
  from p(x_i | x_{-i}) (restricted to the top-k). No surrogate is called inside the design loop.

NOTE (vs ../guided_design): that driver scores z(logp_ESM) - lam*z(|d ex|) - lam*z(|d em|) with
lam=20, T=10; the lam=20 error terms dominate (spread ~+-3-4 after /T) while the z-scored ESM term
is near-uniform (spread ~+-0.15 after /T), so guided design is effectively greedy toward the
surrogate target with ESM as a weak tie-break. Here we drop the surrogate terms; if we had reused
guided's z-scored ESM term at T=10 the choice would be ~uniform over the top-k, so instead we use
RAW log-probs at T=1 to actually follow ESM-2's distribution.

DIAGNOSTIC (never used for guidance): the ALL-DATA surrogate
(../models/surrogate_cnn-max-d1_alldata.pt) still predicts (ex, em) once per round so we can record
where the ESM-2-driven sequence lands relative to the target. There is NO oracle for this task
(the surrogate is trained on ALL data; the real judge is experiment). The recorded prediction
(pred_ex, pred_em) is the SURROGATE's own (ex, em) for the current sequence, and peak_err is the
surrogate's distance to the target -- a diagnostic only, NOT an objective the search optimizes.
ESM-2 pseudo-perplexity is logged as a naturalness diagnostic. Settings: T=1, k=10.

ACCELERATION: within a pair the 6 trials advance together in one GPU batch (same window, so slot j
is genuinely the same set of positions across trials, differing only by each trial's random order);
fp16 autocast on CUDA, sub-batched forwards.

Trajectory: one CSV per pair (designs/design_<scaffold>-<target>.csv), one row per (trial, round);
round 0 = scaffold. The CSV schema is kept identical to ../guided_design for head-to-head
comparison; the lam_ex/lam_em cells are left blank here because no surrogate MAE term is used.

RESUMABLE AT TRIAL GRANULARITY: on (re)run we count how many trials a pair CSV already holds and
compute ONLY the missing trials [have, --trials), APPENDING them to the CSV. Re-running with a
larger --trials therefore EXPANDS a pair (e.g. 6 -> 24 designs) without recomputing the first 6.
Each trial's RNGs (visiting order + residue sampling) are seeded per trial from (SEED, scaffold_idx,
trial), so trial k is bit-for-bit identical no matter when or alongside how many others it is drawn.

Usage
-----
    python design_campaign.py --probe          # time ONE pair (6 trials), project 24 pairs, EXIT
    python design_campaign.py                  # full run, one pair at a time (default --trials 6)
    python design_campaign.py --trials 6 --iters 3 --temp 1 --k 10
    python design_campaign.py --trials 24      # EXPAND every pair to 24 trials (appends 6..23)
    python design_campaign.py --pairs-limit 2  # first 2 pairs only
    python design_campaign.py --backfill-ppl   # fill empty ppl cells in existing CSVs (deduped, batched)

Tip: run designs fast with `--ppl endpoints` (ppl only at scaffold + final round; intermediate
rounds left blank), then fill the blanks later with `--backfill-ppl`.
"""
import argparse
import csv
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent      # .../design-campaign-conventional/gibbs-sampling
CAMPAIGN = HERE.parent                       # .../design-campaign-conventional (shared assets)
REPO = CAMPAIGN.parent                       # .../spectrum-to-fp-design
ESM2 = REPO / "esm2_design"
CUR = REPO / "dataset_pipeline" / "data" / "peak" / "curated"
sys.path.insert(0, str(ESM2))
import peak_models as pm            # noqa: E402

SURR_CKPT = CAMPAIGN / "models" / "surrogate_cnn-max-d1_alldata.pt"
WINDOWS_JSON = CAMPAIGN / "design_windows_24.json"
PAIRS_CSV = CAMPAIGN / "pairs" / "campaign_pairs_24.csv"
OUTDIR = HERE / "designs"

SEED = 42
ESM_BS = 64            # sub-batch for ESM-2 embed / logits forwards
PPL_BS = 128           # sub-batch for pseudo-perplexity single-mask rows

COLS = ["pair", "scaffold_name", "scaffold_idx", "scaffold_pdb", "target_name", "target_idx",
        "selection", "scaffold_ex", "scaffold_em", "target_ex", "target_em", "seq_id_scaf_target",
        "trial", "round", "n_editable", "temp", "k", "lam_ex", "lam_em",
        "pred_ex", "pred_em", "peak_err", "ppl", "ident_to_scaffold",
        "designed_seq", "scaffold_seq", "target_seq"]


def _san(s):
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in s)


def load_dataset():
    rows = list(csv.DictReader(open(CUR / "peaks_assignments.csv")))
    peaks = np.load(CUR / "peaks.npy").astype(np.float32)
    seqs = [None] * len(rows)
    h = None
    for line in open(CUR / "sequences.fasta"):
        line = line.strip()
        if line.startswith(">"):
            h = int(line[1:].split("|")[0])
        elif line:
            seqs[h] = line
    return rows, seqs, peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pairs-limit", type=int, default=0, help="use only the first N pairs")
    ap.add_argument("--ppl", choices=["all", "endpoints"], default="endpoints",
                    help="compute ESM-2 pseudo-perplexity every round ('all') or only scaffold + final ('endpoints')")
    ap.add_argument("--probe", action="store_true",
                    help="time ONE pair (round-0 eval + 1 iteration), project the 24-pair total, and EXIT")
    ap.add_argument("--backfill-ppl", action="store_true",
                    help="do NOT design; instead fill empty ppl cells in existing designs/*.csv. ppl "
                         "depends only on the sequence, so unique sequences are computed once (deduped) "
                         "in large batches and the values fanned out to every matching row.")
    args = ap.parse_args()

    dev = (torch.device("cuda") if torch.cuda.is_available()
           else torch.device("mps") if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
           else torch.device("cpu"))
    use_fp16 = (dev.type == "cuda")
    AMP = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_fp16 else (lambda: nullcontext())
    print(f"device {dev} | fp16 {use_fp16}", flush=True)

    rows, seqs, peaks = load_dataset()
    windows = json.load(open(WINDOWS_JSON))["windows"]
    pairs = list(csv.DictReader(open(PAIRS_CSV)))
    if args.pairs_limit:
        pairs = pairs[:args.pairs_limit]

    # ---- model (surrogate is a DIAGNOSTIC ONLY; it never guides the search) --------
    _sb, s_meta = pm.load_model(str(SURR_CKPT), dev); surrogate_net = pm.wrap(_sb, s_meta["mean"], s_meta["std"], dev)
    s_mae = s_meta.get("train_mae", s_meta.get("val_mae", float("nan")))
    print(f"surrogate (all-data, diagnostic only) train MAE {s_mae:.1f} nm", flush=True)
    esm_model, alphabet, bc = pm.get_esm(dev)

    def _mask(aas):
        m = torch.zeros(len(alphabet.all_toks), dtype=torch.bool)
        for a in aas:
            m[alphabet.get_idx(a)] = True
        return m.to(dev)

    AA_MASK = _mask("ACDEFGHIKLMNPQRSTVWY")
    SPECIAL = {alphabet.cls_idx, alphabet.eos_idx, alphabet.padding_idx, alphabet.mask_idx}

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
    def esm_logits_at(seqlist, positions, bs=ESM_BS):
        outs = []
        for i in range(0, len(seqlist), bs):
            sl = seqlist[i:i + bs]; ps = positions[i:i + bs]
            _, _, tk = bc([(f"s{j}", s) for j, s in enumerate(sl)]); tk = tk.to(dev)
            rr = torch.arange(len(sl), device=dev)
            cols = torch.tensor([p + 1 for p in ps], device=dev)
            tk[rr, cols] = alphabet.mask_idx
            with AMP():
                lg = esm_model(tk)["logits"]
            outs.append(lg[rr, cols].float())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def ppl_batched(seqlist, bs=PPL_BS):
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

    # ---- per-pair design (batches the pair's trials) ------------------------------
    def build_trials(pr, trial_start=0, trial_end=None):
        """Build the pair's trial instances for trials [trial_start, trial_end).

        Both the visiting-order RNG and the residue-sampling RNG are seeded PER TRIAL from
        (SEED, scaffold_idx, trial), so trial k is identical regardless of how many trials are
        requested or when it is computed. This lets us EXPAND a pair (e.g. 6 -> 24 trials) later
        without recomputing the trials we already have."""
        trial_end = args.trials if trial_end is None else trial_end
        si, ti = int(pr["scaffold_idx"]), int(pr["target_idx"])
        w = windows[pr["scaffold_name"]]
        editable = list(w["editable_0based"])
        pos_allowed = {int(p): _mask(aas) for p, aas in w["position_constraints"].items()}
        scaf = w["scaffold_seq"]
        tgt = torch.tensor(peaks[ti], device=dev)
        insts = []
        for trial in range(trial_start, trial_end):
            seed = SEED + si * 131 + trial * 17
            insts.append(dict(si=si, ti=ti, editable=editable, pos_allowed=pos_allowed,
                              scaffold=scaf, seq=scaf, tgt=tgt, trial=trial,
                              rng=np.random.default_rng(seed),
                              gen=torch.Generator(device=dev).manual_seed(seed), hist=[]))
        return insts

    def evaluate_round(insts, r, do_ppl):
        # surrogate (ex, em) recorded as a DIAGNOSTIC ONLY; it does not steer the search
        seqlist = [t["seq"] for t in insts]
        P = surrogate_peaks_batched(seqlist)
        ppls = ppl_batched(seqlist) if do_ppl else [float("nan")] * len(insts)
        for i, t in enumerate(insts):
            ex, em = float(P[i, 0]), float(P[i, 1])
            err = 0.5 * (abs(ex - float(t["tgt"][0])) + abs(em - float(t["tgt"][1])))
            ident = sum(x == y for x, y in zip(t["seq"], t["scaffold"])) / len(t["scaffold"])
            t["hist"].append(dict(round=r, pred_ex=ex, pred_em=em, peak_err=err, ppl=ppls[i], ident=ident, seq=t["seq"]))

    def run_iteration(insts):
        # each trial: fresh random visiting order over the (identical) window this iteration
        for t in insts:
            t["_ed"] = list(t["rng"].permutation(t["editable"]))
        maxlen = max(len(t["_ed"]) for t in insts)
        for j in range(maxlen):
            sub = [t for t in insts if j < len(t["_ed"])]
            positions = [int(t["_ed"][j]) for t in sub]
            logits = esm_logits_at([t["seq"] for t in sub], positions)
            for i, t in enumerate(sub):
                pos = positions[i]
                mh = t["pos_allowed"].get(pos, AA_MASK)
                lg = logits[i].masked_fill(~mh, float("-inf"))
                logp = torch.log_softmax(lg, -1)
                k_eff = min(args.k, int(mh.sum().item()))
                topv, topi = torch.topk(logp, k_eff)
                aas = [alphabet.get_tok(int(x)) for x in topi.tolist()]
                # ESM-2-only scoring: sample directly from the top-k masked-LM conditional.
                # scores are the RAW log-probs (NOT z-scored), so softmax(scores / T) at T=1
                # reconstructs p(x_i | x_{-i}) restricted to the top-k -> true Gibbs draw.
                scores = topv
                ch = (int(torch.multinomial(torch.softmax(scores / args.temp, -1), 1, generator=t["gen"]).item())
                      if args.temp > 0 else int(torch.argmax(scores)))
                t["seq"] = t["seq"][:pos] + aas[ch] + t["seq"][pos + 1:]

    def existing_pair(fn):
        """Inspect an existing pair CSV for trial-level resume / posterior addition.

        Returns (n_trials, rounds_per_trial, temp, k):
          n_trials         = max(trial)+1 (0 if the file is absent -> build from scratch),
          rounds_per_trial = rows for the most-populated trial (= iters+1 for a complete trial),
          temp, k          = the settings recorded in the file (for a consistency check on append).
        """
        if not fn.exists():
            return 0, None, None, None
        mx = -1
        rounds, temp, k = {}, None, None
        with open(fn, newline="") as fh:
            for row in csv.DictReader(fh):
                tr = int(row["trial"]); mx = max(mx, tr)
                rounds[tr] = rounds.get(tr, 0) + 1
                temp, k = row.get("temp"), row.get("k")
        return mx + 1, (max(rounds.values()) if rounds else None), temp, k

    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def write_pair(pr, name, fn, insts, append=False):
        si, ti = int(pr["scaffold_idx"]), int(pr["target_idx"])
        with open(fn, "a" if append else "w", newline="") as fh:
            wr = csv.writer(fh)
            if not append:
                wr.writerow(COLS)
            for t in sorted(insts, key=lambda x: x["trial"]):
                for hh in t["hist"]:
                    wr.writerow([name, pr["scaffold_name"], si, pr["scaffold_pdb"],
                                 pr["target_name"], ti, pr.get("selection", ""),
                                 f"{peaks[si,0]:.0f}", f"{peaks[si,1]:.0f}", f"{peaks[ti,0]:.0f}", f"{peaks[ti,1]:.0f}",
                                 pr["identity"], t["trial"], hh["round"], len(t["editable"]),
                                 args.temp, args.k, "", "",
                                 f"{hh['pred_ex']:.1f}", f"{hh['pred_em']:.1f}", f"{hh['peak_err']:.2f}",
                                 f"{hh['ppl']:.2f}" if hh["ppl"] == hh["ppl"] else "", f"{hh['ident']:.3f}",
                                 hh["seq"], t["scaffold"], seqs[ti]])

    os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    # ---- BACKFILL PPL: fill empty ppl cells in existing CSVs, then exit ------------
    if args.backfill_ppl:
        files = sorted(OUTDIR.glob("design_*.csv"))
        parsed, need = [], set()                 # collect unique sequences that still lack a ppl
        for fn in files:
            with open(fn, newline="") as fh:
                rows = list(csv.DictReader(fh))
            parsed.append((fn, rows))
            for row in rows:
                if row.get("ppl", "").strip() == "":
                    need.add(row["designed_seq"])
        if not need:
            print(f"backfill: scanned {len(files)} files, no empty ppl cells -> nothing to do.", flush=True); return
        seqs_u = sorted(need)
        print(f"backfill: {len(seqs_u)} unique sequences need ppl across {len(files)} files", flush=True)
        t0 = time.time(); vals = {}
        for i in range(0, len(seqs_u), 1024):    # chunk unique seqs (bounds meta list; progress log)
            chunk = seqs_u[i:i + 1024]
            for s, p in zip(chunk, ppl_batched(chunk, bs=256)):
                vals[s] = p
            print(f"  ppl {min(i + 1024, len(seqs_u))}/{len(seqs_u)} ({time.time() - t0:.0f}s)", flush=True)
        filled = 0
        for fn, rows in parsed:
            changed = False
            for row in rows:
                if row.get("ppl", "").strip() == "":
                    p = vals.get(row["designed_seq"])
                    if p is not None and p == p:     # skip NaN just in case
                        row["ppl"] = f"{p:.2f}"; filled += 1; changed = True
            if changed:
                with open(fn, "w", newline="") as fh:
                    wr = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
                    wr.writeheader()
                    for row in rows:
                        wr.writerow({c: row.get(c, "") for c in COLS})
        print(f"backfill: filled {filled} rows from {len(seqs_u)} unique seqs -> {OUTDIR} | {time.time() - t0:.0f}s", flush=True)
        return

    # ---- PROBE: time ONE pair, project 24 pairs, exit -----------------------------
    if args.probe:
        pr = pairs[0]; name = f"{pr['scaffold_name']}-{pr['target_name']}"
        insts = build_trials(pr)
        print(f"probing 1 pair [{name}] | {args.trials} trials | window={len(insts[0]['editable'])} editable "
              f"| iters={args.iters} T={args.temp} k={args.k}", flush=True)
        te0 = time.time(); evaluate_round(insts, 0, do_ppl=True); t_ppl = time.time() - te0
        ti0 = time.time(); run_iteration(insts); t_iter = time.time() - ti0
        te1 = time.time(); evaluate_round(insts, 1, do_ppl=False); t_diag = time.time() - te1
        np_ = len(pairs)
        # 'all': every round has ppl.  'endpoints': only round 0 + final round have ppl.
        per_pair_all = t_ppl * (args.iters + 1) + t_iter * args.iters
        per_pair_ep = t_ppl * 2 + t_diag * (args.iters - 1) + t_iter * args.iters
        print("\n=== PROBE (one pair) ===", flush=True)
        print(f"  eval w/ ppl (surrogate+ppl, {args.trials} trials): {t_ppl:6.1f} s")
        print(f"  eval w/o ppl (surrogate only):                     {t_diag:6.1f} s")
        print(f"  per-iteration design pass (ESM-2 only):            {t_iter:6.1f} s  x {args.iters}")
        print(f"  --------------------------------------------------")
        print(f"  --ppl all       : ~{per_pair_all/60:.1f} min/pair -> {np_} pairs ~{per_pair_all*np_/60:.0f} min")
        print(f"  --ppl endpoints : ~{per_pair_ep/60:.1f} min/pair -> {np_} pairs ~{per_pair_ep*np_/60:.0f} min  (default)")
        print("  probe done; no CSVs written. Re-run without --probe to commit.", flush=True)
        return

    # ---- FULL RUN: one pair at a time ---------------------------------------------
    t_all = time.time()
    done = 0
    for pi, pr in enumerate(pairs):
        name = f"{pr['scaffold_name']}-{pr['target_name']}"
        fn = OUTDIR / f"design_{_san(name)}.csv"
        have, have_rounds, have_temp, have_k = existing_pair(fn)   # trial-level resume / posterior add
        if have >= args.trials:
            print(f"[{pi+1}/{len(pairs)}] {name}: cached {have}/{args.trials} trials -> skip", flush=True); continue
        if have > 0:                                      # appending to an existing CSV: warn on mismatch
            issues = []
            if have_rounds is not None and have_rounds != args.iters + 1:
                issues.append(f"iters {have_rounds-1}!=--iters {args.iters}")
            if _num(have_temp) is not None and _num(have_temp) != args.temp:
                issues.append(f"temp {have_temp}!=--temp {args.temp}")
            if _num(have_k) is not None and int(_num(have_k)) != args.k:
                issues.append(f"k {have_k}!=--k {args.k}")
            if issues:
                print(f"[{pi+1}/{len(pairs)}] {name}: WARNING appending with different settings "
                      f"({'; '.join(issues)}) -> CSV will mix trajectory shapes", flush=True)
        t0 = time.time()
        insts = build_trials(pr, trial_start=have, trial_end=args.trials)   # compute trials [have, N)
        evaluate_round(insts, 0, do_ppl=True)
        for r in range(1, args.iters + 1):
            run_iteration(insts)
            do_ppl = (args.ppl == "all") or (r == args.iters)
            evaluate_round(insts, r, do_ppl=do_ppl)
        write_pair(pr, name, fn, insts, append=(have > 0))
        done += 1
        best = min(t["hist"][-1]["peak_err"] for t in insts)
        mean_end = np.mean([t["hist"][-1]["peak_err"] for t in insts])
        scaf_err = insts[0]["hist"][0]["peak_err"]
        added = f"trials {have}..{args.trials-1} appended" if have else f"{len(insts)} trials"
        print(f"[{pi+1}/{len(pairs)}] {name}: {added} | scaffold err {scaf_err:.1f} -> "
              f"mean {mean_end:.1f} / best {best:.1f} nm (diag, over new trials) "
              f"| {len(insts[0]['editable'])} editable | {time.time()-t0:.0f}s", flush=True)
    print(f"done: {done} pairs updated -> {OUTDIR} | total {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
