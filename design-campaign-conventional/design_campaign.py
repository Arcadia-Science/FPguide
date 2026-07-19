#!/usr/bin/env python
"""Conventional design campaign: ESM-2 guided generation, ONE scaffold->target pair at a time.

We iterate over the 24 pairs (pairs/campaign_pairs_24.csv) sequentially. For each pair we run
``--trials`` independent design trials (default 6) starting from the scaffold sequence. All trials
of a pair share the IDENTICAL design window (design_windows_24.json: chromophore pos1 & pos2 + the
5 A pocket; pos2 restricted to aromatics {Y,W,H,F}; Gly + catalytic Arg/Glu fixed), but each trial
masks/edits those window positions in its OWN random order (any-order masking; a fresh random
permutation is drawn per trial per iteration).

Each iteration (default 3) is a masked-LM proposal + surrogate-guided selection over the window:
  at each visited position, ESM-2 gives the top-k allowed residues; the ALL-DATA surrogate
  (models/surrogate_cnn-max-d1_alldata.pt) predicts (ex, em) for every candidate; we score
  score = z(logp_ESM) - lam_ex*z(|d ex|) - lam_em*z(|d em|) and sample at temperature T.

The ProstT5 oracle (esm2_design/trained_models/oracle_sweep/cnn-max-d2_s0.pt) judges (ex, em) each
round; ESM-2 pseudo-perplexity is the naturalness diagnostic. Settings: T=10, k=10, lam=20.

ACCELERATION: within a pair the 6 trials advance together in one GPU batch (same window, so slot j
is genuinely the same set of positions across trials, differing only by each trial's random order);
fp16 autocast on CUDA, sub-batched forwards.

Trajectory: one CSV per pair (designs/design_<scaffold>-<target>.csv), one row per (trial, round);
round 0 = scaffold. Resumable at pair granularity (existing CSVs are skipped).

Usage
-----
    python design_campaign.py --probe          # time ONE pair (6 trials), project 24 pairs, EXIT
    python design_campaign.py                  # full run, one pair at a time
    python design_campaign.py --trials 6 --iters 3 --temp 10 --k 10
    python design_campaign.py --pairs-limit 2  # first 2 pairs only
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ESM2 = REPO / "esm2_design"
CUR = REPO / "dataset_pipeline" / "data" / "peak" / "curated"
sys.path.insert(0, str(ESM2))
import peak_models as pm            # noqa: E402
import prostt5_embed as pe          # noqa: E402

SURR_CKPT = HERE / "models" / "surrogate_cnn-max-d1_alldata.pt"
ORAC_CKPT = ESM2 / "trained_models" / "oracle_sweep" / "cnn-max-d2_s0.pt"
WINDOWS_JSON = HERE / "design_windows_24.json"
PAIRS_CSV = HERE / "pairs" / "campaign_pairs_24.csv"
OUTDIR = HERE / "designs"

SEED = 42
ESM_BS = 64            # sub-batch for ESM-2 embed / logits forwards
ORAC_BS = 32           # sub-batch for ProstT5 oracle
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
    ap.add_argument("--temp", type=float, default=10.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lam-ex", type=float, default=20.0)
    ap.add_argument("--lam-em", type=float, default=20.0)
    ap.add_argument("--pairs-limit", type=int, default=0, help="use only the first N pairs")
    ap.add_argument("--ppl", choices=["all", "endpoints"], default="endpoints",
                    help="compute ESM-2 pseudo-perplexity every round ('all') or only scaffold + final ('endpoints')")
    ap.add_argument("--probe", action="store_true",
                    help="time ONE pair (round-0 eval + 1 iteration), project the 24-pair total, and EXIT")
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

    # ---- models -------------------------------------------------------------------
    _sb, s_meta = pm.load_model(str(SURR_CKPT), dev); surrogate_net = pm.wrap(_sb, s_meta["mean"], s_meta["std"], dev)
    _ob, o_meta = pm.load_model(str(ORAC_CKPT), dev); oracle_net = pm.wrap(_ob, o_meta["mean"], o_meta["std"], dev)
    s_mae = s_meta.get("train_mae", s_meta.get("val_mae", float("nan")))
    print(f"surrogate (all-data) train MAE {s_mae:.1f} nm | oracle val MAE {o_meta.get('val_mae', float('nan')):.1f} nm", flush=True)
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
    def oracle_peaks_batched(seqlist, bs=ORAC_BS):
        outs = []
        for i in range(0, len(seqlist), bs):
            ch = seqlist[i:i + bs]; H, mask = pe.resid_embed_prostt5(ch, dev, bs=len(ch)); outs.append(oracle_net(H, mask))
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

    def _zc(t):
        return (t - t.mean()) / (t.std() + 1e-6)

    # ---- per-pair design (batches the pair's trials) ------------------------------
    def build_trials(pr):
        """Build the pair's trial instances (identical window, per-trial random-order RNG)."""
        si, ti = int(pr["scaffold_idx"]), int(pr["target_idx"])
        w = windows[pr["scaffold_name"]]
        editable = list(w["editable_0based"])
        pos_allowed = {int(p): _mask(aas) for p, aas in w["position_constraints"].items()}
        scaf = w["scaffold_seq"]
        tgt = torch.tensor(peaks[ti], device=dev)
        insts = []
        for trial in range(args.trials):
            insts.append(dict(si=si, ti=ti, editable=editable, pos_allowed=pos_allowed,
                              scaffold=scaf, seq=scaf, tgt=tgt, trial=trial,
                              rng=np.random.default_rng(SEED + si * 131 + trial * 17), hist=[]))
        return insts

    def evaluate_round(insts, r, do_ppl):
        seqlist = [t["seq"] for t in insts]
        P = oracle_peaks_batched(seqlist)
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
            cand_all = []; meta = []
            for i, t in enumerate(sub):
                pos = positions[i]
                mh = t["pos_allowed"].get(pos, AA_MASK)
                lg = logits[i].masked_fill(~mh, float("-inf"))
                logp = torch.log_softmax(lg, -1)
                k_eff = min(args.k, int(mh.sum().item()))
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
                scores = _zc(topv) - args.lam_ex * _zc(ex_err) - args.lam_em * _zc(em_err)
                ch = (int(torch.multinomial(torch.softmax(scores / args.temp, -1), 1).item())
                      if args.temp > 0 else int(torch.argmax(scores)))
                t["seq"] = t["seq"][:pos] + aas[ch] + t["seq"][pos + 1:]

    def write_pair(pr, name, fn, insts):
        si, ti = int(pr["scaffold_idx"]), int(pr["target_idx"])
        with open(fn, "w", newline="") as fh:
            wr = csv.writer(fh); wr.writerow(COLS)
            for t in sorted(insts, key=lambda x: x["trial"]):
                for hh in t["hist"]:
                    wr.writerow([name, pr["scaffold_name"], si, pr["scaffold_pdb"],
                                 pr["target_name"], ti, pr.get("selection", ""),
                                 f"{peaks[si,0]:.0f}", f"{peaks[si,1]:.0f}", f"{peaks[ti,0]:.0f}", f"{peaks[ti,1]:.0f}",
                                 pr["identity"], t["trial"], hh["round"], len(t["editable"]),
                                 args.temp, args.k, args.lam_ex, args.lam_em,
                                 f"{hh['pred_ex']:.1f}", f"{hh['pred_em']:.1f}", f"{hh['peak_err']:.2f}",
                                 f"{hh['ppl']:.2f}" if hh["ppl"] == hh["ppl"] else "", f"{hh['ident']:.3f}",
                                 hh["seq"], t["scaffold"], seqs[ti]])

    os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    # ---- PROBE: time ONE pair, project 24 pairs, exit -----------------------------
    if args.probe:
        pr = pairs[0]; name = f"{pr['scaffold_name']}-{pr['target_name']}"
        insts = build_trials(pr)
        print(f"probing 1 pair [{name}] | {args.trials} trials | window={len(insts[0]['editable'])} editable "
              f"| iters={args.iters} T={args.temp} k={args.k}", flush=True)
        te0 = time.time(); evaluate_round(insts, 0, do_ppl=True); t_ppl = time.time() - te0
        ti0 = time.time(); run_iteration(insts); t_iter = time.time() - ti0
        te1 = time.time(); evaluate_round(insts, 1, do_ppl=False); t_orac = time.time() - te1
        np_ = len(pairs)
        # 'all': every round has ppl.  'endpoints': only round 0 + final round have ppl.
        per_pair_all = t_ppl * (args.iters + 1) + t_iter * args.iters
        per_pair_ep = t_ppl * 2 + t_orac * (args.iters - 1) + t_iter * args.iters
        print("\n=== PROBE (one pair) ===", flush=True)
        print(f"  eval w/ ppl (oracle+ppl, {args.trials} trials): {t_ppl:6.1f} s")
        print(f"  eval w/o ppl (oracle only):                    {t_orac:6.1f} s")
        print(f"  per-iteration design pass:                     {t_iter:6.1f} s  x {args.iters}")
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
        if fn.exists():
            print(f"[{pi+1}/{len(pairs)}] {name}: cached -> skip", flush=True); continue
        t0 = time.time()
        insts = build_trials(pr)
        evaluate_round(insts, 0, do_ppl=True)
        for r in range(1, args.iters + 1):
            run_iteration(insts)
            do_ppl = (args.ppl == "all") or (r == args.iters)
            evaluate_round(insts, r, do_ppl=do_ppl)
        write_pair(pr, name, fn, insts)
        done += 1
        best = min(t["hist"][-1]["peak_err"] for t in insts)
        mean_end = np.mean([t["hist"][-1]["peak_err"] for t in insts])
        scaf_err = insts[0]["hist"][0]["peak_err"]
        print(f"[{pi+1}/{len(pairs)}] {name}: scaffold err {scaf_err:.1f} -> mean {mean_end:.1f} / best {best:.1f} nm "
              f"| {len(insts[0]['editable'])} editable | {time.time()-t0:.0f}s", flush=True)
    print(f"done: {done} pairs designed ({done*args.trials} trajectories) -> {OUTDIR} | total {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
