#!/usr/bin/env python
"""Shared design-campaign engine for the conventional FP design campaigns.

This is the common library extracted from the (previously duplicated) per-folder
``design_campaign.py`` drivers of the original conventional campaign, since archived
out of the repo (see the top-level README). Every campaign
runs the SAME procedure -- iterate over scaffold->target pairs, run ``--trials`` independent
any-order masked-LM design trials over a per-scaffold edit window, batch the trials on one GPU
forward, and log one CSV per pair (round 0 = scaffold) -- and differs ONLY in:

  * which design-window JSON + pairs CSV it loads,                      (config)
  * the per-position SELECTION rule, and                               (``strategy``)
  * a few surface knobs (default T/k, whether a surrogate-MAE term or  (config flags)
    lambda columns exist, whether resume is at pair or trial granularity).

Two selection strategies are provided, ported verbatim from the original drivers so results are
bit-for-bit reproducible:

  ``guided``  surrogate-guided generation. At each visited position ESM-2 gives the top-k allowed
              residues; the ALL-DATA (ex, em) surrogate scores every candidate and we sample
              score = z(logp_ESM) - lam_ex*z(|d ex|) - lam_em*z(|d em|) at temperature T (T=10,
              lam=20). Selection draws from the GLOBAL torch RNG (seeded once). OPTIONALLY a
              second head -- a classifier-predicted brightness model (``cfg.brightness_ckpt``,
              opt-in) -- adds a ``+ lam_bright*z(pred_bright)`` term so the search is steered toward
              BRIGHT designs as well as on-spectrum ones; that head shares the same ESM-2 embedding
              as the surrogate (no extra embed cost) and its per-design prediction is logged in the
              ``pred_bright`` column.

  ``gibbs``   pure ESM-2 masked-LM Gibbs sampling. We sample DIRECTLY from the top-k masked-LM
              conditional softmax(logp / T) at T=1 -- a true Gibbs draw of p(x_i | x_{-i}); the
              surrogate is loaded but only records (ex, em) as a diagnostic (never steers search).
              Selection draws from a PER-TRIAL torch.Generator so a trial is reproducible no matter
              when/alongside how many others it is drawn (enables trial-granularity resume).

There is NO oracle for these campaigns (the surrogate is trained on ALL data; the real judge is
experiment). ``pred_ex/pred_em`` are the surrogate's own (ex, em) and ``peak_err`` its distance to
the target. ESM-2 pseudo-perplexity is a naturalness diagnostic.

ACCELERATION: within a pair the trials advance together in one GPU batch (identical window, so slot
j is the same set of positions across trials, differing only by each trial's random order); fp16
autocast on CUDA, sub-batched forwards. Full-sequence ppl is the dominant cost, so ``--ppl
endpoints`` computes it only at the scaffold + final round and the blanks are filled afterwards by
``--backfill-ppl`` (dedupes sequences across all pair CSVs, computes each unique one once).

A thin per-folder ``design_campaign.py`` just builds a :class:`CampaignConfig` and calls
:func:`run`; that keeps each campaign's CLI (``--probe``, ``--ppl``, ``--backfill-ppl``, etc.)
identical to before.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent       # .../fpdesign
REPO = HERE.parent                            # .../spectrum-to-fp-design
CUR = REPO / "dataset_pipeline" / "data" / "peak" / "curated"
from . import peak_models as pm     # noqa: E402  (shared ESM-2 + surrogate/oracle model utilities)

# Default surrogate checkpoint now lives inside the library so every campaign shares one weight.
DEFAULT_SURROGATE = HERE / "models" / "surrogate_cnn-max-d1_alldata.pt"

# Optional brightness head for guided campaigns that also steer on predicted brightness. Any
# peak_models out=1 checkpoint works -- a regression head bakes train-split (mean, std) into its
# metadata; a CLASSIFIER head (e.g. bright/dim) carries none and is handled as identity below. Swap
# this for the finalized weight when ready. The default is the sub40k bright/dim classifier
# (cnn-max-d2, val AUROC 0.98) so the workflow is runnable end-to-end today.
DEFAULT_BRIGHTNESS = HERE / "models" / "brightness_cnn-max-d2_40k.pt"

SEED = 42
ESM_BS = 64            # sub-batch for ESM-2 embed / logits forwards
PPL_BS = 128           # sub-batch for pseudo-perplexity single-mask rows

COLS = ["pair", "scaffold_name", "scaffold_idx", "scaffold_pdb", "target_name", "target_idx",
        "selection", "scaffold_ex", "scaffold_em", "target_ex", "target_em", "seq_id_scaf_target",
        "trial", "round", "n_editable", "temp", "k", "lam_ex", "lam_em",
        "pred_ex", "pred_em", "peak_err", "ppl", "ident_to_scaffold",
        "designed_seq", "scaffold_seq", "target_seq"]

# Target-free schema (cfg.target_free, e.g. gibbs): the search never uses a target, so a run is one
# effort PER SCAFFOLD (not per scaffold->target pair). All target columns are dropped; pred_ex/pred_em
# remain as the surrogate's diagnostic (ex, em) for the design's OWN predicted peaks.
COLS_FREE = ["scaffold_name", "scaffold_idx", "scaffold_pdb", "selection",
             "scaffold_ex", "scaffold_em", "trial", "round", "n_editable", "temp", "k",
             "pred_ex", "pred_em", "ppl", "ident_to_scaffold", "designed_seq", "scaffold_seq"]


def _san(s):
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in s)


def _zc(t):
    return (t - t.mean()) / (t.std() + 1e-6)


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


@dataclass
class CampaignConfig:
    """Everything that distinguishes one campaign from another. Paths are absolute ``Path``s."""
    name: str
    strategy: str                              # "guided" | "gibbs"
    windows_json: Path
    pairs_csv: Path
    outdir: Path
    surrogate_ckpt: Path = DEFAULT_SURROGATE
    default_temp: float = 10.0
    default_k: int = 10
    default_lam_ex: float = 20.0
    default_lam_em: float = 20.0
    # optional classifier-predicted-brightness guidance (guided only). When brightness_ckpt is set,
    # each candidate's predicted brightness adds a + lam_bright*z(pred_bright) term to the guided
    # score, steering toward brighter designs; its per-design value is logged in the pred_bright
    # column. Leave brightness_ckpt=None to preserve the peaks-only guided behaviour exactly.
    brightness_ckpt: Path | None = None
    default_lam_bright: float = 20.0
    add_lam_bright_arg: bool = False           # expose --lam-bright (brightness-guided campaigns)
    record_brightness: bool = False            # append pred_bright + lam_bright columns to the CSV
    # optional edit penalty (guided only): subtract lam_edit*z(is_edit) from each candidate's score,
    # where is_edit=1 if the candidate residue differs from the SCAFFOLD residue at that position and 0
    # if it keeps it. This steers the search toward FEWER mutations from the scaffold (staying closer
    # to the -- typically bright, well-folded -- parent). Only bites where the scaffold residue is
    # itself among the ESM top-k allowed candidates; forced positions (e.g. pos2->aromatic) are
    # unaffected. lam_edit=0 preserves the original behaviour exactly.
    default_lam_edit: float = 0.0
    add_lam_edit_arg: bool = False             # expose --lam-edit (guided only)
    # self-documenting output dir (guided brightness/edit campaigns): append the run's guidance
    # weights to ``outdir`` so each lam setting writes to its OWN folder, e.g. outdir="designs"
    # -> "designs_lam-bright60_lam-edit10". The suffix is derived from the parsed args, so CLI
    # overrides (--lam-bright/--lam-edit) are reflected and a plain default run is unambiguous.
    # Resume/rescore/backfill still line up because identical args resolve to the same folder.
    outdir_lambda_suffix: bool = False
    # surface flags (keep each folder's CLI + CSV identical to its original hand-written driver)
    add_lam_args: bool = True                  # expose --lam-ex/--lam-em (guided only)
    add_rescore: bool = True                   # expose --rescore (guided only)
    record_lambda: bool = True                 # write lam_ex/lam_em columns (else left blank)
    trial_resume: bool = False                 # resume/append at trial granularity (gibbs); else pair-level skip
    target_free: bool = False                  # the search ignores the target (gibbs): iterate over UNIQUE
                                               # scaffolds (not pairs), write one design_<scaffold>.csv per
                                               # scaffold with the target-free COLS_FREE schema. The four
                                               # per-target CSVs were byte-identical apart from the target
                                               # columns, so this keeps a single design effort per scaffold.
    per_trial_rng: bool = False                # guided: draw the per-position choice from each trial's
                                               # OWN torch.Generator (seeded per trial) instead of the
                                               # global RNG. Makes a trial bit-reproducible regardless of
                                               # how many run together -> guided becomes trial-resumable
                                               # (pair with trial_resume=True). gibbs ALWAYS uses the
                                               # per-trial generator, so this flag only affects guided.
                                               # NB: changes guided's numeric results vs the global-RNG
                                               # default, so leave False to preserve legacy campaigns.
    description: str = ""

    def __post_init__(self):
        if self.strategy not in ("guided", "gibbs"):
            raise ValueError(f"strategy must be 'guided' or 'gibbs', got {self.strategy!r}")
        if self.brightness_ckpt is not None and self.strategy != "guided":
            raise ValueError("brightness_ckpt is only supported with strategy='guided' "
                             f"(it steers the guided score); got strategy={self.strategy!r}")


def build_argparser(cfg: CampaignConfig) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=cfg.description or f"{cfg.name} design campaign")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--temp", type=float, default=cfg.default_temp)
    ap.add_argument("--k", type=int, default=cfg.default_k)
    if cfg.add_lam_args:
        ap.add_argument("--lam-ex", type=float, default=cfg.default_lam_ex)
        ap.add_argument("--lam-em", type=float, default=cfg.default_lam_em)
    if cfg.add_lam_bright_arg:
        ap.add_argument("--lam-bright", type=float, default=cfg.default_lam_bright,
                        help="weight of the z(predicted-brightness) term in the guided score "
                             "(steers toward brighter designs)")
    if cfg.add_lam_edit_arg:
        ap.add_argument("--lam-edit", type=float, default=cfg.default_lam_edit,
                        help="weight of the z(is_edit) penalty in the guided score (steers toward "
                             "FEWER mutations from the scaffold; 0 disables)")
    ap.add_argument("--pairs-limit", type=int, default=0, help="use only the first N pairs")
    ap.add_argument("--pairs", type=str, default="",
                    help="comma-separated target (or scaffold) names to keep, e.g. 'EBFP,mOrange'; "
                         "default all. Applied before --pairs-limit; lets you pick non-contiguous pairs.")
    ap.add_argument("--ppl", choices=["all", "endpoints"], default="endpoints",
                    help="compute ESM-2 pseudo-perplexity every round ('all') or only scaffold + final ('endpoints')")
    ap.add_argument("--probe", action="store_true",
                    help="time ONE pair (round-0 eval + 1 iteration), project the total, and EXIT")
    if cfg.add_rescore:
        ap.add_argument("--rescore", action="store_true",
                        help="do not design; re-fill pred_ex/pred_em/peak_err of existing designs/*.csv with the surrogate")
    ap.add_argument("--backfill-ppl", action="store_true",
                    help="do NOT design; fill empty ppl cells in existing designs/*.csv. ppl depends only "
                         "on the sequence, so unique sequences are computed once (deduped) in large batches "
                         "and the values fanned out to every matching row. Pairs with '--ppl endpoints'.")
    return ap


class Campaign:
    """Holds the loaded models + dataset and runs the campaign described by ``cfg`` with ``args``."""

    def __init__(self, cfg: CampaignConfig, args):
        self.cfg = cfg
        self.args = args
        self.outdir = Path(cfg.outdir)
        if cfg.outdir_lambda_suffix:
            # encode the run's guidance weights in the folder name so each lam setting is pinned to
            # its own self-documenting directory. Only the ACTIVE terms are appended: brightness
            # campaigns -> designs_lam-bright60_lam-edit10; a peaks-only + edit-penalty campaign
            # (no brightness model) -> designs_lam-edit10.
            le = getattr(args, "lam_edit", cfg.default_lam_edit)
            parts = []
            if cfg.brightness_ckpt is not None:
                lb = getattr(args, "lam_bright", cfg.default_lam_bright)
                parts.append(f"lam-bright{lb:g}")
            parts.append(f"lam-edit{le:g}")
            self.outdir = self.outdir.with_name(self.outdir.name + "_" + "_".join(parts))

        self.dev = (torch.device("cuda") if torch.cuda.is_available()
                    else torch.device("mps") if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
                    else torch.device("cpu"))
        self.use_fp16 = (self.dev.type == "cuda")
        self._amp = ((lambda: torch.autocast("cuda", dtype=torch.float16)) if self.use_fp16
                     else (lambda: nullcontext()))
        print(f"device {self.dev} | fp16 {self.use_fp16}", flush=True)

        self.rows, self.seqs, self.peaks = load_dataset()
        self.windows = json.load(open(cfg.windows_json))["windows"]
        pairs = list(csv.DictReader(open(cfg.pairs_csv)))
        want = (getattr(args, "pairs", "") or "").strip()
        if want:
            keep = {s.strip() for s in want.split(",") if s.strip()}
            pairs = [pr for pr in pairs
                     if pr.get("target_name") in keep or pr.get("scaffold_name") in keep]
            if not pairs:
                raise SystemExit(f"--pairs {want!r} matched no rows in {cfg.pairs_csv}")
        if cfg.target_free:
            # target-independent search: one design effort per UNIQUE scaffold (dedupe the pairs list)
            seen, units = set(), []
            for pr in pairs:
                if pr["scaffold_name"] in seen:
                    continue
                seen.add(pr["scaffold_name"]); units.append(pr)
            pairs = units
        if args.pairs_limit:
            pairs = pairs[:args.pairs_limit]
        self.units = pairs

        # ---- models: surrogate (guides in 'guided', diagnostic-only in 'gibbs') + ESM-2 ----
        _sb, s_meta = pm.load_model(str(cfg.surrogate_ckpt), self.dev)
        self.surrogate_net = pm.wrap(_sb, s_meta["mean"], s_meta["std"], self.dev)
        s_mae = s_meta.get("train_mae", s_meta.get("val_mae", float("nan")))
        role = "guides search" if cfg.strategy == "guided" else "diagnostic only"
        print(f"surrogate (all-data, {role}) train MAE {s_mae:.1f} nm", flush=True)

        # ---- optional brightness classifier (adds a z(pred_bright) term to the guided score) ----
        self.brightness_net = None
        if cfg.brightness_ckpt is not None:
            ckpt = Path(cfg.brightness_ckpt)
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"brightness classifier checkpoint not found: {ckpt}\n"
                    "This campaign guides on ex/em peaks + a classifier-predicted brightness. Train "
                    "the brightness model (a peak_models out=1 checkpoint whose metadata carries its "
                    "architecture + train-split mean/std) and drop it at that path, or point "
                    "cfg.brightness_ckpt elsewhere.")
            _bb, b_meta = pm.load_model(str(ckpt), self.dev, out=1)
            # A regression head bakes (mean, std) to de-standardize to physical units; a brightness
            # CLASSIFIER emits a raw logit and carries no scaler, so fall back to identity (0, 1).
            # The guided score z-scores pred_bright across candidates, so any affine transform is
            # irrelevant to the ranking either way.
            b_mean = b_meta.get("mean", 0.0)
            b_std = b_meta.get("std", 1.0)
            self.brightness_net = pm.wrap(_bb, b_mean, b_std, self.dev)
            b_arch = f"{b_meta.get('arch','?')}-{b_meta.get('pool','?')}-d{b_meta.get('depth','?')}"
            if "val_auroc" in b_meta:                # classifier head: report ranking quality, not MAE
                print(f"brightness classifier ({b_arch}, guides search) val AUROC "
                      f"{b_meta['val_auroc']:.3f}", flush=True)
            else:
                b_mae = b_meta.get("val_mae", b_meta.get("train_mae", float("nan")))
                b_unit = b_meta.get("unit", "brightness")
                print(f"brightness model ({b_arch}, guides search) val MAE {b_mae:.3f} {b_unit}",
                      flush=True)

        self.esm_model, self.alphabet, self.bc = pm.get_esm(self.dev)

        self.AA_MASK = self._mask("ACDEFGHIKLMNPQRSTVWY")
        self.SPECIAL = {self.alphabet.cls_idx, self.alphabet.eos_idx,
                        self.alphabet.padding_idx, self.alphabet.mask_idx}
        self._select = self._select_guided if cfg.strategy == "guided" else self._select_gibbs
        # CSV schema: the peaks-only campaigns keep COLS exactly; brightness-guided ones append
        # pred_bright + lam_bright so downstream tools that read by column name are unaffected.
        self.cols = list(COLS) + (["pred_bright", "lam_bright"] if cfg.record_brightness else [])

    # ---- tokenization / batched model forwards ------------------------------------
    def _mask(self, aas):
        m = torch.zeros(len(self.alphabet.all_toks), dtype=torch.bool)
        for a in aas:
            m[self.alphabet.get_idx(a)] = True
        return m.to(self.dev)

    @torch.no_grad()
    def _esm_embed(self, seqlist):
        _, _, tk = self.bc([(f"s{i}", s) for i, s in enumerate(seqlist)]); tk = tk.to(self.dev)
        with self._amp():
            reps = self.esm_model(tk, repr_layers=[pm.ESM_LAYER])["representations"][pm.ESM_LAYER]
        Lmax = max(len(s) for s in seqlist)
        H = torch.zeros(len(seqlist), Lmax, pm.D_IN, device=self.dev)
        mask = torch.zeros(len(seqlist), Lmax, dtype=torch.bool, device=self.dev)
        for j, s in enumerate(seqlist):
            n = len(s); H[j, :n] = reps[j, 1:1 + n].float(); mask[j, :n] = True
        return H, mask

    @torch.no_grad()
    def surrogate_peaks_batched(self, seqlist, bs=ESM_BS):
        outs = []
        for i in range(0, len(seqlist), bs):
            H, mask = self._esm_embed(seqlist[i:i + bs]); outs.append(self.surrogate_net(H, mask))
        return torch.cat(outs, 0)

    @torch.no_grad()
    def peaks_and_brightness_batched(self, seqlist, bs=ESM_BS):
        """Embed each sequence ONCE and run the (ex, em) surrogate plus the optional brightness head.

        Returns ``(P, B)`` where ``P`` is (N, 2) ex/em in nm and ``B`` is (N, 1) predicted brightness
        (or ``None`` when no brightness classifier is configured). Sharing the ESM-2 embedding keeps
        the brightness term from doubling the dominant embedding cost, and gives numerically
        identical peaks to ``surrogate_peaks_batched`` so peaks-only campaigns are unaffected."""
        peaks, bright = [], []
        for i in range(0, len(seqlist), bs):
            H, mask = self._esm_embed(seqlist[i:i + bs])
            peaks.append(self.surrogate_net(H, mask))
            if self.brightness_net is not None:
                bright.append(self.brightness_net(H, mask))
        P = torch.cat(peaks, 0)
        B = torch.cat(bright, 0) if bright else None
        return P, B

    @torch.no_grad()
    def esm_logits_at(self, seqlist, positions, bs=ESM_BS):
        outs = []
        for i in range(0, len(seqlist), bs):
            sl = seqlist[i:i + bs]; ps = positions[i:i + bs]
            _, _, tk = self.bc([(f"s{j}", s) for j, s in enumerate(sl)]); tk = tk.to(self.dev)
            rr = torch.arange(len(sl), device=self.dev)
            cols = torch.tensor([p + 1 for p in ps], device=self.dev)
            tk[rr, cols] = self.alphabet.mask_idx
            with self._amp():
                lg = self.esm_model(tk)["logits"]
            outs.append(lg[rr, cols].float())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def ppl_batched(self, seqlist, bs=PPL_BS):
        jobs = []
        for s in seqlist:
            _, _, base = self.bc([("s", s.strip().upper())]); base = base[0]
            respos = [i for i, tok in enumerate(base.tolist()) if tok not in self.SPECIAL]
            jobs.append((base, respos))
        rows_meta = [(si, base, p) for si, (base, respos) in enumerate(jobs) for p in respos]
        tot = torch.zeros(len(seqlist), device=self.dev)
        for b in range(0, len(rows_meta), bs):
            chunk = rows_meta[b:b + bs]
            Lmax = max(base.numel() for _, base, _ in chunk)
            bt = torch.full((len(chunk), Lmax), self.alphabet.padding_idx, dtype=torch.long)
            cols = torch.zeros(len(chunk), dtype=torch.long)
            truth = torch.zeros(len(chunk), dtype=torch.long)
            sidx = torch.zeros(len(chunk), dtype=torch.long)
            for r, (si, base, p) in enumerate(chunk):
                L = base.numel(); bt[r, :L] = base; bt[r, p] = self.alphabet.mask_idx
                cols[r] = p; truth[r] = base[p]; sidx[r] = si
            bt = bt.to(self.dev)
            with self._amp():
                lp = torch.log_softmax(self.esm_model(bt)["logits"], -1)
            rr = torch.arange(len(chunk), device=self.dev)
            vals = lp[rr, cols.to(self.dev), truth.to(self.dev)].float()
            tot.index_add_(0, sidx.to(self.dev), vals)
        nsn = torch.tensor([len(rp) for _, rp in jobs], device=self.dev, dtype=torch.float).clamp(min=1)
        return torch.exp(-tot / nsn).cpu().tolist()

    # ---- per-pair design ----------------------------------------------------------
    def build_trials(self, pr, trial_start=0, trial_end=None):
        """Build the pair's trial instances for trials [trial_start, trial_end).

        Both RNGs are seeded PER TRIAL from (SEED, scaffold_idx, trial) so trial k is identical
        regardless of how many trials are requested or when it is computed. The numpy RNG drives
        the any-order visiting permutation (used by both strategies); the torch.Generator drives
        gibbs' per-position sampling (and guided's too when cfg.per_trial_rng is set; otherwise
        guided samples from the global torch RNG and is not trial-resumable)."""
        trial_end = self.args.trials if trial_end is None else trial_end
        si = int(pr["scaffold_idx"])
        w = self.windows[pr["scaffold_name"]]
        editable = list(w["editable_0based"])
        pos_allowed = {int(p): self._mask(aas) for p, aas in w["position_constraints"].items()}
        scaf = w["scaffold_seq"]
        if self.cfg.target_free:                       # no target -> no (ex, em) goal, no peak_err
            ti, tgt = None, None
        else:
            ti = int(pr["target_idx"]); tgt = torch.tensor(self.peaks[ti], device=self.dev)
        insts = []
        for trial in range(trial_start, trial_end):
            seed = SEED + si * 131 + trial * 17
            insts.append(dict(si=si, ti=ti, editable=editable, pos_allowed=pos_allowed,
                              scaffold=scaf, seq=scaf, tgt=tgt, trial=trial,
                              rng=np.random.default_rng(seed),
                              gen=torch.Generator(device=self.dev).manual_seed(seed), hist=[]))
        return insts

    def evaluate_round(self, insts, r, do_ppl):
        seqlist = [t["seq"] for t in insts]
        P, B = self.peaks_and_brightness_batched(seqlist)
        ppls = self.ppl_batched(seqlist) if do_ppl else [float("nan")] * len(insts)
        for i, t in enumerate(insts):
            ex, em = float(P[i, 0]), float(P[i, 1])
            err = (0.5 * (abs(ex - float(t["tgt"][0])) + abs(em - float(t["tgt"][1])))
                   if t["tgt"] is not None else float("nan"))
            ident = sum(x == y for x, y in zip(t["seq"], t["scaffold"])) / len(t["scaffold"])
            t["hist"].append(dict(round=r, pred_ex=ex, pred_em=em, peak_err=err,
                                  ppl=ppls[i], ident=ident, seq=t["seq"],
                                  pred_bright=(float(B[i, 0]) if B is not None else float("nan"))))

    def run_iteration(self, insts):
        # each trial: fresh random visiting order over the (identical) window this iteration
        for t in insts:
            t["_ed"] = list(t["rng"].permutation(t["editable"]))
        maxlen = max(len(t["_ed"]) for t in insts)
        for j in range(maxlen):
            sub = [t for t in insts if j < len(t["_ed"])]
            positions = [int(t["_ed"][j]) for t in sub]
            logits = self.esm_logits_at([t["seq"] for t in sub], positions)
            self._select(sub, positions, logits)

    def _select_guided(self, sub, positions, logits):
        """Surrogate-guided top-k selection. Uses the global torch RNG by default, or each trial's
        own Generator when cfg.per_trial_rng is set (making guided trial-reproducible/resumable)."""
        args = self.args
        per_trial = self.cfg.per_trial_rng
        cand_all = []; meta = []
        for i, t in enumerate(sub):
            pos = positions[i]
            mh = t["pos_allowed"].get(pos, self.AA_MASK)
            lg = logits[i].masked_fill(~mh, float("-inf"))
            logp = torch.log_softmax(lg, -1)
            k_eff = min(args.k, int(mh.sum().item()))
            topv, topi = torch.topk(logp, k_eff)
            aas = [self.alphabet.get_tok(int(x)) for x in topi.tolist()]
            for aa in aas:
                cand_all.append(t["seq"][:pos] + aa + t["seq"][pos + 1:])
            meta.append((t, pos, topv, aas, k_eff))
        Pk, Bk = self.peaks_and_brightness_batched(cand_all)
        lam_bright = getattr(args, "lam_bright", self.cfg.default_lam_bright)
        lam_edit = getattr(args, "lam_edit", self.cfg.default_lam_edit)
        off = 0
        for (t, pos, topv, aas, k_eff) in meta:
            sl = slice(off, off + k_eff); off += k_eff
            Pc = Pk[sl]
            ex_err = (Pc[:, 0] - t["tgt"][0]).abs(); em_err = (Pc[:, 1] - t["tgt"][1]).abs()
            scores = _zc(topv) - args.lam_ex * _zc(ex_err) - args.lam_em * _zc(em_err)
            if Bk is not None:                       # steer toward brighter candidates (maximize)
                scores = scores + lam_bright * _zc(Bk[sl, 0])
            if lam_edit:                             # penalize candidates that differ from the scaffold
                scaf_aa = t["scaffold"][pos]
                is_edit = torch.tensor([0.0 if aa == scaf_aa else 1.0 for aa in aas], device=scores.device)
                scores = scores - lam_edit * _zc(is_edit)
            gen = t["gen"] if per_trial else None
            ch = (int(torch.multinomial(torch.softmax(scores / args.temp, -1), 1, generator=gen).item())
                  if args.temp > 0 else int(torch.argmax(scores)))
            t["seq"] = t["seq"][:pos] + aas[ch] + t["seq"][pos + 1:]

    def _select_gibbs(self, sub, positions, logits):
        """Pure ESM-2 masked-LM Gibbs draw from the top-k conditional (per-trial Generator).

        With lam_edit=0 (default) this is an EXACT Gibbs draw: softmax(topv / T) at T=1 ==
        p(x_i | x_{-i})|topk. When --lam-edit > 0 is supplied, an opt-in edit penalty biases the
        draw toward the scaffold residue by z-scoring the top-k log-probs and subtracting
        lam_edit*z(is_edit) -- mirroring the guided edit penalty so the same lambda scale applies.
        This is no longer a pure Gibbs draw (by design); lam_edit=0 preserves the original exactly."""
        args = self.args
        lam_edit = getattr(args, "lam_edit", self.cfg.default_lam_edit)
        for i, t in enumerate(sub):
            pos = positions[i]
            mh = t["pos_allowed"].get(pos, self.AA_MASK)
            lg = logits[i].masked_fill(~mh, float("-inf"))
            logp = torch.log_softmax(lg, -1)
            k_eff = min(args.k, int(mh.sum().item()))
            topv, topi = torch.topk(logp, k_eff)
            aas = [self.alphabet.get_tok(int(x)) for x in topi.tolist()]
            scores = topv     # RAW top-k log-probs: softmax(scores / T) at T=1 == p(x_i | x_{-i})|topk
            if lam_edit:      # opt-in: penalize candidates that differ from the scaffold (z-scored)
                scaf_aa = t["scaffold"][pos]
                is_edit = torch.tensor([0.0 if aa == scaf_aa else 1.0 for aa in aas], device=scores.device)
                scores = _zc(topv) - lam_edit * _zc(is_edit)
            ch = (int(torch.multinomial(torch.softmax(scores / args.temp, -1), 1, generator=t["gen"]).item())
                  if args.temp > 0 else int(torch.argmax(scores)))
            t["seq"] = t["seq"][:pos] + aas[ch] + t["seq"][pos + 1:]

    # ---- trial-level resume helpers (used when cfg.trial_resume) -------------------
    @staticmethod
    def existing_pair(fn):
        """Returns (n_trials, rounds_per_trial, temp, k) recorded in an existing pair CSV."""
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

    @staticmethod
    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def write_pair(self, pr, name, fn, insts, append=False):
        args = self.args
        si = int(pr["scaffold_idx"])
        if self.cfg.target_free:                       # one target-free effort per scaffold
            with open(fn, "a" if append else "w", newline="") as fh:
                wr = csv.writer(fh)
                if not append:
                    wr.writerow(COLS_FREE)
                for t in sorted(insts, key=lambda x: x["trial"]):
                    for hh in t["hist"]:
                        wr.writerow([pr["scaffold_name"], si, pr["scaffold_pdb"], pr.get("selection", ""),
                                     f"{self.peaks[si,0]:.0f}", f"{self.peaks[si,1]:.0f}",
                                     t["trial"], hh["round"], len(t["editable"]), args.temp, args.k,
                                     f"{hh['pred_ex']:.1f}", f"{hh['pred_em']:.1f}",
                                     f"{hh['ppl']:.2f}" if hh["ppl"] == hh["ppl"] else "",
                                     f"{hh['ident']:.3f}", hh["seq"], t["scaffold"]])
            return
        ti = int(pr["target_idx"])
        lam_ex = f"{args.lam_ex}" if self.cfg.record_lambda else ""
        lam_em = f"{args.lam_em}" if self.cfg.record_lambda else ""
        lam_bright = (f"{getattr(args, 'lam_bright', self.cfg.default_lam_bright)}"
                      if self.cfg.record_brightness else "")
        with open(fn, "a" if append else "w", newline="") as fh:
            wr = csv.writer(fh)
            if not append:
                wr.writerow(self.cols)
            for t in sorted(insts, key=lambda x: x["trial"]):
                for hh in t["hist"]:
                    row = [name, pr["scaffold_name"], si, pr["scaffold_pdb"],
                           pr["target_name"], ti, pr.get("selection", ""),
                           f"{self.peaks[si,0]:.0f}", f"{self.peaks[si,1]:.0f}",
                           f"{self.peaks[ti,0]:.0f}", f"{self.peaks[ti,1]:.0f}",
                           pr["identity"], t["trial"], hh["round"], len(t["editable"]),
                           args.temp, args.k, lam_ex, lam_em,
                           f"{hh['pred_ex']:.1f}", f"{hh['pred_em']:.1f}", f"{hh['peak_err']:.2f}",
                           f"{hh['ppl']:.2f}" if hh["ppl"] == hh["ppl"] else "", f"{hh['ident']:.3f}",
                           hh["seq"], t["scaffold"], self.seqs[ti]]
                    if self.cfg.record_brightness:
                        pb = hh.get("pred_bright", float("nan"))
                        row += [f"{pb:.4f}" if pb == pb else "", lam_bright]
                    wr.writerow(row)

    # ---- top-level modes ----------------------------------------------------------
    def rescore(self):
        import glob
        files = sorted(glob.glob(str(self.outdir / "design_*.csv")))
        if not files:
            print(f"no CSVs found in {self.outdir}; nothing to rescore"); return
        print(f"rescoring {len(files)} CSVs with the surrogate (no design loop) ...", flush=True)
        for fn in files:
            with open(fn) as fh:
                rd = list(csv.DictReader(fh)); hdr = rd[0].keys() if rd else []
            if not rd:
                continue
            seqlist = [r["designed_seq"] for r in rd]
            P = self.surrogate_peaks_batched(seqlist)
            for i, r in enumerate(rd):
                ex, em = float(P[i, 0]), float(P[i, 1])
                r["pred_ex"] = f"{ex:.1f}"; r["pred_em"] = f"{em:.1f}"
                r["peak_err"] = f"{0.5*(abs(ex-float(r['target_ex']))+abs(em-float(r['target_em']))):.2f}"
            with open(fn, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(hdr)); wr.writeheader(); wr.writerows(rd)
            print(f"  rescored {os.path.basename(fn)} ({len(rd)} rows)", flush=True)
        print("rescore done.", flush=True)

    def backfill_ppl(self):
        files = sorted(self.outdir.glob("design_*.csv"))
        parsed, need = [], set()
        for fn in files:
            with open(fn, newline="") as fh:
                rdr = csv.DictReader(fh); rd = list(rdr); hdr = rdr.fieldnames
            parsed.append((fn, rd, hdr))
            for row in rd:
                if row.get("ppl", "").strip() == "":
                    need.add(row["designed_seq"])
        if not need:
            print(f"backfill: scanned {len(files)} files, no empty ppl cells -> nothing to do.", flush=True); return
        seqs_u = sorted(need)
        print(f"backfill: {len(seqs_u)} unique sequences need ppl across {len(files)} files", flush=True)
        t0 = time.time(); vals = {}
        for i in range(0, len(seqs_u), 1024):
            chunk = seqs_u[i:i + 1024]
            for s, p in zip(chunk, self.ppl_batched(chunk, bs=256)):
                vals[s] = p
            print(f"  ppl {min(i + 1024, len(seqs_u))}/{len(seqs_u)} ({time.time() - t0:.0f}s)", flush=True)
        filled = 0
        for fn, rd, hdr in parsed:
            changed = False
            for row in rd:
                if row.get("ppl", "").strip() == "":
                    p = vals.get(row["designed_seq"])
                    if p is not None and p == p:
                        row["ppl"] = f"{p:.2f}"; filled += 1; changed = True
            if changed:
                cols = hdr or COLS
                with open(fn, "w", newline="") as fh:
                    wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                    wr.writeheader()
                    for row in rd:
                        wr.writerow({c: row.get(c, "") for c in cols})
        print(f"backfill: filled {filled} rows from {len(seqs_u)} unique seqs -> {self.outdir} | {time.time() - t0:.0f}s", flush=True)

    def probe(self):
        args = self.args
        pr = self.units[0]
        name = pr["scaffold_name"] if self.cfg.target_free else f"{pr['scaffold_name']}-{pr['target_name']}"
        unit = "scaffold" if self.cfg.target_free else "pair"
        insts = self.build_trials(pr)
        print(f"probing 1 {unit} [{name}] | {args.trials} trials | window={len(insts[0]['editable'])} editable "
              f"| iters={args.iters} T={args.temp} k={args.k}", flush=True)
        te0 = time.time(); self.evaluate_round(insts, 0, do_ppl=True); t_ppl = time.time() - te0
        ti0 = time.time(); self.run_iteration(insts); t_iter = time.time() - ti0
        te1 = time.time(); self.evaluate_round(insts, 1, do_ppl=False); t_orac = time.time() - te1
        np_ = len(self.units)
        per_pair_all = t_ppl * (args.iters + 1) + t_iter * args.iters
        per_pair_ep = t_ppl * 2 + t_orac * (args.iters - 1) + t_iter * args.iters
        print("\n=== PROBE (one pair) ===", flush=True)
        print(f"  eval w/ ppl (surrogate+ppl, {args.trials} trials): {t_ppl:6.1f} s")
        print(f"  eval w/o ppl (surrogate only):                    {t_orac:6.1f} s")
        print(f"  per-iteration design pass:                        {t_iter:6.1f} s  x {args.iters}")
        print(f"  --------------------------------------------------")
        print(f"  --ppl all       : ~{per_pair_all/60:.1f} min/pair -> {np_} pairs ~{per_pair_all*np_/60:.0f} min")
        print(f"  --ppl endpoints : ~{per_pair_ep/60:.1f} min/pair -> {np_} pairs ~{per_pair_ep*np_/60:.0f} min  (default)")
        print("  probe done; no CSVs written. Re-run without --probe to commit.", flush=True)

    def run_full(self):
        args = self.args
        t_all = time.time()
        done = 0
        n_units = len(self.units)
        unit_word = "scaffolds" if self.cfg.target_free else "pairs"
        for pi, pr in enumerate(self.units):
            name = pr["scaffold_name"] if self.cfg.target_free else f"{pr['scaffold_name']}-{pr['target_name']}"
            fn = self.outdir / f"design_{_san(name)}.csv"
            if self.cfg.trial_resume:
                have, have_rounds, have_temp, have_k = self.existing_pair(fn)
                if have >= args.trials:
                    print(f"[{pi+1}/{n_units}] {name}: cached {have}/{args.trials} trials -> skip", flush=True); continue
                if have > 0:
                    issues = []
                    if have_rounds is not None and have_rounds != args.iters + 1:
                        issues.append(f"iters {have_rounds-1}!=--iters {args.iters}")
                    if self._num(have_temp) is not None and self._num(have_temp) != args.temp:
                        issues.append(f"temp {have_temp}!=--temp {args.temp}")
                    if self._num(have_k) is not None and int(self._num(have_k)) != args.k:
                        issues.append(f"k {have_k}!=--k {args.k}")
                    if issues:
                        print(f"[{pi+1}/{n_units}] {name}: WARNING appending with different settings "
                              f"({'; '.join(issues)}) -> CSV will mix trajectory shapes", flush=True)
                trial_start = have
            else:
                if fn.exists():
                    print(f"[{pi+1}/{n_units}] {name}: cached -> skip", flush=True); continue
                trial_start = 0
            t0 = time.time()
            insts = self.build_trials(pr, trial_start=trial_start, trial_end=args.trials)
            self.evaluate_round(insts, 0, do_ppl=True)
            for r in range(1, args.iters + 1):
                self.run_iteration(insts)
                do_ppl = (args.ppl == "all") or (r == args.iters)
                self.evaluate_round(insts, r, do_ppl=do_ppl)
            self.write_pair(pr, name, fn, insts, append=(trial_start > 0))
            done += 1
            lead = (f"trials {trial_start}..{args.trials-1} appended | "
                    if (self.cfg.trial_resume and trial_start > 0) else "")
            if self.cfg.target_free:                   # no target -> report edit depth, not peak error
                mean_ident = np.mean([t["hist"][-1]["ident"] for t in insts])
                print(f"[{pi+1}/{n_units}] {name}: {lead}{len(insts)} trials x {args.iters} iters | "
                      f"{len(insts[0]['editable'])} editable | mean ident-to-scaffold {mean_ident:.2f} "
                      f"| {time.time()-t0:.0f}s", flush=True)
            else:
                best = min(t["hist"][-1]["peak_err"] for t in insts)
                mean_end = np.mean([t["hist"][-1]["peak_err"] for t in insts])
                scaf_err = insts[0]["hist"][0]["peak_err"]
                print(f"[{pi+1}/{n_units}] {name}: {lead}scaffold err {scaf_err:.1f} -> "
                      f"mean {mean_end:.1f} / best {best:.1f} nm | {len(insts[0]['editable'])} editable "
                      f"| {time.time()-t0:.0f}s", flush=True)
        print(f"done: {done} {unit_word} designed ({done*args.trials} trajectories) -> {self.outdir} "
              f"| total {time.time()-t_all:.0f}s", flush=True)

    def run(self):
        os.makedirs(self.outdir, exist_ok=True)
        torch.manual_seed(SEED); np.random.seed(SEED)
        if getattr(self.args, "rescore", False):
            self.rescore(); return
        if self.args.backfill_ppl:
            self.backfill_ppl(); return
        if self.args.probe:
            self.probe(); return
        self.run_full()


def run(cfg: CampaignConfig, argv=None):
    """Parse CLI args for ``cfg`` and run the campaign. Called by each folder's thin driver."""
    args = build_argparser(cfg).parse_args(argv)
    Campaign(cfg, args).run()
