#!/usr/bin/env python
"""Does lam_bright=0 with the classifier loaded == the peaks-only strategy with no classifier?

This is the load-bearing assumption of the whole effort. The sweep's lam_bright=0 cells ARE read
as campaign strategy 2 (peaks only), and that reading is only legitimate if zero-weighting the
brightness term is indistinguishable from not having the term at all. If it is not, the "one sweep
covers strategies 2 and 4" claim in design_campaign.py collapses and those cells are a lookalike
rather than the real strategy.

The reason to keep the classifier loaded at lam_bright=0 rather than running a separate
brightness-free driver is that such a driver sets brightness_ckpt=None and so cannot RECORD
pred_bright. Logging it for every cell is what lets the uniform ID-and-bright filter every other
strategy is judged by apply to these cells too.

EMPIRICAL TEST. Same driver, same seed, lam_bright=0, run twice: once with the classifier loaded
(term present but zero-weighted) and once with brightness_ckpt=None (term absent). Compare
designed_seq row for row. Brightness inference is deterministic under no_grad and draws no RNG, so
identical selection is expected -- but "expected" is not "checked".

Result at the time of writing: 12/12 designed_seq identical. Rerun after any change to
fpdesign.campaign._select_guided.

HISTORICAL NOTE. This script used to open with a part (a): a field-by-field CampaignConfig diff of
this effort (brightness_ckpt=None) against ../guided-design-constraint/'s config, checking that
nothing differed outside {name, outdir, description, brightness_ckpt, CLI-overridden defaults}. It
ran clean -- no unexpected diffs. That effort has since been retired to
../archive/superseded-unmatched-runs/ (gitignored, not read by active code), and the constrained
spectra guide it implemented was dropped from the campaign entirely, so part (a) is no longer
re-runnable and has been removed. The empirical half below was always the load-bearing one.

    python check_lam_bright0.py
"""
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMP = HERE.parent
SCRATCH = HERE / "_equiv"


def load(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


matched = load(CAMP / "esm2_guided" / "design_campaign.py", "m_matched")

print("=" * 78)
print("EMPIRICAL: lam_bright=0, classifier loaded vs absent")
print("=" * 78)
ARGV = ["--trials", "3", "--iters", "3", "--pairs", "mOrange", "--temp", "1",
        "--lam-ex", "1", "--lam-em", "1", "--lam-edit", "1", "--lam-bright", "0"]
args = matched.build_argparser(matched.BASE).parse_args(ARGV)

seqs = {}
for label, ckpt in [("classifier loaded (lam_bright=0)", matched.BRIGHTNESS_CKPT),
                    ("classifier absent  (ckpt=None)  ", None)]:
    cfg = replace(matched.BASE, outdir=SCRATCH / ("load" if ckpt else "none"),
                  brightness_ckpt=ckpt,
                  record_brightness=ckpt is not None)
    print(f"\n--- {label}")
    matched.SweepCampaign(cfg, args).run()
    import pandas as pd
    csv = cfg.outdir / "design_EGFP-mOrange.csv"
    df = pd.read_csv(csv).sort_values(["trial", "round"])
    seqs[label] = list(df.designed_seq)

k = list(seqs)
same = seqs[k[0]] == seqs[k[1]]
n = len(seqs[k[0]])
print()
print(f"  rows compared        : {n}")
print(f"  designed_seq IDENTICAL: {same}")
if not same:
    for i, (x, y) in enumerate(zip(seqs[k[0]], seqs[k[1]])):
        if x != y:
            d = [j for j, (p, q) in enumerate(zip(x, y)) if p != q]
            print(f"  row {i}: differs at {len(d)} positions {d[:10]}")
print()
print("VERDICT:", "PASS" if same else "FAIL")
