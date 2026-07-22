#!/usr/bin/env python3
"""Data-driven bright/dim threshold for GFP DMS log-brightness distributions.

Every GFP scaffold assayed by deep mutational scanning shows a **bimodal** brightness
distribution: a pile of non-functional ("dim"/dead) variants censored at the assay's
detection floor, and a functional ("bright") mode near the wild type. The floor and the
bright mode sit at scaffold-specific brightness values (e.g. avGFP's dead floor is at
log10 ~1.3, whereas the orthologue assay of Gonzalez Somermeyer et al. floors near ~2.7),
so a single hard-coded cutoff is wrong -- the split must be learned **per scaffold**.

`log_brightness_threshold` picks the split as the **KDE antimode**: the lowest-density
point between the two most prominent peaks of a Gaussian kernel-density estimate of the
(log10) brightness. That is the natural valley separating the dead pile from the
functional mode. If no clear bimodal valley is found it falls back to the median.

Used by both transform pipelines (`transform_dms.py`, `transform_ortho_dms.py`) so the
avGFP and orthologue datasets are labelled by the same principled rule.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

DIM, BRIGHT = "dim", "bright"


def log_brightness_threshold(values, grid: int = 2000, bw=None, min_sep_frac: float = 0.15):
    """Return (threshold, info) that splits a bimodal log-brightness sample into dim|bright.

    Parameters
    ----------
    values      : 1-D array of log10 brightness for one scaffold.
    grid        : number of points to evaluate the KDE on.
    bw          : bandwidth passed to `gaussian_kde` (None -> Scott's rule).
    min_sep_frac: the second mode must be at least this fraction of the data range away
                  from the first, so a shoulder is not mistaken for a separate peak.

    Method: KDE antimode (valley between the two most prominent density peaks). Falls back
    to the median when fewer than two well-separated modes are detected.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    info: dict = {"n": int(x.size)}
    if x.size < 10 or np.ptp(x) < 1e-6:
        thr = float(np.median(x)) if x.size else float("nan")
        info.update(method="degenerate", modes=[], threshold=thr)
        return thr, info

    lo, hi = float(x.min()), float(x.max())
    xs = np.linspace(lo, hi, grid)
    dens = gaussian_kde(x, bw_method=bw)(xs)

    # interior local maxima of the density
    peaks = [i for i in range(1, grid - 1) if dens[i] > dens[i - 1] and dens[i] >= dens[i + 1]]
    peaks.sort(key=lambda i: dens[i], reverse=True)

    sep = min_sep_frac * (hi - lo)
    p1 = peaks[0] if peaks else int(dens.argmax())
    p2 = next((i for i in peaks[1:] if abs(xs[i] - xs[p1]) > sep), None)

    if p2 is None:                                   # unimodal-looking -> no principled valley
        thr = float(np.median(x))
        info.update(method="median-fallback", modes=[round(float(xs[p1]), 4)], threshold=thr)
        return thr, info

    a, b = sorted((p1, p2))
    valley = a + int(dens[a:b + 1].argmin())
    thr = float(xs[valley])
    info.update(method="kde-antimode",
                modes=sorted(round(float(xs[p]), 4) for p in (p1, p2)),
                valley_density=float(dens[valley]),
                peak_density=float(max(dens[p1], dens[p2])),
                threshold=round(thr, 4))
    return thr, info


def classify(values, threshold) -> np.ndarray:
    """Map log-brightness values to 'bright' (>= threshold) or 'dim' (< threshold)."""
    v = np.asarray(values, dtype=float)
    return np.where(v >= threshold, BRIGHT, DIM)
