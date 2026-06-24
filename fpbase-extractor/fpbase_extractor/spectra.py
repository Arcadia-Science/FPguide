"""Normalize and write full excitation/emission spectra from FPbase.

The REST spectra endpoint returns, per protein, a list of spectra. Each spectrum
is identified by a `state` string whose suffix encodes the spectrum type
(`_ex` excitation, `_em` emission, `_ab` absorption, `_2p` two-photon) and
carries `data`: a list of [wavelength_nm, intensity] pairs (intensity is the
normalized 0–1 spectrum).
"""

import csv
import json
import os

# Suffix on the `state` string -> readable spectrum type.
SPECTRUM_TYPES = {
    "ex": "excitation",
    "em": "emission",
    "ab": "absorption",
    "2p": "two-photon",
    "2pa": "two-photon",
}


def _split_state(state):
    """Split a spectrum `state` like 'default_ex' into (state_label, type).

    The suffix after the final underscore is the spectrum type; the rest is the
    state label. Falls back gracefully if there is no recognizable suffix.
    """
    if state and "_" in state:
        label, _, suffix = state.rpartition("_")
        return label or state, SPECTRUM_TYPES.get(suffix.lower(), suffix.lower())
    return state or "", ""


def normalize_spectra(records):
    """Normalize raw spectra records into a uniform per-protein structure.

    Returns a list of dicts:
        {"name", "slug", "spectra": [
            {"state", "spectrum_type", "raw_state", "ec", "max",
             "data": [[wavelength, intensity], ...]}, ...]}
    """
    out = []
    for rec in records:
        spectra = []
        for sp in rec.get("spectra") or []:
            label, stype = _split_state(sp.get("state"))
            spectra.append(
                {
                    "state": label,
                    "spectrum_type": stype,
                    "raw_state": sp.get("state"),
                    "ec": sp.get("ec"),
                    "max": sp.get("max"),
                    "data": sp.get("data") or [],
                }
            )
        out.append({"name": rec.get("name"), "slug": rec.get("slug"), "spectra": spectra})
    return out


def write_spectra_json(spectra, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spectra, fh, ensure_ascii=False)
    return len(spectra)


LONG_COLUMNS = ["slug", "name", "state", "spectrum_type", "wavelength", "intensity"]


def write_spectra_long_csv(spectra, path):
    """Write tidy long-format CSV: one row per (protein, spectrum, wavelength).

    This shape loads directly into pandas/R/ggplot for plotting spectra.
    Returns the number of data rows written.
    """
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(LONG_COLUMNS)
        for prot in spectra:
            slug, name = prot.get("slug"), prot.get("name")
            for sp in prot.get("spectra") or []:
                state, stype = sp.get("state"), sp.get("spectrum_type")
                for point in sp.get("data") or []:
                    if len(point) < 2:
                        continue
                    writer.writerow([slug, name, state, stype, point[0], point[1]])
                    rows += 1
    return rows


def write_spectra_outputs(spectra, outdir, formats, basename="fpbase_spectra"):
    """Write spectra in the requested formats. formats: subset of {json, csv}."""
    os.makedirs(outdir, exist_ok=True)
    results = {}
    if "json" in formats:
        path = os.path.join(outdir, f"{basename}.json")
        results["spectra-json"] = (path, write_spectra_json(spectra, path))
    if "csv" in formats:
        path = os.path.join(outdir, f"{basename}_long.csv")
        results["spectra-csv"] = (path, write_spectra_long_csv(spectra, path))
    return results
