"""Normalize FPbase protein records and write FASTA / CSV / JSON outputs.

`normalize_protein` flattens a raw record (from either the GraphQL or REST
source) into a uniform dict. `write_*` functions render the normalized records.
"""

import csv
import json
import os

from . import mappings


def _first(record, *keys):
    """Return the first present, non-None value among `keys`."""
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _normalize_state(state):
    """Flatten one photophysical state (handles GraphQL and REST key names)."""
    return {
        "name": _first(state, "name"),
        "slug": _first(state, "slug"),
        "ex_max": _first(state, "exMax", "ex_max"),
        "em_max": _first(state, "emMax", "em_max"),
        "ext_coeff": _first(state, "extCoeff", "ext_coeff"),
        "qy": _first(state, "qy"),
        "brightness": _first(state, "brightness"),
        "pka": _first(state, "pka"),
        "maturation": _first(state, "maturation"),
        "lifetime": _first(state, "lifetime"),
        "twop_ex_max": _first(state, "twopExMax"),
        "twop_peak_gm": _first(state, "twopPeakGm"),
        "twop_qy": _first(state, "twopQy"),
        "is_dark": _first(state, "isDark"),
        "emhex": _first(state, "emhex"),
        "exhex": _first(state, "exhex"),
    }


def normalize_protein(record):
    """Normalize one raw protein record into a uniform dict.

    Works for both the GraphQL (camelCase, nested organism/reference) and REST
    (snake_case, flat) payloads.
    """
    # Parent organism: nested object in GraphQL, absent in REST.
    organism = None
    org = record.get("parentOrganism")
    if isinstance(org, dict):
        organism = org.get("scientificName")

    # Primary reference: nested in GraphQL; REST only exposes a bare DOI.
    ref = record.get("primaryReference") or {}
    doi = _first(record, "doi") or ref.get("doi")

    agg = _first(record, "agg")
    switch = _first(record, "switchType", "switch_type")

    return {
        "slug": _first(record, "slug"),
        "uuid": _first(record, "uuid"),
        "name": _first(record, "name"),
        "aliases": record.get("aliases") or [],
        "seq": _first(record, "seq"),
        "parent_organism": organism,
        "agg": agg,
        "oligomerization": mappings.oligomerization(agg),
        "switch_type": switch,
        "switch_type_label": mappings.switch_type(switch),
        "cofactor": _first(record, "cofactor"),
        "genbank": _first(record, "genbank"),
        "uniprot": _first(record, "uniprot"),
        "ipg_id": _first(record, "ipgId", "ipg_id"),
        "pdb": record.get("pdb") or [],
        "doi": doi,
        "ref_year": ref.get("year"),
        "ref_journal": ref.get("journal"),
        "ref_title": ref.get("title"),
        "states": [_normalize_state(s) for s in (record.get("states") or [])],
    }


def normalize_all(records):
    return [normalize_protein(r) for r in records]


# --- Output writers -------------------------------------------------------

# Per-protein identity columns, repeated on each state row in the CSV.
_PROTEIN_COLUMNS = [
    "slug",
    "name",
    "aliases",
    "parent_organism",
    "agg",
    "oligomerization",
    "switch_type",
    "switch_type_label",
    "cofactor",
    "genbank",
    "uniprot",
    "ipg_id",
    "pdb",
    "doi",
    "ref_year",
    "ref_journal",
    "ref_title",
    "seq_length",
    "seq",
]

# Per-state phenotype columns.
_STATE_COLUMNS = [
    "state_name",
    "ex_max",
    "em_max",
    "ext_coeff",
    "qy",
    "brightness",
    "pka",
    "maturation",
    "lifetime",
    "twop_ex_max",
    "twop_peak_gm",
    "twop_qy",
    "is_dark",
    "emhex",
    "exhex",
]

CSV_COLUMNS = _PROTEIN_COLUMNS + _STATE_COLUMNS


def _join(value):
    """Render a list field for CSV as a semicolon-separated string."""
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    return value


def iter_csv_rows(proteins):
    """Yield one flat dict per (protein, state). Stateless proteins yield one
    row with empty phenotype columns."""
    for p in proteins:
        base = {col: _join(p.get(col)) for col in _PROTEIN_COLUMNS if col != "seq_length"}
        base["seq_length"] = len(p["seq"]) if p.get("seq") else ""
        base["seq"] = p.get("seq") or ""
        states = p.get("states") or [None]
        for state in states:
            row = dict(base)
            if state is None:
                for col in _STATE_COLUMNS:
                    row[col] = ""
            else:
                row["state_name"] = state.get("name")
                for col in _STATE_COLUMNS[1:]:
                    row[col] = state.get(col)
            yield row


def write_csv(proteins, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        rows = 0
        for row in iter_csv_rows(proteins):
            writer.writerow(row)
            rows += 1
    return rows


def write_json(proteins, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(proteins, fh, indent=2, ensure_ascii=False)
    return len(proteins)


def _fasta_header(p):
    """Build a descriptive FASTA header from a protein's key phenotype."""
    parts = [p.get("slug") or p.get("name") or "unknown"]
    parts.append(p.get("name") or "")
    # Use the first (default) state's spectra for the header summary.
    states = p.get("states") or []
    if states:
        s = states[0]
        if s.get("ex_max") is not None:
            parts.append(f"ex={s['ex_max']}")
        if s.get("em_max") is not None:
            parts.append(f"em={s['em_max']}")
    if p.get("oligomerization"):
        parts.append(f"olig={p['oligomerization']}")
    if p.get("parent_organism"):
        parts.append(f"org={p['parent_organism']}")
    return " | ".join(str(x) for x in parts if x != "")


def write_fasta(proteins, path, wrap=60):
    """Write one FASTA record per protein that has a sequence."""
    written = 0
    with open(path, "w", encoding="utf-8") as fh:
        for p in proteins:
            seq = p.get("seq")
            if not seq:
                continue
            fh.write(f">{_fasta_header(p)}\n")
            if wrap and wrap > 0:
                for i in range(0, len(seq), wrap):
                    fh.write(seq[i : i + wrap] + "\n")
            else:
                fh.write(seq + "\n")
            written += 1
    return written


def write_outputs(proteins, outdir, formats, basename="fpbase_proteins"):
    """Write requested output formats into `outdir`. Returns {format: (path, count)}."""
    os.makedirs(outdir, exist_ok=True)
    results = {}
    if "fasta" in formats:
        path = os.path.join(outdir, f"{basename}.fasta")
        results["fasta"] = (path, write_fasta(proteins, path))
    if "csv" in formats:
        path = os.path.join(outdir, f"{basename}.csv")
        results["csv"] = (path, write_csv(proteins, path))
    if "json" in formats:
        path = os.path.join(outdir, f"{basename}.json")
        results["json"] = (path, write_json(proteins, path))
    return results
