"""Tests for normalization and output writers (no network required)."""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpbase_extractor import extract  # noqa: E402

# A GraphQL-shaped record (camelCase, nested organism/reference, two states).
GRAPHQL_RECORD = {
    "name": "Dronpa",
    "slug": "dronpa",
    "seq": "MAGTLP",
    "aliases": ["22G"],
    "genbank": "AB180726",
    "uniprot": "Q5S5G4",
    "ipgId": "12345",
    "pdb": ["2IE2", "2POX"],
    "agg": "m",
    "switchType": "ps",
    "cofactor": None,
    "parentOrganism": {"scientificName": "Pectiniidae"},
    "primaryReference": {"doi": "10.1/x", "year": 2004, "journal": "Science", "title": "T"},
    "states": [
        {"name": "on", "exMax": 503, "emMax": 518, "extCoeff": 95000, "qy": 0.85, "isDark": False},
        {"name": "off", "exMax": 390, "emMax": None, "qy": None, "isDark": True},
    ],
}

# A REST-shaped record (snake_case, flat doi, no organism/reference/aliases).
REST_RECORD = {
    "uuid": "ABC12",
    "name": "EGFP",
    "slug": "egfp",
    "seq": "MVSKGE",
    "genbank": "AAB02572",
    "uniprot": "C5MKY7",
    "pdb": ["2Y0G"],
    "agg": "wd",
    "switch_type": "b",
    "doi": "10.1016/x",
    "states": [{"name": "default", "ex_max": 488, "em_max": 507, "ext_coeff": 55900, "qy": 0.6}],
}


def test_normalize_graphql():
    p = extract.normalize_protein(GRAPHQL_RECORD)
    assert p["name"] == "Dronpa"
    assert p["aliases"] == ["22G"]
    assert p["parent_organism"] == "Pectiniidae"
    assert p["oligomerization"] == "monomer"
    assert p["switch_type_label"] == "photoswitchable"
    assert p["doi"] == "10.1/x"
    assert p["ref_year"] == 2004
    assert len(p["states"]) == 2
    assert p["states"][0]["ex_max"] == 503
    assert p["states"][1]["is_dark"] is True


def test_normalize_rest():
    p = extract.normalize_protein(REST_RECORD)
    assert p["name"] == "EGFP"
    assert p["oligomerization"] == "weak dimer"
    assert p["switch_type_label"] == "basic"
    # REST exposes doi flat, with no nested reference detail.
    assert p["doi"] == "10.1016/x"
    assert p["ref_year"] is None
    assert p["states"][0]["em_max"] == 507


def test_unknown_codes_fall_back_to_raw():
    p = extract.normalize_protein({"name": "X", "agg": "zz", "switchType": "qq"})
    assert p["oligomerization"] == "zz"
    assert p["switch_type_label"] == "qq"


def test_csv_one_row_per_state(tmp_path):
    proteins = extract.normalize_all([GRAPHQL_RECORD, REST_RECORD])
    path = tmp_path / "out.csv"
    rows_written = extract.write_csv(proteins, str(path))
    assert rows_written == 3  # 2 Dronpa states + 1 EGFP state
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["name"] == "Dronpa"
    assert rows[0]["state_name"] == "on"
    assert rows[0]["seq_length"] == "6"
    assert rows[0]["pdb"] == "2IE2; 2POX"
    assert rows[1]["state_name"] == "off"
    assert rows[2]["name"] == "EGFP"


def test_stateless_protein_yields_one_row(tmp_path):
    proteins = extract.normalize_all([{"name": "NoState", "slug": "nostate", "seq": "AA"}])
    path = tmp_path / "out.csv"
    assert extract.write_csv(proteins, str(path)) == 1


def test_fasta_skips_seqless_and_wraps(tmp_path):
    proteins = extract.normalize_all([GRAPHQL_RECORD, {"name": "NoSeq", "slug": "noseq"}])
    path = tmp_path / "out.fasta"
    written = extract.write_fasta(proteins, str(path), wrap=3)
    assert written == 1
    text = path.read_text()
    assert text.startswith(">dronpa | Dronpa | ex=503 | em=518")
    # MAGTLP wrapped at width 3 -> two lines.
    assert "MAG\nTLP\n" in text


def test_json_roundtrip(tmp_path):
    proteins = extract.normalize_all([GRAPHQL_RECORD])
    path = tmp_path / "out.json"
    extract.write_json(proteins, str(path))
    loaded = json.loads(path.read_text())
    assert loaded[0]["name"] == "Dronpa"
    assert loaded[0]["states"][1]["is_dark"] is True


# --- spectra -------------------------------------------------------------

from fpbase_extractor import spectra as spectra_mod  # noqa: E402

SPECTRA_RECORD = {
    "name": "EGFP",
    "slug": "egfp",
    "spectra": [
        {"state": "default_ex", "ec": None, "max": 488, "data": [[480, 0.9], [481, 0.95]]},
        {"state": "default_em", "ec": 55900, "max": 507, "data": [[505, 0.8], [506, 1.0]]},
    ],
}


def test_split_state_types():
    assert spectra_mod._split_state("default_ex") == ("default", "excitation")
    assert spectra_mod._split_state("on_state_em") == ("on_state", "emission")
    assert spectra_mod._split_state("weird") == ("weird", "")


def test_normalize_spectra():
    norm = spectra_mod.normalize_spectra([SPECTRA_RECORD])
    assert norm[0]["slug"] == "egfp"
    types = {s["spectrum_type"] for s in norm[0]["spectra"]}
    assert types == {"excitation", "emission"}


def test_spectra_long_csv(tmp_path):
    norm = spectra_mod.normalize_spectra([SPECTRA_RECORD])
    path = tmp_path / "spectra.csv"
    rows = spectra_mod.write_spectra_long_csv(norm, str(path))
    assert rows == 4  # 2 ex points + 2 em points
    lines = path.read_text().strip().splitlines()
    assert lines[0] == "slug,name,state,spectrum_type,wavelength,intensity"
    assert lines[1] == "egfp,EGFP,default,excitation,480,0.9"
