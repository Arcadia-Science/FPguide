# structures — shared RCSB PDBx cache

Experimental fluorescent-protein structures, downloaded from [RCSB](https://www.rcsb.org) and kept
here so no experiment folder re-downloads them. **Data only — no code.**

| path | what it is |
|---|---|
| `experimental/` | 174 `.pdbx` entries, one per PDB ID that some part of the repo resolves a design window against |
| `1GFL.pdbx`, `1G7K.pdbx`, `3M22.pdbx` | the three whole-barrel scaffold references hard-coded in [`fpdesign/pockets.py`](../fpdesign/pockets.py)'s `STRUCT` map (avGFP, DsRed, eqFP578) |

## Who reads it

| consumer | how |
|---|---|
| [`fpdesign/build_design_windows.py`](../fpdesign/build_design_windows.py) | `STRUCTDIR = REPO / "structures" / "experimental"` |
| [`msa_conservation/conservation.py`](../msa_conservation/conservation.py) | `1GFL.pdbx`, for the barrel-core reference |
| [`in-silico-test/`](../in-silico-test) | `in-silico-test/structures` is a **symlink** to this folder; its `design_common.STRUCT_DIR` resolves here |

## Self-populating

`fpdesign/pockets.py` calls `biotite.database.rcsb.fetch(pdb, "pdbx", structdir)`, which downloads
only on a cache miss. So a PDB ID that isn't here yet is fetched on first use and written into
`experimental/` — nothing needs pre-seeding, and deleting a file only costs one re-download.

## Contents

The cache is a strict superset of every PDB ID referenced by live code in the repo: the
`scaffold_pdb` fields of the campaign and in-silico-test design windows, every row of
`in-silico-test/structure_hits.csv` (the "this protein has a ≥97%-identity PDB entry" table, so a
future cohort draw finds its structures already cached), and the three hard-coded scaffold
references above.

23 entries that only the archived peak-conditioned experiment ever touched were moved to
`archive/structures/experimental/` (untracked). They are not referenced by any live code, and
`pockets.py` would re-fetch any of them on demand if that changed.
