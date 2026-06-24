# training_data

Cluster-wise train/val/test splits for the surrogate and oracle models (see fpbase_cluster.ipynb sec 7).

Files
-----
- split_assignments.csv : one row per FP; columns index, slug, name, cluster, cluster_size,
  ex_max, em_max, seq_len, surrogate_split, oracle_split
- spectra.npy           : (N, 1002) excitation[0:501]+emission[501:1002], row-aligned to `index`
- sequences.fasta       : one record per FP, header `>index|slug|sur=..|ora=..`
- split_meta.json       : threshold, seed, target/achieved counts, design-target sizes

Roles
-----
- Surrogate: train on surrogate_split==train, early-stop on ==val, report regression on ==test.
- Oracle:    train on oracle_split==train (different architecture!), val on ==val,
             CERTIFY accuracy on ==test  -- stratify by surrogate_split:
               ID  cert/targets = oracle_split==test & surrogate_split==train
               OOD cert/targets = oracle_split==test & surrogate_split==test
- Design test: design toward the oracle-test spectra (they double as cert + targets); report
  success separately for ID vs OOD. Biggest cluster is in train for both models.

Load a split
------------
    import csv, numpy as np
    rows = list(csv.DictReader(open("training_data/split_assignments.csv")))
    spectra = np.load("training_data/spectra.npy")
    seqs = {int(r["index"]): None for r in rows}  # fill from sequences.fasta if needed
    tr = [int(r["index"]) for r in rows if r["surrogate_split"] == "train"]
    Y_train = spectra[tr]
