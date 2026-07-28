"""Sequence-keyed disk cache for ESM-2 max-pool embeddings.

Design CSVs record everything the search itself needed (`pred_ex`, `pred_em`, `peak_err`,
`pred_bright`), but NOT the 1280-d ESM-2 max-pool embedding. That embedding is only needed
afterwards, to place a design on the GFP-DMS PCA and to compute the in-distribution test
(nearest-neighbour distance to the 40k DMS cloud vs its own 99th percentile), so nothing upstream
stores it. Recomputing it is the expensive part of both `make_shortlist_case.py` and
`visualize_campaign.ipynb`, and the same sequences get embedded several times: once for the pooled
OOD cell, again per strategy for the per-strategy panels, and again by the shortlist builder.

Wrapping the GPU embed function in `MaxPoolCache` keys each vector by sequence, so a sequence is
embedded once ever. The design pool only grows, so reruns after the first are nearly free. Caching
on the sequence (not on a file/row) is what makes it work across the overlapping pools.

    cache = MaxPoolCache(path, embed_fn)   # embed_fn(list[str]) -> (n, dim) float array
    E = cache(seqs)                        # (len(seqs), dim) float32, order preserved

The cache is a regenerable artifact -- delete the file to rebuild. It is keyed by the exact
sequence string, so the caller must be consistent about framing (e.g. always the DMS window
`seq[3:238]`); a differently framed sequence is simply a different key, never a silent wrong hit.
"""
import hashlib
from pathlib import Path

import numpy as np


class MaxPoolCache:
    def __init__(self, path, embed_fn, dim=1280):
        self.path = Path(path)
        self.embed_fn = embed_fn
        self.dim = dim
        self.emb = np.zeros((0, dim), np.float32)
        self.idx = {}
        self.n_hit = self.n_miss = 0
        if self.path.exists():
            z = np.load(self.path)
            self.emb = z["emb"].astype(np.float32)
            self.idx = {k: i for i, k in enumerate(z["keys"].astype(str))}

    @staticmethod
    def _key(seq):
        return hashlib.sha1(seq.encode()).hexdigest()

    def __call__(self, seqs):
        seqs = list(seqs)
        if not seqs:
            return np.zeros((0, self.dim), np.float32)
        keys = [self._key(s) for s in seqs]
        # dict() dedupes within this call too, so a sequence repeated across pools is embedded once
        missing = {k: s for k, s in zip(keys, seqs) if k not in self.idx}
        self.n_hit += len(seqs) - sum(k in missing for k in keys)
        self.n_miss += len(missing)
        if missing:
            mk, ms = list(missing.keys()), list(missing.values())
            new = np.asarray(self.embed_fn(ms), np.float32)
            base = len(self.emb)
            self.emb = np.concatenate([self.emb, new])
            self.idx.update({k: base + i for i, k in enumerate(mk)})
            self._save()
        return self.emb[[self.idx[k] for k in keys]]

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(sorted(self.idx, key=self.idx.get))
        tmp = self.path.with_suffix(".tmp.npz")          # atomic: never leave a half-written cache
        np.savez(tmp, keys=keys, emb=self.emb)
        tmp.replace(self.path)

    def __repr__(self):
        return (f"MaxPoolCache({self.path.name}: {len(self.idx)} cached, "
                f"{self.n_hit} hits / {self.n_miss} embedded this session)")
