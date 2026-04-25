"""Hybrid retriever: alpha * BM25 + (1 - alpha) * dense, score-normalized."""

from __future__ import annotations

import numpy as np

from ..indexing.bm25_index import BM25Index
from ..indexing.dense_index import DenseIndex
from ..indexing.metadata_store import MetadataStore
from .base import OutcomeFilter, RetrievalResult, apply_outcome_filter


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        store: MetadataStore,
        alpha: float = 0.5,
        recall_multiplier: int = 5,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.store = store
        self.alpha = alpha
        self.recall_multiplier = recall_multiplier
        self.name = f"hybrid-a{alpha:g}"

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_outcome: OutcomeFilter = None,
    ) -> list[RetrievalResult]:
        n = max(k * self.recall_multiplier, k)
        bm_hits = dict(self.bm25.search(query, k=n))
        dn_hits = dict(self.dense.search(query, k=n))

        bm_norm = _minmax(bm_hits)
        dn_norm = _minmax(dn_hits)

        candidates = set(bm_hits) | set(dn_hits)
        combined: list[tuple[str, float]] = []
        for eid in candidates:
            b = bm_norm.get(eid, 0.0)
            d = dn_norm.get(eid, 0.0)
            score = self.alpha * b + (1 - self.alpha) * d
            combined.append((eid, score))
        combined.sort(key=lambda x: -x[1])

        pool_k = k * 3 if filter_outcome else k
        top = combined[:pool_k]
        results = [
            RetrievalResult(episode=self.store.get(eid), score=s) for eid, s in top
        ]
        return apply_outcome_filter(results, filter_outcome, k)
