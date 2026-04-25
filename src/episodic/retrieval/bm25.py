"""BM25 retriever."""

from __future__ import annotations

from ..indexing.bm25_index import BM25Index
from ..indexing.metadata_store import MetadataStore
from .base import OutcomeFilter, RetrievalResult, apply_outcome_filter


class BM25Retriever:
    name = "bm25"

    def __init__(self, index: BM25Index, store: MetadataStore, oversample: int = 5):
        self.index = index
        self.store = store
        self.oversample = oversample

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_outcome: OutcomeFilter = None,
    ) -> list[RetrievalResult]:
        pool_k = k * self.oversample if filter_outcome else k
        hits = self.index.search(query, k=pool_k)
        results = [
            RetrievalResult(episode=self.store.get(eid), score=score)
            for eid, score in hits
        ]
        return apply_outcome_filter(results, filter_outcome, k)
