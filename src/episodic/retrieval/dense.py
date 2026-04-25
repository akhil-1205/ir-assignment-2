"""Dense retriever over a FAISS index."""

from __future__ import annotations

from ..indexing.dense_index import DenseIndex
from ..indexing.metadata_store import MetadataStore
from .base import OutcomeFilter, RetrievalResult, apply_outcome_filter


class DenseRetriever:
    def __init__(
        self,
        index: DenseIndex,
        store: MetadataStore,
        oversample: int = 5,
        name: str | None = None,
    ):
        self.index = index
        self.store = store
        self.oversample = oversample
        self.name = name or f"dense-{index.field}"

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
