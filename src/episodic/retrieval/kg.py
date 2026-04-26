"""Knowledge-graph retriever using personalized PageRank with restart."""

from __future__ import annotations

from ..indexing.kg_index import KGIndex
from ..indexing.metadata_store import MetadataStore
from .base import OutcomeFilter, RetrievalResult, apply_outcome_filter


class KGRetriever:
    def __init__(
        self,
        index: KGIndex,
        store: MetadataStore,
        alpha: float = 0.85,
        n_iter: int = 30,
        oversample: int = 5,
        name: str | None = None,
    ):
        self.index = index
        self.store = store
        self.alpha = alpha
        self.n_iter = n_iter
        self.oversample = oversample
        self.name = name or f"kg-ppr-a{alpha:g}"

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_outcome: OutcomeFilter = None,
    ) -> list[RetrievalResult]:
        pool_k = k * self.oversample if filter_outcome else k
        hits = self.index.search(query, k=pool_k, alpha=self.alpha, n_iter=self.n_iter)
        results = [
            RetrievalResult(episode=self.store.get(eid), score=score)
            for eid, score in hits
        ]
        return apply_outcome_filter(results, filter_outcome, k)
