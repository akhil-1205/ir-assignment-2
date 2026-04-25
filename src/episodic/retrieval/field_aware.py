"""Field-aware retriever combining state similarity, plan similarity,
tool overlap, and outcome match into a weighted score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..indexing.bm25_index import BM25Index
from ..indexing.dense_index import DenseIndex
from ..indexing.metadata_store import MetadataStore
from ..schema import Episode
from .base import OutcomeFilter, RetrievalResult, apply_outcome_filter


@dataclass
class FieldWeights:
    w_state: float = 0.5
    w_plan: float = 0.2
    w_tools: float = 0.2
    w_outcome: float = 0.1


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _extract_tool_hints(query: str, vocab: list[str]) -> set[str]:
    q = query.lower()
    return {t for t in vocab if t.lower() in q}


class FieldAwareRetriever:
    """Recall via BM25 ∪ dense-state, then re-score with weighted fields."""

    def __init__(
        self,
        bm25: BM25Index,
        state_index: DenseIndex,
        plan_index: DenseIndex,
        store: MetadataStore,
        weights: FieldWeights | None = None,
        tool_vocab: list[str] | None = None,
        recall_multiplier: int = 5,
    ):
        self.bm25 = bm25
        self.state_index = state_index
        self.plan_index = plan_index
        self.store = store
        self.weights = weights or FieldWeights()
        self.tool_vocab = tool_vocab or self._infer_tool_vocab(store)
        self.recall_multiplier = recall_multiplier
        w = self.weights
        self.name = (
            f"field-aware-s{w.w_state:g}-p{w.w_plan:g}-t{w.w_tools:g}-o{w.w_outcome:g}"
        )

    @staticmethod
    def _infer_tool_vocab(store: MetadataStore) -> list[str]:
        vocab: set[str] = set()
        for ep in store.all_episodes():
            vocab.update(ep.tools_used)
        return sorted(vocab)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_outcome: OutcomeFilter = None,
    ) -> list[RetrievalResult]:
        n = max(k * self.recall_multiplier, k)
        candidate_ids: set[str] = set()
        candidate_ids.update(eid for eid, _ in self.bm25.search(query, k=n))
        candidate_ids.update(eid for eid, _ in self.state_index.search(query, k=n))

        if not candidate_ids:
            return []

        q_state = self.state_index.encode_query(query)
        q_plan = self.plan_index.encode_query(query)

        # vectorized cosine
        ids = list(candidate_ids)
        state_idx = [self.state_index.episode_ids.index(i) for i in ids]
        plan_idx = [self.plan_index.episode_ids.index(i) for i in ids]
        state_emb = self.state_index.embeddings[state_idx]
        plan_emb = self.plan_index.embeddings[plan_idx]
        cos_state = state_emb @ q_state
        cos_plan = plan_emb @ q_plan

        query_tools = _extract_tool_hints(query, self.tool_vocab)
        target_outcome = filter_outcome  # may be None

        w = self.weights
        scored: list[tuple[str, float]] = []
        for j, eid in enumerate(ids):
            ep: Episode = self.store.get(eid)
            tool_overlap = _jaccard(query_tools, set(ep.tools_used))
            outcome_match = (
                1.0 if (target_outcome is not None and ep.outcome_label == target_outcome)
                else (0.5 if target_outcome is None else 0.0)
            )
            s = (
                w.w_state * float(cos_state[j])
                + w.w_plan * float(cos_plan[j])
                + w.w_tools * tool_overlap
                + w.w_outcome * outcome_match
            )
            scored.append((eid, s))
        scored.sort(key=lambda x: -x[1])

        pool_k = k * 3 if filter_outcome else k
        top = scored[:pool_k]
        results = [
            RetrievalResult(episode=self.store.get(eid), score=s) for eid, s in top
        ]
        return apply_outcome_filter(results, filter_outcome, k)
