"""Two-stage retrieval: cheap first-stage recall + cross-encoder rerank.

A bi-encoder (e.g. our DenseRetriever) embeds query and document
independently; a cross-encoder concatenates them and runs them through
a transformer together, so attention can flow across the boundary.
That joint scoring is meaningfully more discriminating, at the cost of
N-way model calls per query (where N is the candidate pool from the
first stage). Standard practice: recall a wide pool cheaply, rerank
the top-N with the cross-encoder.

We use `sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2`
by default — ~80 MB, MS-MARCO-trained on relevance, downloads on
first use.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..schema import Episode
from .base import OutcomeFilter, RetrievalResult, Retriever, apply_outcome_filter

DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Wrap a first-stage retriever; rerank its top-N with a cross-encoder."""

    def __init__(
        self,
        first_stage: Retriever,
        model_name: str = DEFAULT_CE_MODEL,
        rerank_pool: int = 30,
        oversample_for_filter: int = 3,
        text_for: Optional[Callable[[Episode], str]] = None,
        name: Optional[str] = None,
        score_fn: Optional[Callable[[list[tuple[str, str]]], list[float]]] = None,
    ):
        """
        first_stage     -- any Retriever; its top `rerank_pool` candidates are reranked.
        rerank_pool     -- size of the candidate set sent to the cross-encoder.
        oversample_for_filter -- when filter_outcome is set, multiply pool to leave room.
        text_for        -- maps an Episode to the text the cross-encoder sees.
                           Defaults to the episode's full_document.
        score_fn        -- callable(list_of_(query,doc)_pairs) -> list of scores.
                           Lets tests inject a stub model. If None, lazy-loads
                           a real CrossEncoder on first use.
        """
        self.first_stage = first_stage
        self.model_name = model_name
        self.rerank_pool = rerank_pool
        self.oversample_for_filter = oversample_for_filter
        self.text_for = text_for or (lambda ep: ep.full_document)
        self.name = name or f"ce-rerank({getattr(first_stage, 'name', 'first')})"
        self._score_fn = score_fn
        self._model = None

    # --- model lazy-load -----------------------------------------------------

    def _ensure_model(self):
        if self._score_fn is not None:
            return
        if self._model is None:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(self.model_name)

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._score_fn is not None:
            return list(self._score_fn(pairs))
        self._ensure_model()
        scores = self._model.predict(pairs, convert_to_numpy=True)  # type: ignore[union-attr]
        return [float(s) for s in scores]

    # --- retrieval -----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_outcome: OutcomeFilter = None,
    ) -> list[RetrievalResult]:
        pool_size = self.rerank_pool
        if filter_outcome:
            pool_size = max(pool_size, k * self.oversample_for_filter)

        # First-stage recall
        candidates = self.first_stage.retrieve(query, k=pool_size, filter_outcome=None)
        if not candidates:
            return []

        # Cross-encoder scoring of (query, episode_text) pairs
        pairs = [(query, self.text_for(c.episode)) for c in candidates]
        scores = self._score(pairs)

        reranked = [
            RetrievalResult(episode=c.episode, score=float(s))
            for c, s in zip(candidates, scores)
        ]
        reranked.sort(key=lambda r: -r.score)

        # Outcome filter is applied AFTER rerank so the strongest
        # filter-matching candidates rise to the top.
        keep = k if not filter_outcome else len(reranked)
        return apply_outcome_filter(reranked[:keep], filter_outcome, k)
