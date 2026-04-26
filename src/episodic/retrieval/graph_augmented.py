"""Graph-Augmented Multi-Signal Reranker (GAR).

A five-stage pipeline that fuses lexical, dense, structural (KG), and
neural (cross-encoder) signals, then diversifies the final top-k via
MMR. Each stage addresses a failure mode of the others:

  Stage 1 — Reciprocal Rank Fusion over diverse first-stage retrievers.
            BM25 catches lexical exact matches, dense catches paraphrase,
            KG-PPR catches multi-hop structural connections (shared tools
            / entities). RRF combines them without needing comparable
            score scales — it uses ranks alone.

  Stage 2 — Cross-encoder pairwise scoring of (query, episode.full_doc).
            The strongest single signal. Joint attention catches
            fine-grained relevance no bi-encoder can.

  Stage 3 — Graph features per candidate:
              * PPR mass under the same query seeds (centrality in the
                heterogeneous graph)
              * Jaccard tool overlap between query-inferred tools and
                episode tools (symbolic match)

  Stage 4 — Min-max normalized weighted fusion of all four signals
            within the candidate pool.

  Stage 5 — MMR (Maximal Marginal Relevance) greedy selection of the
            final top-k using dense-embedding cosine for similarity.
            Prevents the top-k from collapsing onto near-duplicate
            episodes — a real risk when multiple highly-relevant docs
            share content.

Outcome filter is applied between stages 4 and 5 so the diversifier
operates on already filter-correct candidates.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..indexing.dense_index import DenseIndex
from ..indexing.kg_index import KGIndex
from ..indexing.metadata_store import MetadataStore
from ..schema import Episode
from .base import OutcomeFilter, RetrievalResult, Retriever, apply_outcome_filter
from .cross_encoder import DEFAULT_CE_MODEL


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class GraphAugmentedReranker:
    """Five-stage reranker that fuses RRF + CE + KG features + MMR."""

    def __init__(
        self,
        first_stages: list[Retriever],
        kg_index: KGIndex,
        store: MetadataStore,
        dense_index_for_mmr: Optional[DenseIndex] = None,
        score_fn: Optional[Callable[[list[tuple[str, str]]], list[float]]] = None,
        ce_model_name: str = DEFAULT_CE_MODEL,
        recall_pool: int = 50,
        rerank_pool: int = 40,
        rrf_k: int = 60,
        weight_ce: float = 0.65,
        weight_ppr: float = 0.20,
        weight_tool: float = 0.10,
        weight_rrf: float = 0.05,
        mmr_lambda: float = 1.0,  # 1.0 = pure relevance; lower to diversify
        ppr_alpha: float = 0.85,
        ppr_iter: int = 25,
        oversample_for_filter: int = 3,
        name: Optional[str] = None,
    ):
        self.first_stages = first_stages
        self.kg = kg_index
        self.store = store
        self.dense_for_mmr = dense_index_for_mmr
        self._score_fn = score_fn
        self.ce_model_name = ce_model_name
        self.recall_pool = recall_pool
        self.rerank_pool = rerank_pool
        self.rrf_k = rrf_k
        self.weight_ce = weight_ce
        self.weight_ppr = weight_ppr
        self.weight_tool = weight_tool
        self.weight_rrf = weight_rrf
        self.mmr_lambda = mmr_lambda
        self.ppr_alpha = ppr_alpha
        self.ppr_iter = ppr_iter
        self.oversample_for_filter = oversample_for_filter
        self._ce_model = None
        self.name = name or "graph-augmented"

    # ---- model lazy load ----------------------------------------------------

    def _ensure_ce(self):
        if self._score_fn is not None or self._ce_model is not None:
            return
        from sentence_transformers import CrossEncoder  # type: ignore

        self._ce_model = CrossEncoder(self.ce_model_name)

    def _ce_score(self, query: str, episodes: list[Episode]) -> np.ndarray:
        if not episodes:
            return np.array([], dtype=np.float64)
        pairs = [(query, ep.full_document) for ep in episodes]
        if self._score_fn is not None:
            return np.asarray(self._score_fn(pairs), dtype=np.float64)
        self._ensure_ce()
        scores = self._ce_model.predict(pairs, convert_to_numpy=True)  # type: ignore[union-attr]
        return np.asarray(scores, dtype=np.float64)

    # ---- stage 1: RRF -------------------------------------------------------

    def _rrf_recall(self, query: str) -> list[tuple[Episode, float]]:
        rrf: dict[str, float] = {}
        ep_lookup: dict[str, Episode] = {}
        for r in self.first_stages:
            hits = r.retrieve(query, k=self.recall_pool)
            for rank, hit in enumerate(hits, start=1):
                eid = hit.episode.episode_id
                rrf[eid] = rrf.get(eid, 0.0) + 1.0 / (self.rrf_k + rank)
                ep_lookup[eid] = hit.episode
        ordered_ids = sorted(rrf.keys(), key=lambda e: -rrf[e])
        return [(ep_lookup[e], rrf[e]) for e in ordered_ids]

    # ---- stage 3: graph features --------------------------------------------

    def _ppr_mass(self, query: str, episodes: list[Episode]) -> np.ndarray:
        masses = self.kg.ppr_mass_for_episodes(
            query,
            [ep.episode_id for ep in episodes],
            alpha=self.ppr_alpha,
            n_iter=self.ppr_iter,
        )
        return np.asarray(masses, dtype=np.float64)

    def _tool_overlap(self, query: str, episodes: list[Episode]) -> np.ndarray:
        q_lower = (query or "").lower()
        q_tools = {t for t in self.kg.tool_vocab if t.lower() in q_lower}
        if not q_tools:
            return np.zeros(len(episodes), dtype=np.float64)
        out = np.zeros(len(episodes), dtype=np.float64)
        for i, ep in enumerate(episodes):
            ep_tools = set(ep.tools_used)
            union = q_tools | ep_tools
            if union:
                out[i] = len(q_tools & ep_tools) / len(union)
        return out

    # ---- stage 5: MMR -------------------------------------------------------

    def _episode_embeddings(self, episodes: list[Episode]) -> Optional[np.ndarray]:
        if self.dense_for_mmr is None:
            return None
        idx_map = {eid: i for i, eid in enumerate(self.dense_for_mmr.episode_ids)}
        rows = []
        for ep in episodes:
            i = idx_map.get(ep.episode_id)
            if i is None:
                return None  # missing — skip MMR cleanly
            rows.append(self.dense_for_mmr.embeddings[i])
        return np.stack(rows)

    def _mmr_select(
        self,
        episodes: list[Episode],
        relevance: np.ndarray,
        k: int,
    ) -> list[RetrievalResult]:
        if not episodes:
            return []
        n = len(episodes)
        emb = self._episode_embeddings(episodes)
        sim = emb @ emb.T if emb is not None else None  # cosine since vectors are normalized

        selected: list[int] = []
        available = set(range(n))

        # First pick: highest relevance.
        first = int(np.argmax(relevance))
        selected.append(first)
        available.discard(first)

        while len(selected) < min(k, n) and available:
            best_idx = -1
            best_score = -np.inf
            for i in available:
                if sim is not None:
                    max_sim = float(max(sim[i, s] for s in selected))
                else:
                    max_sim = 0.0
                mmr = self.mmr_lambda * relevance[i] - (1.0 - self.mmr_lambda) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            if best_idx < 0:
                break
            selected.append(best_idx)
            available.discard(best_idx)

        return [
            RetrievalResult(episode=episodes[i], score=float(relevance[i]))
            for i in selected
        ]

    # ---- main entry ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_outcome: OutcomeFilter = None,
    ) -> list[RetrievalResult]:
        # Stage 1: RRF recall
        rrf_pool = self._rrf_recall(query)
        if not rrf_pool:
            return []

        pool_size = self.rerank_pool
        if filter_outcome:
            pool_size = max(pool_size, k * self.oversample_for_filter * 2)
        rrf_pool = rrf_pool[:pool_size]

        episodes = [ep for ep, _ in rrf_pool]
        rrf_scores = np.asarray([s for _, s in rrf_pool], dtype=np.float64)

        # Stage 2: cross-encoder
        ce_scores = self._ce_score(query, episodes)

        # Stage 3: graph features
        ppr = self._ppr_mass(query, episodes)
        tool = self._tool_overlap(query, episodes)

        # Stage 4: normalized weighted fusion
        relevance = (
            self.weight_ce * _minmax(ce_scores)
            + self.weight_ppr * _minmax(ppr)
            + self.weight_tool * _minmax(tool)
            + self.weight_rrf * _minmax(rrf_scores)
        )

        # Outcome filter before MMR so diversification picks among matches.
        if filter_outcome:
            keep = [
                i for i, ep in enumerate(episodes)
                if ep.outcome_label == filter_outcome
            ]
            if not keep:
                return []
            episodes = [episodes[i] for i in keep]
            relevance = relevance[keep]

        # Stage 5: MMR diversification
        return self._mmr_select(episodes, relevance, k)
