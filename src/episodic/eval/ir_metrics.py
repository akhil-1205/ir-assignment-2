"""IR metrics: P@k, R@k, MRR, nDCG@k. Implemented from scratch and unit-tested."""

from __future__ import annotations

import math
from typing import Iterable

from ..retrieval.base import RetrievalResult


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for x in top if x in relevant) / float(k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for x in top if x in relevant) / float(len(relevant))


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(
    retrieved: list[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    if k <= 0 or not retrieved:
        return 0.0
    gains = [relevance_scores.get(d, 0.0) for d in retrieved[:k]]
    ideal = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return dcg(gains) / idcg


def _ids(results: list[RetrievalResult] | list[str]) -> list[str]:
    if not results:
        return []
    if isinstance(results[0], str):
        return results  # type: ignore[return-value]
    return [r.episode.episode_id for r in results]  # type: ignore[union-attr]


def evaluate(
    results_by_query: dict[str, list[RetrievalResult]],
    qrels: dict[str, dict[str, float]],
    k_values: Iterable[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Aggregate IR metrics over a query batch.

    qrels[query_id] maps episode_id -> graded relevance (>=1 = relevant).
    """
    p_at: dict[int, list[float]] = {k: [] for k in k_values}
    r_at: dict[int, list[float]] = {k: [] for k in k_values}
    ndcg_at: dict[int, list[float]] = {k: [] for k in k_values}
    mrrs: list[float] = []

    for qid, results in results_by_query.items():
        rel_scores = qrels.get(qid, {})
        relevant = {d for d, s in rel_scores.items() if s > 0}
        ids = _ids(results)
        for k in k_values:
            p_at[k].append(precision_at_k(ids, relevant, k))
            r_at[k].append(recall_at_k(ids, relevant, k))
            ndcg_at[k].append(ndcg_at_k(ids, rel_scores, k))
        mrrs.append(mrr(ids, relevant))

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    out: dict[str, float] = {"MRR": _mean(mrrs)}
    for k in k_values:
        out[f"P@{k}"] = _mean(p_at[k])
        out[f"R@{k}"] = _mean(r_at[k])
        out[f"nDCG@{k}"] = _mean(ndcg_at[k])
    return out
