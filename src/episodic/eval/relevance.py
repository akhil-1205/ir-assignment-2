"""Build queries + relevance judgments from a held-out split.

Relevance rule (per plan §7.1):
  An episode is relevant to a held-out query if
    (a) task_type matches, AND
    (b) outcome label is informative for the requested mode.

We grade with a simple two-level scheme: 2 if (a) and (b), 1 if only (a),
0 otherwise — works for nDCG and threshold-based P/R.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from ..schema import Episode

QueryMode = Literal["success", "failure", "any"]


@dataclass
class HeldOutQuery:
    query_id: str
    text: str
    task_type: str
    mode: QueryMode
    source_episode_id: str  # the held-out episode this query was derived from


def split_episodes(
    episodes: list[Episode],
    held_out_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[Episode], list[Episode]]:
    rng = random.Random(seed)
    shuffled = episodes[:]
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * held_out_frac))
    return shuffled[cut:], shuffled[:cut]


def build_queries(
    held_out: list[Episode],
    mode: QueryMode = "any",
) -> list[HeldOutQuery]:
    queries: list[HeldOutQuery] = []
    for ep in held_out:
        queries.append(
            HeldOutQuery(
                query_id=f"q-{ep.episode_id}",
                text=ep.state_text,
                task_type=ep.task_type,
                mode=mode,
                source_episode_id=ep.episode_id,
            )
        )
    return queries


def build_qrels(
    queries: list[HeldOutQuery],
    corpus: list[Episode],
) -> dict[str, dict[str, float]]:
    """Returns qrels[query_id] -> {episode_id: graded_relevance}."""
    by_type: dict[str, list[Episode]] = {}
    for ep in corpus:
        by_type.setdefault(ep.task_type, []).append(ep)

    qrels: dict[str, dict[str, float]] = {}
    for q in queries:
        rel: dict[str, float] = {}
        for ep in by_type.get(q.task_type, []):
            type_match = True
            outcome_match = (
                q.mode == "any"
                or (q.mode == "success" and ep.outcome_label == "success")
                or (q.mode == "failure" and ep.outcome_label == "failure")
            )
            if type_match and outcome_match:
                rel[ep.episode_id] = 2.0
            elif type_match:
                rel[ep.episode_id] = 1.0
        qrels[q.query_id] = rel
    return qrels
