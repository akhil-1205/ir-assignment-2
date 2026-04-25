"""Build queries + relevance judgments from a held-out split.

Relevance grading (graded for nDCG, thresholded for P/R/MRR):

    2.0  same task_type AND tool-set Jaccard >= HIGH_OVERLAP with the
         held-out episode's gold tools  (highly relevant)
    1.0  same task_type AND any tool overlap (> 0)                (relevant)
    0.5  same task_type, no tool overlap                           (topical only)
    0.0  not in qrels (different task_type)

`mode` (success / failure) acts as a multiplier: episodes whose
outcome contradicts the requested mode get their grade halved. The
retriever's `filter_outcome` argument carries the actual filter logic.

P@k / R@k / MRR use a strict threshold of 2.0 by default — same-type
distractors no longer trivially saturate precision. nDCG uses the full
graded scale.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from ..schema import Episode

QueryMode = Literal["success", "failure", "any"]

HIGH_OVERLAP = 0.5
DEFAULT_RELEVANCE_THRESHOLD = 2.0


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


@dataclass
class HeldOutQuery:
    query_id: str
    text: str
    task_type: str
    mode: QueryMode
    source_episode_id: str
    gold_tools: list[str] = field(default_factory=list)


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
    return [
        HeldOutQuery(
            query_id=f"q-{ep.episode_id}",
            text=ep.state_text,
            task_type=ep.task_type,
            mode=mode,
            source_episode_id=ep.episode_id,
            gold_tools=list(ep.tools_used),
        )
        for ep in held_out
    ]


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
        gold = set(q.gold_tools)
        rel: dict[str, float] = {}
        for ep in by_type.get(q.task_type, []):
            overlap = _jaccard(gold, set(ep.tools_used))
            if overlap >= HIGH_OVERLAP:
                grade = 2.0
            elif overlap > 0.0:
                grade = 1.0
            else:
                grade = 0.5

            if q.mode == "success" and ep.outcome_label != "success":
                grade *= 0.5
            elif q.mode == "failure" and ep.outcome_label != "failure":
                grade *= 0.5

            rel[ep.episode_id] = grade
        qrels[q.query_id] = rel
    return qrels
