"""Common retrieval interface and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from ..schema import Episode

OutcomeFilter = Literal["success", "failure"] | None


@dataclass
class RetrievalResult:
    episode: Episode
    score: float


class Retriever(Protocol):
    name: str

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_outcome: OutcomeFilter = None,
    ) -> list[RetrievalResult]: ...


def apply_outcome_filter(
    candidates: list[RetrievalResult],
    filter_outcome: OutcomeFilter,
    k: int,
) -> list[RetrievalResult]:
    if filter_outcome is None:
        return candidates[:k]
    keep = [r for r in candidates if r.episode.outcome_label == filter_outcome]
    return keep[:k]
