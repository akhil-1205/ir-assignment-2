"""Pure-function reranking heuristics composable on RetrievalResult lists."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Literal

from ..retrieval.base import RetrievalResult


def recency_boost(
    score: float, timestamp: float, now: float | None = None, half_life_seconds: float = 30 * 86400.0
) -> float:
    now = now if now is not None else time.time()
    age = max(0.0, now - timestamp)
    decay = 0.5 ** (age / half_life_seconds)
    # multiplicative boost in [0.5, 1.0] then mapped to a small additive lift
    return score * (0.5 + 0.5 * decay)


def failure_penalty(
    score: float, episode_outcome: str, mode: Literal["success", "failure"], factor: float = 0.5
) -> float:
    if mode == "success" and episode_outcome == "failure":
        return score * factor
    if mode == "failure" and episode_outcome == "success":
        return score * factor
    return score


def tool_match_boost(
    score: float, query_tools: Iterable[str], episode_tools: Iterable[str], factor: float = 1.2
) -> float:
    if not set(query_tools):
        return score
    if set(query_tools) & set(episode_tools):
        return score * factor
    return score


@dataclass
class RerankConfig:
    use_recency: bool = True
    use_failure_penalty: bool = False
    failure_mode: Literal["success", "failure"] = "success"
    use_tool_boost: bool = False
    half_life_seconds: float = 30 * 86400.0
    failure_factor: float = 0.5
    tool_factor: float = 1.2


def apply_rerank(
    results: list[RetrievalResult],
    cfg: RerankConfig,
    query_tools: Iterable[str] = (),
    now: float | None = None,
) -> list[RetrievalResult]:
    if not results:
        return results
    out: list[RetrievalResult] = []
    for r in results:
        s = r.score
        if cfg.use_recency:
            s = recency_boost(s, r.episode.timestamp, now=now,
                              half_life_seconds=cfg.half_life_seconds)
        if cfg.use_failure_penalty:
            s = failure_penalty(s, r.episode.outcome_label, cfg.failure_mode,
                                factor=cfg.failure_factor)
        if cfg.use_tool_boost:
            s = tool_match_boost(s, query_tools, r.episode.tools_used,
                                 factor=cfg.tool_factor)
        out.append(RetrievalResult(episode=r.episode, score=s))
    out.sort(key=lambda r: -r.score)
    return out
