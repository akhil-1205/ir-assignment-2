"""Downstream agent-task metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log

from ..agent.simulated_agent import AgentRollout


@dataclass
class DownstreamMetrics:
    n: int
    success_rate: float
    failure_repetition_rate: float
    tool_distribution: dict[str, float]


def _norm_counter(c: Counter) -> dict[str, float]:
    total = sum(c.values()) or 1
    return {k: v / total for k, v in c.items()}


def aggregate(rollouts: list[AgentRollout]) -> DownstreamMetrics:
    n = len(rollouts)
    if n == 0:
        return DownstreamMetrics(0, 0.0, 0.0, {})

    successes = sum(1 for r in rollouts if r.success)
    failed = [r for r in rollouts if not r.success]
    repeated = sum(1 for r in failed if r.used_failure_tools)
    repetition_rate = repeated / len(failed) if failed else 0.0

    tools = Counter()
    for r in rollouts:
        tools.update(r.chosen_tools)

    return DownstreamMetrics(
        n=n,
        success_rate=successes / n,
        failure_repetition_rate=repetition_rate,
        tool_distribution=_norm_counter(tools),
    )


def kl_divergence(p: dict[str, float], q: dict[str, float], eps: float = 1e-9) -> float:
    """KL(p || q) over the union of tool keys."""
    keys = set(p) | set(q)
    out = 0.0
    for k in keys:
        pk = p.get(k, 0.0) + eps
        qk = q.get(k, 0.0) + eps
        out += pk * log(pk / qk)
    return float(out)
