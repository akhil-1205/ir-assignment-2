"""Deterministic rule-based agent that consumes retrieved episodes.

Designed as a measurement instrument for downstream-impact evaluation.
The agent receives a task (held-out episode), optionally consults a
retriever, biases its tool distribution, and selects a tool set. Success
is decided against ground-truth tools_used with bounded noise so the
distribution is not degenerate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..retrieval.base import Retriever
from ..schema import Episode
from .tool_bias import biased_tool_distribution


@dataclass
class AgentRollout:
    task_id: str
    chosen_tools: list[str]
    success: bool
    used_failure_tools: bool
    tool_distribution: dict[str, float] = field(default_factory=dict)
    retrieved_success_ids: list[str] = field(default_factory=list)
    retrieved_failure_ids: list[str] = field(default_factory=list)


class SimulatedAgent:
    def __init__(
        self,
        tool_vocab: list[str],
        retriever: Optional[Retriever] = None,
        k: int = 3,
        n_tools_per_action: int = 2,
        noise: float = 0.1,
        seed: int = 1234,
        beta_success: float = 0.7,
        beta_failure: float = 0.5,
    ):
        self.tool_vocab = tool_vocab
        self.retriever = retriever
        self.k = k
        self.n_tools_per_action = n_tools_per_action
        self.noise = noise
        self.rng = random.Random(seed)
        self.beta_success = beta_success
        self.beta_failure = beta_failure

    def _retrieve_memory(self, task: Episode) -> tuple[list, list]:
        if self.retriever is None:
            return [], []
        s = self.retriever.retrieve(task.state_text, k=self.k, filter_outcome="success")
        f = self.retriever.retrieve(task.state_text, k=self.k, filter_outcome="failure")
        return s, f

    def _pick_tools(self, distribution: dict[str, float]) -> list[str]:
        tools = list(distribution.keys())
        probs = np.array([distribution[t] for t in tools], dtype=np.float64)
        probs = probs / probs.sum()
        n = min(self.n_tools_per_action, len(tools))
        # sample without replacement using numpy with our seeded rng
        seed = self.rng.randint(0, 2**31 - 1)
        local_rng = np.random.default_rng(seed)
        chosen_idx = local_rng.choice(len(tools), size=n, replace=False, p=probs)
        return [tools[i] for i in chosen_idx]

    def act(self, task: Episode) -> AgentRollout:
        successes, failures = self._retrieve_memory(task)
        if successes or failures:
            dist = biased_tool_distribution(
                self.tool_vocab, successes, failures,
                beta_success=self.beta_success, beta_failure=self.beta_failure,
            )
        else:
            n = len(self.tool_vocab)
            dist = {t: 1.0 / n for t in self.tool_vocab} if n else {}

        chosen = self._pick_tools(dist) if dist else []

        # ground-truth success rule: any overlap with the held-out tool set,
        # plus small noise so retrievers are differentiable.
        gold = set(task.tools_used)
        overlap = bool(set(chosen) & gold) if gold else False
        success = overlap and self.rng.random() > self.noise
        if not gold:
            # no gold tools => task counted as success if non-empty action
            success = bool(chosen) and self.rng.random() > self.noise

        # detect "repeated failure" — chose tools that match a known failure episode
        failure_tool_sets = [set(r.episode.tools_used) for r in failures]
        used_failure_tools = any(set(chosen) & fs for fs in failure_tool_sets)

        return AgentRollout(
            task_id=task.episode_id,
            chosen_tools=chosen,
            success=success,
            used_failure_tools=used_failure_tools,
            tool_distribution=dist,
            retrieved_success_ids=[r.episode.episode_id for r in successes],
            retrieved_failure_ids=[r.episode.episode_id for r in failures],
        )

    def run(self, tasks: list[Episode]) -> list[AgentRollout]:
        return [self.act(t) for t in tasks]
