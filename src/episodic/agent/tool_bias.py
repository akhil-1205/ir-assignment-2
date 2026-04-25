"""Adjust tool prior using retrieved success/failure episodes."""

from __future__ import annotations

import numpy as np

from ..retrieval.base import RetrievalResult


def biased_tool_distribution(
    tool_vocab: list[str],
    successes: list[RetrievalResult],
    failures: list[RetrievalResult],
    beta_success: float = 0.7,
    beta_failure: float = 0.5,
) -> dict[str, float]:
    """Return a (renormalized) probability over tool_vocab.

    Tools appearing in retrieved success episodes are amplified;
    tools from failure episodes are dampened.
    """
    if not tool_vocab:
        return {}
    n = len(tool_vocab)
    prior = np.full(n, 1.0 / n)
    idx = {t: i for i, t in enumerate(tool_vocab)}

    for r in successes:
        for t in r.episode.tools_used:
            if t in idx:
                prior[idx[t]] *= 1.0 + beta_success

    for r in failures:
        for t in r.episode.tools_used:
            if t in idx:
                prior[idx[t]] *= max(1.0 - beta_failure, 1e-3)

    total = prior.sum()
    if total <= 0:
        prior = np.full(n, 1.0 / n)
    else:
        prior /= total
    return {t: float(prior[i]) for t, i in idx.items()}
