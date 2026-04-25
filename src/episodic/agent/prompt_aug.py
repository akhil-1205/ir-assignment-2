"""Format retrieved episodes into a 'memory context' the agent can read."""

from __future__ import annotations

from ..retrieval.base import RetrievalResult


def _fmt(r: RetrievalResult) -> str:
    ep = r.episode
    return (
        f"[score={r.score:.3f} type={ep.task_type} outcome={ep.outcome_label}]\n"
        f"  state: {ep.state_text}\n"
        f"  plan:  {ep.plan_text}\n"
        f"  tools: {', '.join(ep.tools_used)}\n"
        f"  note:  {ep.outcome_text}"
    )


def build_memory_context(
    successes: list[RetrievalResult],
    failures: list[RetrievalResult],
) -> str:
    parts: list[str] = []
    if successes:
        parts.append("=== WHAT HAS WORKED ===")
        parts.extend(_fmt(r) for r in successes)
    if failures:
        parts.append("=== WHAT TO AVOID ===")
        parts.extend(_fmt(r) for r in failures)
    return "\n".join(parts)
