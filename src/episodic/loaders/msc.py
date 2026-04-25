"""MSC (Multi-Session Chat) memory dataset loader.

MSC episodes are conversation turns annotated with persona / memory facts.
We project each session into an Episode where:
  - state_text  = user query / situation in the session
  - plan_text   = response trajectory or memory-grounded plan
  - tools_used  = derived from any tool calls, else simulated based on topic
  - outcome     = success/failure heuristic from response quality signals
"""

from __future__ import annotations

import hashlib
import random
import time

from ..schema import Episode

MSC_HF_CANDIDATES = (
    ("facebook/msc", None),
    ("msc", None),
    ("ParlAI/msc", None),
)

_MSC_TOPICS = [
    "personal_qa", "recommendation", "memory_recall",
    "scheduling", "small_talk", "follow_up",
]
_MSC_TOOLS = [
    "memory_lookup", "user_profile", "calendar", "knowledge_base",
    "summarizer", "web_search",
]


def _id_for(idx: int, content: str) -> str:
    h = hashlib.sha1(content.encode("utf-8")).hexdigest()[:8]
    return f"msc-{idx:06d}-{h}"


def _record_to_episode(idx: int, rec: dict) -> Episode | None:
    dialog = rec.get("dialog") or rec.get("turns") or rec.get("messages")
    persona = rec.get("personas") or rec.get("persona") or []
    if not dialog:
        return None

    if isinstance(dialog, list) and dialog and isinstance(dialog[0], dict):
        utts = [str(t.get("text") or t.get("content") or "") for t in dialog]
    else:
        utts = [str(t) for t in dialog]

    if len(utts) < 2:
        return None

    state = utts[-2]
    plan = utts[-1]
    tools = ["memory_lookup"]
    if persona:
        tools.append("user_profile")

    # heuristic: longer, persona-grounded responses are "successes"
    success = len(plan.split()) >= 8 and bool(persona)
    label = "success" if success else "failure"

    content = state + plan
    return Episode(
        episode_id=_id_for(idx, content),
        state_text=state,
        plan_text=plan,
        tools_used=tools,
        outcome_label=label,  # type: ignore[arg-type]
        outcome_text="grounded response" if success else "weak/ungrounded response",
        timestamp=time.time() - idx * 60.0,
        source="msc",
        task_type="memory_recall",
    )


def load_from_huggingface(max_records: int | None = None) -> list[Episode]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return []

    for repo, config in MSC_HF_CANDIDATES:
        try:
            ds = load_dataset(repo, config) if config else load_dataset(repo)
        except Exception:
            continue
        split = max(ds.keys(), key=lambda s: len(ds[s]))
        records = ds[split]
        out: list[Episode] = []
        for i, rec in enumerate(records):
            ep = _record_to_episode(i, dict(rec))
            if ep is not None:
                out.append(ep)
            if max_records is not None and len(out) >= max_records:
                break
        if out:
            return out
    return []


def _synthetic(n: int = 400, seed: int = 23) -> list[Episode]:
    rng = random.Random(seed)
    out: list[Episode] = []
    for i in range(n):
        topic = rng.choice(_MSC_TOPICS)
        n_tools = rng.randint(1, 3)
        tools = rng.sample(_MSC_TOOLS, k=n_tools)
        success = "memory_lookup" in tools and rng.random() > 0.25
        label = "success" if success else "failure"
        state = f"User asks about {topic} referencing earlier session {i}."
        plan = (
            f"Recall prior session, query {tools[0]}, answer using grounded fact."
            if success
            else f"Attempted with {tools[0]} but missed prior context."
        )
        out.append(
            Episode(
                episode_id=f"msc-syn-{i:06d}",
                state_text=state,
                plan_text=plan,
                tools_used=tools,
                outcome_label=label,  # type: ignore[arg-type]
                outcome_text="recalled fact" if success else "context missed",
                timestamp=time.time() - (n - i) * 3600.0,
                source="msc",
                task_type=topic,
            )
        )
    return out


def load(max_records: int | None = None, allow_synthetic: bool = True) -> list[Episode]:
    eps = load_from_huggingface(max_records=max_records)
    if eps:
        return eps
    if not allow_synthetic:
        return []
    return _synthetic(n=max_records or 400)
