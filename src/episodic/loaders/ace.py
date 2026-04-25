"""ACE episodic-retrieval loader.

The ACE benchmark publishes tool-use trajectories with task descriptions,
plans, and outcomes. The HF schema has shifted across releases, so we try
the most common field aliases and fall back to a synthetic generator when
the dataset is unreachable. The fallback is deterministic so experiments
remain reproducible offline.
"""

from __future__ import annotations

import hashlib
import random
import time
from typing import Iterable

from ..schema import Episode

ACE_HF_CANDIDATES = (
    ("ace-bench/ace-episodic", None),
    ("ACE/episodic", None),
    ("agent-eval/ace", "episodic"),
)

_ACE_TOOLS = [
    "web_search", "calculator", "code_executor", "file_reader",
    "sql_query", "api_call", "image_classifier", "translator",
    "summarizer", "calendar", "shell", "browser",
]
_ACE_TASK_TYPES = [
    "research", "coding", "data_analysis", "scheduling",
    "debugging", "qa", "extraction", "translation",
]


def _id_for(source: str, idx: int, content: str) -> str:
    h = hashlib.sha1(content.encode("utf-8")).hexdigest()[:8]
    return f"{source}-{idx:06d}-{h}"


def _from_hf_record(idx: int, rec: dict) -> Episode | None:
    """Best-effort adapter across HF schema variants."""
    state = rec.get("state") or rec.get("task") or rec.get("instruction") or rec.get("query")
    plan = rec.get("plan") or rec.get("trajectory") or rec.get("response") or ""
    tools = (
        rec.get("tools_used")
        or rec.get("tools")
        or rec.get("actions")
        or []
    )
    outcome = rec.get("outcome") or rec.get("result") or rec.get("label")
    outcome_text = rec.get("outcome_text") or rec.get("error") or rec.get("explanation") or ""
    task_type = rec.get("task_type") or rec.get("category") or ""

    if isinstance(tools, str):
        tools = [t.strip() for t in tools.split(",") if t.strip()]
    if not isinstance(tools, list):
        tools = list(tools) if tools else []

    if not state:
        return None

    if isinstance(outcome, bool):
        label = "success" if outcome else "failure"
    elif isinstance(outcome, (int, float)):
        label = "success" if float(outcome) > 0 else "failure"
    elif isinstance(outcome, str):
        label = "success" if outcome.lower() in {"success", "true", "1", "ok", "pass"} else "failure"
    else:
        label = "success"

    state_str = str(state)
    return Episode(
        episode_id=_id_for("ace", idx, state_str + str(plan)),
        state_text=state_str,
        plan_text=str(plan),
        tools_used=[str(t) for t in tools],
        outcome_label=label,  # type: ignore[arg-type]
        outcome_text=str(outcome_text),
        timestamp=time.time() - (idx * 60.0),
        source="ace",
        task_type=str(task_type),
    )


def load_from_huggingface(max_records: int | None = None) -> list[Episode]:
    """Try a few likely HF dataset paths. Returns [] if none load."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return []

    for repo, config in ACE_HF_CANDIDATES:
        try:
            ds = load_dataset(repo, config) if config else load_dataset(repo)
        except Exception:
            continue
        # pick the largest split
        split = max(ds.keys(), key=lambda s: len(ds[s]))
        records = ds[split]
        out: list[Episode] = []
        for i, rec in enumerate(records):
            ep = _from_hf_record(i, dict(rec))
            if ep is not None:
                out.append(ep)
            if max_records is not None and len(out) >= max_records:
                break
        if out:
            return out
    return []


def _synthetic(n: int = 800, seed: int = 17) -> list[Episode]:
    rng = random.Random(seed)
    out: list[Episode] = []
    for i in range(n):
        task_type = rng.choice(_ACE_TASK_TYPES)
        n_tools = rng.randint(1, 4)
        tools = rng.sample(_ACE_TOOLS, k=n_tools)
        # success probability depends on tool/task affinity
        affinity = {
            "research": {"web_search", "summarizer"},
            "coding": {"code_executor", "shell"},
            "data_analysis": {"sql_query", "calculator"},
            "scheduling": {"calendar"},
            "debugging": {"shell", "code_executor"},
            "qa": {"web_search", "summarizer"},
            "extraction": {"file_reader", "api_call"},
            "translation": {"translator"},
        }[task_type]
        success = bool(set(tools) & affinity) and rng.random() > 0.15
        outcome = "success" if success else "failure"
        state = f"Task: {task_type} job number {i}. Need to handle {task_type} use case."
        plan = f"Use {', '.join(tools)} to address the {task_type} request step by step."
        outcome_text = (
            f"Completed {task_type} successfully using {tools[0]}."
            if success
            else f"Failed {task_type}: tool {tools[0]} returned an error."
        )
        ep = Episode(
            episode_id=f"ace-syn-{i:06d}",
            state_text=state,
            plan_text=plan,
            tools_used=tools,
            outcome_label=outcome,  # type: ignore[arg-type]
            outcome_text=outcome_text,
            timestamp=time.time() - (n - i) * 3600.0,
            source="ace",
            task_type=task_type,
        )
        out.append(ep)
    return out


def load(max_records: int | None = None, allow_synthetic: bool = True) -> list[Episode]:
    """Load ACE episodes. Falls back to a deterministic synthetic set."""
    eps = load_from_huggingface(max_records=max_records)
    if eps:
        return eps
    if not allow_synthetic:
        return []
    return _synthetic(n=max_records or 800)
