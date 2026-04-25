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


_ACE_AFFINITY = {
    "research": {"web_search", "summarizer"},
    "coding": {"code_executor", "shell"},
    "data_analysis": {"sql_query", "calculator"},
    "scheduling": {"calendar"},
    "debugging": {"shell", "code_executor"},
    "qa": {"web_search", "summarizer"},
    "extraction": {"file_reader", "api_call"},
    "translation": {"translator"},
}

# Templates per task_type that DO NOT contain the task_type literal.
# Each template has slots filled from a per-type entity pool, and
# wording varies. Lexical overlap across types is intentional so BM25
# alone cannot resolve task_type from the surface form.
_ACE_TEMPLATES = {
    "research": [
        "Find recent material about {topic} and condense the key points.",
        "Look up {topic} across reliable sources and write a short brief.",
        "Pull together what is known about {topic} from the open web.",
    ],
    "coding": [
        "Implement a function that {action} for inputs of {entity}.",
        "Write code to {action} given {entity} as the structure.",
        "Build a small program that handles {entity} and {action}.",
    ],
    "data_analysis": [
        "Compute aggregate statistics over the {entity} table for {topic}.",
        "Analyze trends in {entity} grouped by {topic}.",
        "Run a query to summarize {topic} from the {entity} dataset.",
    ],
    "scheduling": [
        "Set up a meeting on {topic} with the team next week.",
        "Find a slot to discuss {topic} between {entity}.",
        "Block off time for a recurring sync about {topic}.",
    ],
    "debugging": [
        "Investigate why {entity} fails when {action}.",
        "Trace the cause of the error in {entity} during {action}.",
        "Reproduce the issue where {action} on {entity} returns wrong output.",
    ],
    "qa": [
        "Answer the question: which {entity} is associated with {topic}?",
        "Look up the answer to a query about {topic} involving {entity}.",
        "Provide a short, sourced answer about {topic}.",
    ],
    "extraction": [
        "Pull structured fields about {topic} out of the {entity} document.",
        "Read the {entity} file and extract every reference to {topic}.",
        "Parse {entity} and return a list of {topic} entries.",
    ],
    "translation": [
        "Translate the passage about {topic} into another language.",
        "Render the {entity} content into the target locale.",
        "Convert the {topic} text from one language to another.",
    ],
}

_ACE_ENTITIES = {
    "research": ["renewables", "graph databases", "climate policy", "vector search",
                 "supply chains", "rare earths", "robotics safety"],
    "coding": ["a binary tree", "a CSV reader", "a rate limiter", "a hash map",
               "a websocket client", "a retry loop"],
    "data_analysis": ["sales", "users", "events", "orders", "logins", "transactions"],
    "scheduling": ["the design review", "the quarterly plan", "onboarding",
                   "the offsite", "code freeze"],
    "debugging": ["the auth service", "the indexer", "the payment job",
                  "the search ranker", "the cache layer"],
    "qa": ["paper", "report", "manual", "spec", "ticket"],
    "extraction": ["invoice", "contract PDF", "earnings statement",
                   "transcript", "support email"],
    "translation": ["product page", "research abstract", "release notes",
                    "user manual", "blog post"],
}

_ACE_TOPICS = [
    "latency", "Q3 results", "Paris", "Tokyo", "GPU usage", "billing",
    "compliance", "throughput", "an outage", "an SLA breach", "deprecation",
    "a migration", "a refactor", "feature flags",
]

_ACE_ACTIONS = [
    "deduplicates entries", "validates input", "merges streams",
    "renders the result", "ranks candidates", "logs structured events",
    "retries with backoff", "reads a partial file",
]


def _synthetic(n: int = 800, seed: int = 17, distractor_frac: float = 0.18) -> list[Episode]:
    """Harder synthetic ACE.

    Changes vs. v0:
      - state_text never contains the task_type literal
      - per-type templates + entity/topic/action pools produce lexical variation
      - ~18% of episodes are *distractors*: state borrowed from one task_type
        but labeled (and tooled) for a different one. Lexically similar but
        actually irrelevant — BM25 will rank them up; better retrievers
        should not.
      - tool sets vary widely within a task_type so tool-Jaccard relevance
        is meaningful.
    """
    rng = random.Random(seed)
    out: list[Episode] = []
    for i in range(n):
        is_distractor = rng.random() < distractor_frac
        true_type = rng.choice(_ACE_TASK_TYPES)
        text_type = rng.choice([t for t in _ACE_TASK_TYPES if t != true_type]) if is_distractor else true_type

        template = rng.choice(_ACE_TEMPLATES[text_type])
        state = template.format(
            topic=rng.choice(_ACE_TOPICS),
            entity=rng.choice(_ACE_ENTITIES[text_type]),
            action=rng.choice(_ACE_ACTIONS),
        )

        # Tool set: 60% from affinity, 40% random — within the TRUE type
        affinity = _ACE_AFFINITY[true_type]
        n_tools = rng.randint(1, 4)
        tools: list[str] = []
        if rng.random() < 0.7 and affinity:
            tools.append(rng.choice(list(affinity)))
        while len(tools) < n_tools:
            t = rng.choice(_ACE_TOOLS)
            if t not in tools:
                tools.append(t)

        success = bool(set(tools) & affinity) and rng.random() > 0.2
        outcome = "success" if success else "failure"
        plan = f"Run {tools[0]} first; chain {', '.join(tools[1:]) or 'no further tools'}; verify output."
        outcome_text = (
            f"Returned a usable result via {tools[0]}."
            if success
            else f"{tools[0]} produced an error or wrong output; retry needed."
        )
        out.append(
            Episode(
                episode_id=f"ace-syn-{i:06d}",
                state_text=state,
                plan_text=plan,
                tools_used=tools,
                outcome_label=outcome,  # type: ignore[arg-type]
                outcome_text=outcome_text,
                timestamp=time.time() - (n - i) * 3600.0,
                source="ace",
                task_type=true_type,
            )
        )
    return out


def load(max_records: int | None = None, allow_synthetic: bool = True) -> list[Episode]:
    """Load ACE episodes. Falls back to a deterministic synthetic set."""
    eps = load_from_huggingface(max_records=max_records)
    if eps:
        return eps
    if not allow_synthetic:
        return []
    return _synthetic(n=max_records or 800)
