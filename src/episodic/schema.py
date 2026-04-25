"""Episode dataclass and JSONL I/O.

The single source of truth for the three storage formats described in the
plan: a raw concatenated document for BM25, structured fields for the
metadata store and field-aware retriever, and text fragments for dense
embedding.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal

OutcomeLabel = Literal["success", "failure"]


@dataclass
class Episode:
    episode_id: str
    state_text: str
    plan_text: str
    tools_used: list[str]
    outcome_label: OutcomeLabel
    outcome_text: str
    timestamp: float
    source: str
    task_type: str = ""
    full_document: str = field(default="")

    def __post_init__(self) -> None:
        if not self.full_document:
            self.full_document = self._build_document()

    def _build_document(self) -> str:
        tools = " ".join(self.tools_used)
        return (
            f"STATE: {self.state_text}\n"
            f"PLAN: {self.plan_text}\n"
            f"TOOLS: {tools}\n"
            f"OUTCOME ({self.outcome_label}): {self.outcome_text}"
        )

    def state_plan_text(self) -> str:
        return f"{self.state_text} {self.plan_text}".strip()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Episode":
        return cls(**payload)


def write_jsonl(episodes: Iterable[Episode], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Episode]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Episode.from_dict(json.loads(line))


def load_episodes(path: str | Path) -> list[Episode]:
    return list(read_jsonl(path))
