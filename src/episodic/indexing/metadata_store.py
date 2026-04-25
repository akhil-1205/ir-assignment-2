"""Metadata store: episode_id -> Episode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..schema import Episode, load_episodes, write_jsonl


@dataclass
class MetadataStore:
    by_id: dict[str, Episode]

    def __len__(self) -> int:
        return len(self.by_id)

    def get(self, episode_id: str) -> Episode:
        return self.by_id[episode_id]

    def all_episodes(self) -> list[Episode]:
        return list(self.by_id.values())

    def filter_outcome(self, label: str) -> list[Episode]:
        return [ep for ep in self.by_id.values() if ep.outcome_label == label]

    @classmethod
    def from_episodes(cls, episodes: list[Episode]) -> "MetadataStore":
        return cls(by_id={ep.episode_id: ep for ep in episodes})

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "MetadataStore":
        return cls.from_episodes(load_episodes(path))

    def save(self, path: str | Path) -> None:
        write_jsonl(self.by_id.values(), path)
