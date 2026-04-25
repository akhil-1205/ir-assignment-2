"""BM25 lexical index over Episode.full_document."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ..schema import Episode
from .tokenize import tokenize


@dataclass
class BM25Index:
    bm25: BM25Okapi
    episode_ids: list[str]

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        if k >= len(scores):
            top = np.argsort(-scores)
        else:
            top = np.argpartition(-scores, k)[:k]
            top = top[np.argsort(-scores[top])]
        return [(self.episode_ids[i], float(scores[i])) for i in top[:k]]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump({"bm25": self.bm25, "ids": self.episode_ids}, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        return cls(bm25=obj["bm25"], episode_ids=obj["ids"])

    @classmethod
    def build(cls, episodes: list[Episode], field: str = "full_document") -> "BM25Index":
        corpus_tokens: list[list[str]] = []
        ids: list[str] = []
        for ep in episodes:
            text = getattr(ep, field, None)
            if text is None and field == "state_plan":
                text = ep.state_plan_text()
            if text is None:
                text = ep.full_document
            corpus_tokens.append(tokenize(str(text)))
            ids.append(ep.episode_id)
        bm25 = BM25Okapi(corpus_tokens)
        return cls(bm25=bm25, episode_ids=ids)
