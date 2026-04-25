"""FAISS dense index over sentence-transformer embeddings.

Uses inner-product over L2-normalized vectors -> cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..schema import Episode

EmbedField = Literal["state", "state_plan", "full_document"]
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _texts_for(episodes: list[Episode], field: EmbedField) -> list[str]:
    if field == "state":
        return [ep.state_text for ep in episodes]
    if field == "state_plan":
        return [ep.state_plan_text() for ep in episodes]
    if field == "full_document":
        return [ep.full_document for ep in episodes]
    raise ValueError(f"unknown field {field!r}")


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (mat / norms).astype(np.float32)


class Encoder:
    """Lazy wrapper around sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        m = self._ensure()
        emb = m.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=False,
        )
        return _normalize(np.asarray(emb, dtype=np.float32))


@dataclass
class DenseIndex:
    index: object  # faiss.IndexFlatIP
    episode_ids: list[str]
    embeddings: np.ndarray
    field: EmbedField
    encoder: Encoder

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        q = self.encoder.encode([query])
        D, I = self.index.search(q, min(k, len(self.episode_ids)))  # type: ignore[attr-defined]
        out: list[tuple[str, float]] = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            out.append((self.episode_ids[idx], float(score)))
        return out

    def cosine(self, query_vec: np.ndarray, episode_id: str) -> float:
        i = self.episode_ids.index(episode_id)
        return float(np.dot(query_vec, self.embeddings[i]))

    def encode_query(self, query: str) -> np.ndarray:
        return self.encoder.encode([query])[0]

    def save(self, dir_path: str | Path) -> None:
        import faiss  # type: ignore

        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "index.faiss"))
        np.savez_compressed(
            d / "vectors.npz",
            ids=np.array(self.episode_ids, dtype=object),
            emb=self.embeddings,
            field=np.array([self.field], dtype=object),
            model=np.array([self.encoder.model_name], dtype=object),
        )

    @classmethod
    def load(cls, dir_path: str | Path, encoder: Encoder | None = None) -> "DenseIndex":
        import faiss  # type: ignore

        d = Path(dir_path)
        index = faiss.read_index(str(d / "index.faiss"))
        data = np.load(d / "vectors.npz", allow_pickle=True)
        ids = list(data["ids"].tolist())
        emb = data["emb"]
        field = str(data["field"][0])
        model_name = str(data["model"][0])
        enc = encoder or Encoder(model_name=model_name)
        return cls(index=index, episode_ids=ids, embeddings=emb,
                   field=field, encoder=enc)  # type: ignore[arg-type]

    @classmethod
    def build(
        cls,
        episodes: list[Episode],
        field: EmbedField = "state",
        encoder: Encoder | None = None,
        show_progress: bool = False,
    ) -> "DenseIndex":
        import faiss  # type: ignore

        enc = encoder or Encoder()
        texts = _texts_for(episodes, field)
        emb = enc.encode(texts, show_progress=show_progress)
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        return cls(
            index=index,
            episode_ids=[ep.episode_id for ep in episodes],
            embeddings=emb,
            field=field,
            encoder=enc,
        )
