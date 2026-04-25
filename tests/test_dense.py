"""Dense retrieval test with a stub encoder so tests don't depend on
network access or the sentence-transformers download.
"""

from __future__ import annotations

import numpy as np
import pytest

from episodic.indexing.metadata_store import MetadataStore
from episodic.schema import Episode


@pytest.fixture
def stub_dense_index(monkeypatch):
    faiss = pytest.importorskip("faiss")
    from episodic.indexing.dense_index import DenseIndex, Encoder, _normalize

    eps = [
        Episode("e1", "alpha apple", "p1", ["t1"], "success", "", 1, "t",
                task_type="x"),
        Episode("e2", "beta banana", "p2", ["t2"], "success", "", 2, "t",
                task_type="x"),
        Episode("e3", "gamma grape", "p3", ["t3"], "failure", "", 3, "t",
                task_type="x"),
    ]

    # deterministic stub: each token maps to a 1-hot vector
    vocab = {"alpha": 0, "apple": 1, "beta": 2, "banana": 3,
             "gamma": 4, "grape": 5}
    dim = len(vocab) + 1  # +1 for unknowns

    def _vec(text: str) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        for tok in text.lower().split():
            if tok in vocab:
                v[vocab[tok]] += 1.0
            else:
                v[-1] += 1.0
        return v

    class StubEncoder(Encoder):
        def __init__(self):
            self.model_name = "stub"
            self._model = "stub"
        def encode(self, texts, batch_size=64, show_progress=False):
            mat = np.stack([_vec(t) for t in texts]).astype(np.float32)
            return _normalize(mat)

    enc = StubEncoder()
    texts = [ep.state_text for ep in eps]
    emb = enc.encode(texts)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    di = DenseIndex(
        index=index,
        episode_ids=[ep.episode_id for ep in eps],
        embeddings=emb,
        field="state",
        encoder=enc,
    )
    return di, MetadataStore.from_episodes(eps)


def test_dense_retrieves_token_match(stub_dense_index):
    from episodic.retrieval.dense import DenseRetriever

    idx, store = stub_dense_index
    r = DenseRetriever(idx, store)
    hits = r.retrieve("alpha", k=1)
    assert hits[0].episode.episode_id == "e1"


def test_dense_outcome_filter(stub_dense_index):
    from episodic.retrieval.dense import DenseRetriever

    idx, store = stub_dense_index
    r = DenseRetriever(idx, store, oversample=10)
    hits = r.retrieve("gamma", k=1, filter_outcome="failure")
    assert hits and hits[0].episode.outcome_label == "failure"
