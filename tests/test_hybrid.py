"""Hybrid retriever sanity check using a stub dense index."""

from __future__ import annotations

import numpy as np
import pytest

from episodic.indexing.bm25_index import BM25Index
from episodic.indexing.metadata_store import MetadataStore
from episodic.schema import Episode


def _eps():
    return [
        Episode("e1", "alpha apple flight", "use api", ["api"],
                "success", "", 1, "t", task_type="x"),
        Episode("e2", "beta banana train", "use api2", ["api2"],
                "failure", "", 2, "t", task_type="x"),
        Episode("e3", "gamma grape boat", "use api3", ["api3"],
                "success", "", 3, "t", task_type="x"),
    ]


@pytest.fixture
def stub_setup():
    faiss = pytest.importorskip("faiss")
    from episodic.indexing.dense_index import DenseIndex, Encoder, _normalize

    eps = _eps()
    vocab = {"alpha": 0, "apple": 1, "flight": 2, "beta": 3, "banana": 4,
             "train": 5, "gamma": 6, "grape": 7, "boat": 8}
    dim = len(vocab) + 1

    def _vec(text: str) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        for tok in text.lower().split():
            v[vocab.get(tok, len(vocab))] += 1
        return v

    class StubEncoder(Encoder):
        def __init__(self):
            self.model_name = "stub"
            self._model = "stub"
        def encode(self, texts, batch_size=64, show_progress=False):
            return _normalize(np.stack([_vec(t) for t in texts]).astype(np.float32))

    enc = StubEncoder()
    bm25 = BM25Index.build(eps, field="full_document")
    state_emb = enc.encode([ep.state_text for ep in eps])
    state_index = faiss.IndexFlatIP(state_emb.shape[1])
    state_index.add(state_emb)
    di = DenseIndex(
        index=state_index,
        episode_ids=[ep.episode_id for ep in eps],
        embeddings=state_emb,
        field="state",
        encoder=enc,
    )
    return bm25, di, MetadataStore.from_episodes(eps)


def test_hybrid_combines_signals(stub_setup):
    from episodic.retrieval.hybrid import HybridRetriever

    bm25, di, store = stub_setup
    r = HybridRetriever(bm25, di, store, alpha=0.5)
    hits = r.retrieve("alpha apple", k=1)
    assert hits and hits[0].episode.episode_id == "e1"


def test_hybrid_alpha_extremes_match_components(stub_setup):
    from episodic.retrieval.bm25 import BM25Retriever
    from episodic.retrieval.dense import DenseRetriever
    from episodic.retrieval.hybrid import HybridRetriever

    bm25, di, store = stub_setup
    bm25r = BM25Retriever(bm25, store)
    dnr = DenseRetriever(di, store)
    hyb_bm = HybridRetriever(bm25, di, store, alpha=1.0)
    hyb_dn = HybridRetriever(bm25, di, store, alpha=0.0)
    q = "gamma grape"
    assert hyb_bm.retrieve(q, k=1)[0].episode.episode_id == bm25r.retrieve(q, k=1)[0].episode.episode_id
    assert hyb_dn.retrieve(q, k=1)[0].episode.episode_id == dnr.retrieve(q, k=1)[0].episode.episode_id
