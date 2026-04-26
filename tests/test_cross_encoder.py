"""Cross-encoder reranker test using a deterministic stub scorer.

Avoids downloading the real cross-encoder during tests.
"""

from __future__ import annotations

from episodic.indexing.bm25_index import BM25Index
from episodic.indexing.metadata_store import MetadataStore
from episodic.retrieval.bm25 import BM25Retriever
from episodic.retrieval.cross_encoder import CrossEncoderReranker
from episodic.schema import Episode


def _eps():
    return [
        Episode("e1", "alpha apple flight", "use api", ["api"],
                "success", "", 1, "t", task_type="x"),
        Episode("e2", "beta banana train", "use api2", ["api2"],
                "failure", "", 2, "t", task_type="x"),
        Episode("e3", "gamma grape boat", "use api3", ["api3"],
                "success", "", 3, "t", task_type="x"),
        Episode("e4", "delta date plane", "use api4", ["api4"],
                "success", "", 4, "t", task_type="x"),
    ]


def _stub_scorer(pairs):
    """Score is # of overlapping content words between query and doc."""
    out = []
    for q, d in pairs:
        qs = set(q.lower().split())
        ds = set(d.lower().split())
        out.append(float(len(qs & ds)))
    return out


def test_rerank_promotes_better_candidate():
    eps = _eps()
    store = MetadataStore.from_episodes(eps)
    bm25 = BM25Index.build(eps)
    first = BM25Retriever(bm25, store)
    ce = CrossEncoderReranker(first_stage=first, rerank_pool=4, score_fn=_stub_scorer)

    hits = ce.retrieve("alpha apple flight", k=1)
    assert hits and hits[0].episode.episode_id == "e1"


def test_rerank_outcome_filter():
    eps = _eps()
    store = MetadataStore.from_episodes(eps)
    bm25 = BM25Index.build(eps)
    first = BM25Retriever(bm25, store, oversample=10)
    ce = CrossEncoderReranker(
        first_stage=first, rerank_pool=10, score_fn=_stub_scorer,
        oversample_for_filter=10,
    )

    hits = ce.retrieve("beta banana train", k=1, filter_outcome="failure")
    assert hits and hits[0].episode.outcome_label == "failure"


def test_empty_first_stage_yields_empty():
    eps = _eps()
    store = MetadataStore.from_episodes(eps)
    bm25 = BM25Index.build(eps)
    first = BM25Retriever(bm25, store)
    ce = CrossEncoderReranker(first_stage=first, rerank_pool=4, score_fn=_stub_scorer)

    # query has no content tokens our tokenizer keeps -> empty first stage
    hits = ce.retrieve("", k=3)
    assert hits == []
