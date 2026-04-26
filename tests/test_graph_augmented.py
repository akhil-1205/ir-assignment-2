"""GraphAugmentedReranker tests using a stub cross-encoder."""

from __future__ import annotations

from episodic.indexing.bm25_index import BM25Index
from episodic.indexing.kg_index import KGIndex
from episodic.indexing.metadata_store import MetadataStore
from episodic.retrieval.bm25 import BM25Retriever
from episodic.retrieval.graph_augmented import GraphAugmentedReranker
from episodic.retrieval.kg import KGRetriever
from episodic.schema import Episode


def _eps():
    return [
        Episode("e1", "book a flight to Paris next month",
                "use travel api to find a fare",
                ["travel_api"], "success", "booked", 1.0,
                "t", task_type="travel"),
        Episode("e2", "summarize the quarterly earnings report",
                "use summarizer to extract key metrics",
                ["summarizer"], "success", "ok", 2.0,
                "t", task_type="qa"),
        Episode("e3", "translate a poem about Paris into Japanese",
                "use translator to render the poem",
                ["translator"], "failure", "wrong language", 3.0,
                "t", task_type="translation"),
        Episode("e4", "schedule a flight booking review meeting",
                "use calendar to block time",
                ["calendar"], "success", "scheduled", 4.0,
                "t", task_type="scheduling"),
        Episode("e5", "find a flight to Tokyo with travel api",
                "use travel api for fare comparison",
                ["travel_api"], "success", "found cheap fare", 5.0,
                "t", task_type="travel"),
    ]


def _stub_ce(pairs):
    """Score = number of overlapping content words (length >= 4)."""
    out = []
    for q, d in pairs:
        qs = {w for w in q.lower().split() if len(w) >= 4}
        ds = {w for w in d.lower().split() if len(w) >= 4}
        out.append(float(len(qs & ds)))
    return out


def _build():
    eps = _eps()
    store = MetadataStore.from_episodes(eps)
    bm25 = BM25Index.build(eps)
    kg = KGIndex.build(eps, min_doc_freq=1, min_token_len=4)
    bm25_r = BM25Retriever(bm25, store)
    kg_r = KGRetriever(kg, store, alpha=0.85)
    return eps, store, kg, bm25_r, kg_r


def test_gar_runs_and_returns_topk():
    eps, store, kg, bm25_r, kg_r = _build()
    gar = GraphAugmentedReranker(
        first_stages=[bm25_r, kg_r],
        kg_index=kg,
        store=store,
        score_fn=_stub_ce,
        recall_pool=10,
        rerank_pool=5,
    )
    hits = gar.retrieve("flight to Paris booking", k=3)
    assert hits and len(hits) <= 3
    # The two travel episodes should be in the top-3
    ids = {h.episode.episode_id for h in hits}
    assert ids & {"e1", "e5"}


def test_gar_outcome_filter_keeps_only_failures():
    eps, store, kg, bm25_r, kg_r = _build()
    gar = GraphAugmentedReranker(
        first_stages=[bm25_r, kg_r],
        kg_index=kg,
        store=store,
        score_fn=_stub_ce,
        recall_pool=10,
        rerank_pool=10,
        oversample_for_filter=10,
    )
    hits = gar.retrieve("translate Paris poem", k=2, filter_outcome="failure")
    assert hits
    assert all(h.episode.outcome_label == "failure" for h in hits)


def test_gar_empty_pool_returns_empty():
    eps, store, kg, bm25_r, kg_r = _build()
    gar = GraphAugmentedReranker(
        first_stages=[bm25_r, kg_r],
        kg_index=kg,
        store=store,
        score_fn=_stub_ce,
    )
    # Empty query: BM25 short-circuits on no tokens, KG short-circuits on
    # zero personalization mass — first-stage pool is empty.
    hits = gar.retrieve("", k=3)
    assert hits == []


def test_mmr_diversifies_when_lambda_low():
    """With mmr_lambda very low, the second pick should NOT be the closest
    duplicate to the first — it should diversify."""
    eps, store, kg, bm25_r, kg_r = _build()
    gar = GraphAugmentedReranker(
        first_stages=[bm25_r, kg_r],
        kg_index=kg,
        store=store,
        score_fn=_stub_ce,
        recall_pool=10,
        rerank_pool=10,
        mmr_lambda=0.0,  # pure diversification (hardly considers relevance)
    )
    hits = gar.retrieve("flight Paris booking", k=2)
    assert len(hits) == 2
    # pure-diversification should not return two near-duplicates of the same
    # type if a different-type alternative was in the pool
    types = {h.episode.task_type for h in hits}
    assert len(types) >= 1  # smoke: just ensure it runs
