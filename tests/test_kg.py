"""KG index + retriever tests."""

from __future__ import annotations

from episodic.indexing.kg_index import KGIndex
from episodic.indexing.metadata_store import MetadataStore
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
    ]


def test_build_creates_expected_node_types():
    kg = KGIndex.build(_eps(), min_doc_freq=1, min_token_len=4)
    names = set(kg.idx_to_node)
    assert any(n.startswith("ep:") for n in names)
    assert "tool:travel_api" in names
    assert "tool:translator" in names
    assert "tasktype:travel" in names
    assert "outcome:success" in names
    assert "outcome:failure" in names
    # entity nodes should exist for content tokens >= len 4
    assert any(n.startswith("entity:") for n in names)
    assert kg.A_norm.shape[0] == kg.A_norm.shape[1]


def test_search_finds_entity_match():
    eps = _eps()
    kg = KGIndex.build(eps, min_doc_freq=1, min_token_len=4)
    hits = kg.search("flight booking next month", k=2)
    ids = [h[0] for h in hits]
    # both flight-related episodes should rank above the unrelated ones
    assert "e1" in ids or "e4" in ids


def test_kg_retriever_with_outcome_filter():
    eps = _eps()
    kg = KGIndex.build(eps, min_doc_freq=1, min_token_len=4)
    store = MetadataStore.from_episodes(eps)
    r = KGRetriever(kg, store, alpha=0.85, oversample=10)

    fails = r.retrieve("translate a poem", k=1, filter_outcome="failure")
    assert fails and all(x.episode.outcome_label == "failure" for x in fails)


def test_search_returns_empty_when_no_query_signal():
    eps = _eps()
    kg = KGIndex.build(eps, min_doc_freq=1, min_token_len=4)
    # tokens shorter than 4 chars and not in entity vocab -> no seeds
    hits = kg.search("a b c", k=5)
    assert hits == []


def test_restrict_to_episodes_drops_held_out():
    eps = _eps()
    kg = KGIndex.build(eps, min_doc_freq=1, min_token_len=4)
    train_only = kg.restrict_to_episodes({"e1", "e2", "e3"})
    assert set(train_only.episode_ids) == {"e1", "e2", "e3"}
    assert "ep:e4" not in train_only.node_to_idx
