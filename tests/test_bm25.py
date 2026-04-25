from episodic.indexing.bm25_index import BM25Index
from episodic.indexing.metadata_store import MetadataStore
from episodic.retrieval.bm25 import BM25Retriever
from episodic.schema import Episode


def _eps():
    return [
        Episode("e1", "book a flight to Paris", "use travel api", ["travel_api"],
                "success", "booked", 1.0, "t", task_type="travel"),
        Episode("e2", "summarize earnings report", "use summarizer",
                ["summarizer"], "success", "ok", 2.0, "t", task_type="qa"),
        Episode("e3", "translate a poem", "use translator", ["translator"],
                "failure", "wrong language", 3.0, "t", task_type="translation"),
    ]


def test_bm25_returns_relevant_first():
    eps = _eps()
    idx = BM25Index.build(eps)
    hits = idx.search("flight Paris", k=3)
    assert hits, "expected at least one hit"
    assert hits[0][0] == "e1"


def test_bm25_retriever_with_filter():
    eps = _eps()
    idx = BM25Index.build(eps)
    store = MetadataStore.from_episodes(eps)
    r = BM25Retriever(idx, store)

    fail = r.retrieve("translate", k=2, filter_outcome="failure")
    assert all(x.episode.outcome_label == "failure" for x in fail)

    succ = r.retrieve("flight", k=2, filter_outcome="success")
    assert all(x.episode.outcome_label == "success" for x in succ)
