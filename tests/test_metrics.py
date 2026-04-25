from episodic.eval.ir_metrics import (
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert precision_at_k(retrieved, relevant, 1) == 0.0


def test_recall_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"b", "x"}
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert recall_at_k(retrieved, set(), 3) == 0.0


def test_mrr():
    assert mrr(["x", "y", "z"], {"y"}) == 0.5
    assert mrr(["a", "b"], {"c"}) == 0.0
    assert mrr(["a"], {"a"}) == 1.0


def test_ndcg_perfect_ordering_is_one():
    retrieved = ["a", "b", "c"]
    rel = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert abs(ndcg_at_k(retrieved, rel, 3) - 1.0) < 1e-9


def test_ndcg_handles_empty():
    assert ndcg_at_k([], {"a": 1.0}, 5) == 0.0
    assert ndcg_at_k(["a"], {}, 5) == 0.0
