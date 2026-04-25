from pathlib import Path

from episodic.schema import Episode, load_episodes, write_jsonl


def _ep(i=0, label="success"):
    return Episode(
        episode_id=f"ep-{i}",
        state_text=f"state {i}",
        plan_text=f"plan {i}",
        tools_used=["a", "b"],
        outcome_label=label,
        outcome_text="done",
        timestamp=1000.0 + i,
        source="test",
        task_type="qa",
    )


def test_full_document_built_from_fields():
    ep = _ep()
    assert "STATE" in ep.full_document
    assert "PLAN" in ep.full_document
    assert "TOOLS" in ep.full_document
    assert "success" in ep.full_document


def test_round_trip_jsonl(tmp_path: Path):
    eps = [_ep(0), _ep(1, label="failure")]
    out = tmp_path / "x.jsonl"
    n = write_jsonl(eps, out)
    assert n == 2
    loaded = load_episodes(out)
    assert len(loaded) == 2
    assert loaded[0].episode_id == "ep-0"
    assert loaded[1].outcome_label == "failure"
