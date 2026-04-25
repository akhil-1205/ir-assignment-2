"""Smoke test for tool bias and simulated agent."""

from episodic.agent.simulated_agent import SimulatedAgent
from episodic.agent.tool_bias import biased_tool_distribution
from episodic.retrieval.base import RetrievalResult
from episodic.schema import Episode


def _ep(eid, tools, outcome):
    return Episode(eid, "s", "p", tools, outcome, "", 1, "t", task_type="qa")


def test_tool_bias_amplifies_success_tools():
    vocab = ["a", "b", "c"]
    successes = [RetrievalResult(_ep("s1", ["a"], "success"), 1.0)]
    failures = [RetrievalResult(_ep("f1", ["c"], "failure"), 1.0)]
    dist = biased_tool_distribution(vocab, successes, failures)
    assert dist["a"] > dist["b"] > dist["c"]
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_simulated_agent_runs_without_retriever():
    vocab = ["a", "b"]
    agent = SimulatedAgent(tool_vocab=vocab, retriever=None,
                           n_tools_per_action=1, noise=0.0, seed=0)
    rollouts = agent.run([_ep("t1", ["a"], "success")])
    assert len(rollouts) == 1
    assert rollouts[0].chosen_tools[0] in vocab
