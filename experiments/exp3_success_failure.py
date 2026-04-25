"""Experiment 3: success-only vs failure-only vs combined retrieval.

Evaluates IR metrics for each filter mode AND downstream agent impact.
"""

from __future__ import annotations

import argparse

from _common import evaluate_retriever, load_setup, save_bar_plot, write_table
from episodic.agent.simulated_agent import SimulatedAgent
from episodic.eval.downstream import aggregate, kl_divergence
from episodic.eval.relevance import build_qrels, build_queries
from episodic.retrieval.hybrid import HybridRetriever


class _FilterRetriever:
    def __init__(self, base, mode):
        self.base = base
        self.mode = mode  # "success" | "failure" | None
        self.name = f"hybrid-{mode or 'both'}"

    def retrieve(self, query, k=5, filter_outcome=None):
        # ignore caller's filter; force ours
        return self.base.retrieve(query, k=k, filter_outcome=self.mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    s = load_setup()

    base = HybridRetriever(s.bm25, s.dense_state, s.store, alpha=0.5)

    # IR-side: build mode-aware queries/qrels for each filter
    rows_ir = []
    for mode in ("success", "failure", "any"):
        qs = build_queries(s.held_out, mode=mode)  # type: ignore[arg-type]
        qrels = build_qrels(qs, s.train)
        if args.quick:
            qs = qs[:30]
        retr = _FilterRetriever(base, None if mode == "any" else mode)
        m = evaluate_retriever(retr, qs, qrels, top_k=10)
        m["mode"] = mode
        print(f"[ir] mode={mode:8s} P@5={m['P@5']:.3f} nDCG@5={m['nDCG@5']:.3f}")
        rows_ir.append(m)
    write_table(rows_ir, "results/tables/exp3_success_failure_ir.csv")
    save_bar_plot(rows_ir, metric="nDCG@5", label_col="mode",
                  out_path="results/figures/exp3_ir_ndcg5.png",
                  title="Exp 3: nDCG@5 by retrieval mode")

    # Downstream-side: agent rollouts on held-out tasks
    tasks = s.held_out[:80] if args.quick else s.held_out
    rows_ds = []
    baseline_dist = None
    for label, retriever in [
        ("no-retrieval", None),
        ("success-only", _FilterRetriever(base, "success")),
        ("failure-only", _FilterRetriever(base, "failure")),
        ("both", base),
    ]:
        agent = SimulatedAgent(tool_vocab=s.tool_vocab, retriever=retriever, k=3)
        rollouts = agent.run(tasks)
        m = aggregate(rollouts)
        if baseline_dist is None:
            baseline_dist = m.tool_distribution
        kl = kl_divergence(m.tool_distribution, baseline_dist)
        print(f"[ds] {label:14s} success={m.success_rate:.3f} "
              f"fail-repeat={m.failure_repetition_rate:.3f} KL={kl:.3f}")
        rows_ds.append({
            "agent": label, "n": m.n,
            "success_rate": m.success_rate,
            "failure_repetition_rate": m.failure_repetition_rate,
            "tool_kl_vs_baseline": kl,
        })
    write_table(rows_ds, "results/tables/exp3_success_failure_downstream.csv")
    save_bar_plot(rows_ds, metric="success_rate", label_col="agent",
                  out_path="results/figures/exp3_ds_success.png",
                  title="Exp 3: agent success rate by retrieval mode")


if __name__ == "__main__":
    main()
