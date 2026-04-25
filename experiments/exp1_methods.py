"""Experiment 1: BM25 vs dense (state) vs dense (state+plan) vs hybrid vs field-aware."""

from __future__ import annotations

import argparse

from _common import evaluate_retriever, load_setup, save_bar_plot, write_table
from episodic.retrieval import (
    BM25Retriever,
    DenseRetriever,
    FieldAwareRetriever,
    HybridRetriever,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Use only the first 30 queries for a fast smoke test.")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    s = load_setup()
    queries = s.queries[:30] if args.quick else s.queries
    print(f"running exp1 on {len(queries)} queries")

    retrievers = [
        BM25Retriever(s.bm25, s.store),
        DenseRetriever(s.dense_state, s.store, name="dense-state"),
        DenseRetriever(s.dense_state_plan, s.store, name="dense-state-plan"),
        HybridRetriever(s.bm25, s.dense_state, s.store, alpha=0.5),
        FieldAwareRetriever(s.bm25, s.dense_state, s.dense_plan, s.store,
                            tool_vocab=s.tool_vocab),
    ]

    rows = []
    for r in retrievers:
        m = evaluate_retriever(r, queries, s.qrels, top_k=max(args.k, 10))
        m["retriever"] = r.name
        print(f"{r.name:30s} P@5={m['P@5']:.3f} R@5={m['R@5']:.3f} "
              f"MRR={m['MRR']:.3f} nDCG@5={m['nDCG@5']:.3f}")
        rows.append(m)

    table = write_table(rows, "results/tables/exp1_methods.csv")
    print(f"wrote {table}")
    plot = save_bar_plot(rows, metric="nDCG@5", label_col="retriever",
                         out_path="results/figures/exp1_methods_ndcg5.png",
                         title="Exp 1: nDCG@5 by retriever")
    print(f"wrote {plot}")


if __name__ == "__main__":
    main()
