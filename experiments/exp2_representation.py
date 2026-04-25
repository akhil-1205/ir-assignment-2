"""Experiment 2: state vs state+plan vs full document, dense retriever."""

from __future__ import annotations

import argparse

from _common import evaluate_retriever, load_setup, save_bar_plot, write_table
from episodic.indexing.dense_index import DenseIndex
from episodic.retrieval.dense import DenseRetriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    s = load_setup()
    queries = s.queries[:30] if args.quick else s.queries

    # state and state_plan indices already loaded by setup; build full_document
    full_idx = DenseIndex.build(s.train, field="full_document", encoder=s.encoder)

    retrievers = [
        DenseRetriever(s.dense_state, s.store, name="state"),
        DenseRetriever(s.dense_state_plan, s.store, name="state+plan"),
        DenseRetriever(full_idx, s.store, name="full_document"),
    ]

    rows = []
    for r in retrievers:
        m = evaluate_retriever(r, queries, s.qrels, top_k=10)
        m["representation"] = r.name
        print(f"{r.name:18s} P@5={m['P@5']:.3f} nDCG@5={m['nDCG@5']:.3f}")
        rows.append(m)

    write_table(rows, "results/tables/exp2_representation.csv")
    save_bar_plot(rows, metric="nDCG@5", label_col="representation",
                  out_path="results/figures/exp2_representation_ndcg5.png",
                  title="Exp 2: nDCG@5 by document representation")


if __name__ == "__main__":
    main()
