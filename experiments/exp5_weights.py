"""Experiment 5: hybrid alpha sweep and field-aware weight grid."""

from __future__ import annotations

import argparse
from itertools import product

from _common import evaluate_retriever, load_setup, save_bar_plot, write_table
from episodic.retrieval.field_aware import FieldAwareRetriever, FieldWeights
from episodic.retrieval.hybrid import HybridRetriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    s = load_setup()
    queries = s.queries[:30] if args.quick else s.queries

    # alpha sweep
    alpha_rows = []
    for a in (0.3, 0.5, 0.7):
        retr = HybridRetriever(s.bm25, s.dense_state, s.store, alpha=a)
        m = evaluate_retriever(retr, queries, s.qrels, top_k=10)
        m["alpha"] = a
        print(f"alpha={a}: P@5={m['P@5']:.3f} nDCG@5={m['nDCG@5']:.3f}")
        alpha_rows.append(m)
    write_table(alpha_rows, "results/tables/exp5_alpha.csv")
    save_bar_plot(alpha_rows, metric="nDCG@5", label_col="alpha",
                  out_path="results/figures/exp5_alpha_ndcg5.png",
                  title="Exp 5: hybrid alpha vs nDCG@5")

    # field-aware weight grid (small): vary state vs plan emphasis
    fw_rows = []
    grid = [
        FieldWeights(0.5, 0.2, 0.2, 0.1),
        FieldWeights(0.4, 0.4, 0.1, 0.1),
        FieldWeights(0.6, 0.1, 0.2, 0.1),
        FieldWeights(0.3, 0.3, 0.3, 0.1),
    ]
    for w in grid:
        retr = FieldAwareRetriever(s.bm25, s.dense_state, s.dense_plan, s.store,
                                   weights=w, tool_vocab=s.tool_vocab)
        m = evaluate_retriever(retr, queries, s.qrels, top_k=10)
        m["weights"] = f"s={w.w_state},p={w.w_plan},t={w.w_tools},o={w.w_outcome}"
        print(f"{m['weights']}: P@5={m['P@5']:.3f} nDCG@5={m['nDCG@5']:.3f}")
        fw_rows.append(m)
    write_table(fw_rows, "results/tables/exp5_field_weights.csv")
    save_bar_plot(fw_rows, metric="nDCG@5", label_col="weights",
                  out_path="results/figures/exp5_fields_ndcg5.png",
                  title="Exp 5: field-aware weights vs nDCG@5")


if __name__ == "__main__":
    main()
