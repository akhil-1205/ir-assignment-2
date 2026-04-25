"""Experiment 4: top-k sensitivity for the hybrid retriever."""

from __future__ import annotations

import argparse

from _common import evaluate_retriever, load_setup, save_bar_plot, write_table
from episodic.retrieval.hybrid import HybridRetriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    s = load_setup()
    queries = s.queries[:30] if args.quick else s.queries
    retriever = HybridRetriever(s.bm25, s.dense_state, s.store, alpha=0.5)

    rows = []
    for k in (1, 3, 5, 10):
        m = evaluate_retriever(retriever, queries, s.qrels, k_values=(k,), top_k=max(k, 10))
        row = {"k": k, "P@k": m[f"P@{k}"], "R@k": m[f"R@{k}"],
               "nDCG@k": m[f"nDCG@{k}"], "MRR": m["MRR"]}
        print(f"k={k}: P={row['P@k']:.3f} R={row['R@k']:.3f} nDCG={row['nDCG@k']:.3f}")
        rows.append(row)

    write_table(rows, "results/tables/exp4_topk.csv")
    save_bar_plot(rows, metric="nDCG@k", label_col="k",
                  out_path="results/figures/exp4_topk_ndcg.png",
                  title="Exp 4: nDCG@k vs k (hybrid)")


if __name__ == "__main__":
    main()
