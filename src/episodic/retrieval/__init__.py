"""Retriever implementations.

Submodules import their own optional dependencies (rank_bm25, faiss,
sentence-transformers); we avoid eagerly importing them here so that
modules with no IR-runtime needs (eval.ir_metrics, schema, etc.) can be
used without all optional deps installed.
"""

from .base import OutcomeFilter, RetrievalResult, Retriever, apply_outcome_filter

__all__ = [
    "Retriever",
    "RetrievalResult",
    "OutcomeFilter",
    "apply_outcome_filter",
]


def __getattr__(name):
    if name == "BM25Retriever":
        from .bm25 import BM25Retriever
        return BM25Retriever
    if name == "DenseRetriever":
        from .dense import DenseRetriever
        return DenseRetriever
    if name == "HybridRetriever":
        from .hybrid import HybridRetriever
        return HybridRetriever
    if name == "FieldAwareRetriever":
        from .field_aware import FieldAwareRetriever
        return FieldAwareRetriever
    if name == "KGRetriever":
        from .kg import KGRetriever
        return KGRetriever
    if name == "CrossEncoderReranker":
        from .cross_encoder import CrossEncoderReranker
        return CrossEncoderReranker
    if name == "GraphAugmentedReranker":
        from .graph_augmented import GraphAugmentedReranker
        return GraphAugmentedReranker
    raise AttributeError(name)
