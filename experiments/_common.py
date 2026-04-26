"""Shared experiment scaffolding."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd

from episodic.eval.ir_metrics import evaluate
from episodic.eval.relevance import (
    HeldOutQuery,
    build_qrels,
    build_queries,
    split_episodes,
)
from episodic.indexing.bm25_index import BM25Index
from episodic.indexing.dense_index import DenseIndex, Encoder
from episodic.indexing.kg_index import KGIndex
from episodic.indexing.metadata_store import MetadataStore
from episodic.retrieval.base import Retriever
from episodic.schema import Episode, load_episodes

DEFAULT_EPISODES = "data/processed/episodes.jsonl"
DEFAULT_INDEX_DIR = "data/index"


@dataclass
class Setup:
    train: list[Episode]
    held_out: list[Episode]
    queries: list[HeldOutQuery]
    qrels: dict[str, dict[str, float]]
    store: MetadataStore
    bm25: BM25Index
    dense_state: DenseIndex
    dense_state_plan: DenseIndex
    dense_plan: DenseIndex
    kg: KGIndex
    encoder: Encoder
    tool_vocab: list[str]


def _load_dense(dir_path: Path, encoder: Encoder) -> DenseIndex:
    return DenseIndex.load(dir_path, encoder=encoder)


def _train_dense_from_saved(
    dir_path: Path, train_id_set: set[str], encoder: Encoder
) -> DenseIndex:
    """Load saved embeddings, subset to train IDs, build a fresh FAISS index.

    Avoids re-encoding while guaranteeing the index never returns held-out IDs.
    """
    import faiss  # type: ignore

    data = np.load(dir_path / "vectors.npz", allow_pickle=True)
    all_ids = list(data["ids"].tolist())
    emb_all = data["emb"]
    field = str(data["field"][0])

    keep = [i for i, eid in enumerate(all_ids) if eid in train_id_set]
    ids = [all_ids[i] for i in keep]
    emb = np.ascontiguousarray(emb_all[keep])

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return DenseIndex(
        index=index,
        episode_ids=ids,
        embeddings=emb,
        field=field,  # type: ignore[arg-type]
        encoder=encoder,
    )


def load_setup(
    episodes_path: str = DEFAULT_EPISODES,
    index_dir: str = DEFAULT_INDEX_DIR,
    held_out_frac: float = 0.1,
    mode: str = "any",
    rebuild: bool = False,
) -> Setup:
    """Load episodes, split, build qrels, load indices.

    If `rebuild` is True or indices are missing, rebuilds in-memory indices
    over the train split only (avoids leakage of held-out episodes into
    BM25/dense corpora).
    """
    episodes = load_episodes(episodes_path)
    train, held_out = split_episodes(episodes, held_out_frac=held_out_frac)
    queries = build_queries(held_out, mode=mode)  # type: ignore[arg-type]
    qrels = build_qrels(queries, train)
    store = MetadataStore.from_episodes(train)

    encoder = Encoder()
    idx_dir = Path(index_dir)
    bm25_path = idx_dir / "bm25.pkl"
    dense_state_dir = idx_dir / "dense_state"
    dense_sp_dir = idx_dir / "dense_state_plan"
    dense_plan_dir = idx_dir / "dense_plan"
    kg_path = idx_dir / "kg.pkl"

    has_all = (
        bm25_path.exists()
        and (dense_state_dir / "index.faiss").exists()
        and (dense_sp_dir / "index.faiss").exists()
        and (dense_plan_dir / "index.faiss").exists()
    )

    train_id_set = {ep.episode_id for ep in train}

    if has_all and not rebuild:
        # Saved indices contain the full corpus. For a clean held-out
        # evaluation we always restrict to train: rebuild BM25 from train
        # tokens (cheap) and subset the saved dense embeddings by train
        # IDs (no re-encoding needed).
        bm25 = BM25Index.build(train, field="full_document")
        dense_state = _train_dense_from_saved(dense_state_dir, train_id_set, encoder)
        dense_sp = _train_dense_from_saved(dense_sp_dir, train_id_set, encoder)
        dense_plan = _train_dense_from_saved(dense_plan_dir, train_id_set, encoder)
    else:
        bm25 = BM25Index.build(train, field="full_document")
        dense_state = DenseIndex.build(train, field="state", encoder=encoder)
        dense_sp = DenseIndex.build(train, field="state_plan", encoder=encoder)
        # plan-only index: encode plan text directly
        import faiss  # type: ignore
        plan_texts = [ep.plan_text or ep.state_text for ep in train]
        emb = encoder.encode(plan_texts)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        dense_plan = DenseIndex(
            index=index,
            episode_ids=[ep.episode_id for ep in train],
            embeddings=emb,
            field="state",  # treated as opaque
            encoder=encoder,
        )

    # KG: build over train (graphs are cheap to construct)
    kg = KGIndex.build(train)

    tool_vocab = sorted({t for ep in train for t in ep.tools_used})

    return Setup(
        train=train,
        held_out=held_out,
        queries=queries,
        qrels=qrels,
        store=store,
        bm25=bm25,
        dense_state=dense_state,
        dense_state_plan=dense_sp,
        dense_plan=dense_plan,
        kg=kg,
        encoder=encoder,
        tool_vocab=tool_vocab,
    )


def run_retrieval(retriever: Retriever, queries: list[HeldOutQuery], k: int = 10):
    """Returns dict[query_id] -> list[RetrievalResult]."""
    return {q.query_id: retriever.retrieve(q.text, k=k) for q in queries}


def evaluate_retriever(
    retriever: Retriever,
    queries: list[HeldOutQuery],
    qrels: dict[str, dict[str, float]],
    k_values: Iterable[int] = (1, 3, 5, 10),
    top_k: int = 10,
) -> dict[str, float]:
    runs = run_retrieval(retriever, queries, k=top_k)
    return evaluate(runs, qrels, k_values=k_values)


def write_table(rows: list[dict], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return out


def save_bar_plot(
    rows: list[dict],
    metric: str,
    label_col: str,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(r[label_col]) for r in rows]
    values = [float(r[metric]) for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
