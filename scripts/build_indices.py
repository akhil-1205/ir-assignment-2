"""Build BM25 + FAISS indices from data/processed/episodes.jsonl.

Outputs:
  data/index/bm25.pkl
  data/index/dense_state/{index.faiss,vectors.npz}
  data/index/dense_state_plan/{index.faiss,vectors.npz}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from episodic.indexing.bm25_index import BM25Index
from episodic.indexing.dense_index import DenseIndex, Encoder
from episodic.indexing.kg_index import KGIndex
from episodic.schema import load_episodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="data/processed/episodes.jsonl")
    ap.add_argument("--out", default="data/index")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    episodes = load_episodes(args.episodes)
    print(f"loaded {len(episodes)} episodes")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[bm25] building over full_document ...")
    bm25 = BM25Index.build(episodes, field="full_document")
    bm25.save(out / "bm25.pkl")
    print(f"[bm25] saved -> {out/'bm25.pkl'}")

    encoder = Encoder(model_name=args.model)

    print("[dense] building state index ...")
    state_idx = DenseIndex.build(episodes, field="state", encoder=encoder, show_progress=True)
    state_idx.save(out / "dense_state")
    print(f"[dense] state -> {out/'dense_state'}")

    print("[dense] building state+plan index ...")
    sp_idx = DenseIndex.build(episodes, field="state_plan", encoder=encoder, show_progress=True)
    sp_idx.save(out / "dense_state_plan")
    print(f"[dense] state_plan -> {out/'dense_state_plan'}")

    print("[dense] building plan index (for field-aware) ...")
    # Plan-only index built from full_document fallback when plan_text empty
    plan_eps = [ep for ep in episodes]
    # Reuse Encoder but encode plan_text directly
    from episodic.indexing.dense_index import _normalize, _texts_for  # type: ignore
    import numpy as np
    import faiss  # type: ignore

    plan_texts = [ep.plan_text or ep.state_text for ep in plan_eps]
    emb = encoder.encode(plan_texts, show_progress=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    plan_dir = out / "dense_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(plan_dir / "index.faiss"))
    np.savez_compressed(
        plan_dir / "vectors.npz",
        ids=np.array([ep.episode_id for ep in plan_eps], dtype=object),
        emb=emb,
        field=np.array(["plan"], dtype=object),
        model=np.array([encoder.model_name], dtype=object),
    )
    print(f"[dense] plan -> {plan_dir}")

    print("[kg] building knowledge graph ...")
    kg = KGIndex.build(episodes)
    kg.save(out / "kg.pkl")
    print(f"[kg] saved -> {out/'kg.pkl'} "
          f"(nodes={len(kg.idx_to_node)}, "
          f"entities={len(kg.entity_idf)}, "
          f"tools={len(kg.tool_vocab)})")

    print("done.")


if __name__ == "__main__":
    main()
