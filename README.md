# Episodic Memory Retrieval for Adaptive Agent Planning

An information-retrieval system over episodic memory for tool-using agents.
The core contribution is the design, comparison, and evaluation of retrieval
strategies (BM25, dense, hybrid, field-aware) measured with both IR metrics
and downstream agent-task metrics.

## Quick start

```bash
pip install -e ".[dev]"
python scripts/download_datasets.py
python scripts/build_indices.py
pytest -q
python experiments/exp1_methods.py --quick
python experiments/run_all.py
```

## Layout

- `src/episodic/` — library code
  - `schema.py` — `Episode` dataclass + JSONL I/O
  - `loaders/` — ACE / MSC loaders
  - `indexing/` — BM25, FAISS, metadata store
  - `retrieval/` — BM25, dense, hybrid, field-aware retrievers
  - `rerank/` — recency / tool-overlap / failure-penalty heuristics
  - `agent/` — simulated rule-based agent + tool-bias + prompt augmentation
  - `eval/` — IR metrics, relevance builder, downstream metrics
- `experiments/` — five controlled studies (methods, representation, success/failure, top-k, weights)
- `scripts/` — dataset download + index build
- `tests/` — pytest unit tests
- `results/` — generated tables + figures

## Stack

- BM25: `rank_bm25`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Dense index: FAISS (CPU)
- Knowledge graph: `scipy.sparse` adjacency + Personalized PageRank
- Datasets: ACE episodic + MSC via HuggingFace `datasets`
- Agent: simulated rule-based policy
