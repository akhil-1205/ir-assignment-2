# Handover Guide — Episodic Memory Retrieval System

This document is meant to onboard the next person taking over this project.
Read it top-to-bottom once, then keep it open as a reference while you work.

---

## 0. What this project is

An **Information Retrieval system over episodic memory** for a tool-using
agent. The contribution is the **design, comparison, and evaluation of
retrieval strategies** — not the agent itself. The agent is a deliberately
simple, deterministic instrument used to measure whether retrieval
actually helps a downstream policy do better.

Everything in this repo serves one of three roles:

1. **Build** an episodic-memory corpus (state, plan, tools, outcome) from
   public datasets (ACE, MSC).
2. **Retrieve** relevant past episodes using four strategies (BM25, dense,
   hybrid, field-aware), with optional success/failure filtering and
   heuristic re-ranking.
3. **Evaluate** retrieval with IR metrics (P@k, R@k, MRR, nDCG) and with
   downstream metrics on a simulated agent (success rate, failure
   repetition, tool-distribution shift).

---

## 1. Pipeline at a glance

```
                                 ┌──────────────────────┐
   HF datasets (ACE, MSC)  ───▶  │ loaders/ace.py        │
                                 │ loaders/msc.py        │   (synthetic
                                 └──────────┬───────────┘    fallback)
                                            ▼
                                 ┌──────────────────────┐
                                 │ Episode dataclass     │   schema.py
                                 │  • state_text         │
                                 │  • plan_text          │
                                 │  • tools_used         │
                                 │  • outcome_label      │
                                 │  • outcome_text       │
                                 │  • full_document      │
                                 └──────────┬───────────┘
                                            ▼
                              data/processed/episodes.jsonl
                                            │
                                            ▼
                                ┌─────────────────────────┐
                                │ build_indices.py         │
                                │   ┌──────────────────┐   │
                                │   │ BM25Index (pkl)  │   │
                                │   │ DenseIndex state │   │
                                │   │ DenseIndex s+p   │   │
                                │   │ DenseIndex plan  │   │
                                │   │ MetadataStore    │   │
                                │   └──────────────────┘   │
                                └────────────┬────────────┘
                                             ▼
                              ┌──────────────────────────────┐
                              │ Retrievers (common interface) │
                              │   BM25Retriever               │
                              │   DenseRetriever              │
                              │   HybridRetriever             │
                              │   FieldAwareRetriever         │
                              │   + filter_outcome=success/   │
                              │     failure                   │
                              └──────────────┬───────────────┘
                                             │
                          ┌──────────────────┴──────────────────┐
                          ▼                                     ▼
              ┌──────────────────────┐              ┌──────────────────────┐
              │ rerank.heuristics     │              │ Held-out queries +   │
              │ (recency, failure,    │              │ qrels (relevance.py) │
              │ tool-match boost)     │              └──────────┬───────────┘
              └──────────┬───────────┘                         ▼
                         │                          ┌──────────────────────┐
                         ▼                          │ ir_metrics.evaluate  │
              ┌──────────────────────┐              │ P@k R@k MRR nDCG     │
              │ SimulatedAgent       │              └──────────────────────┘
              │  + tool_bias         │
              │  + prompt_aug        │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │ downstream.aggregate │
              │  success rate        │
              │  failure repetition  │
              │  tool dist KL        │
              └──────────────────────┘

         5 controlled experiments → results/tables/*.csv + results/figures/*.png
```

### The five experiments

| # | Question | What varies | Output |
|---|----------|-------------|--------|
| 1 | Which retrieval method is best? | retriever | `exp1_methods.csv` |
| 2 | Which document representation is best? | embedded text (state / state+plan / full) | `exp2_representation.csv` |
| 3 | Should we retrieve successes, failures, or both? | filter mode | `exp3_*.csv` (IR + downstream) |
| 4 | How sensitive is retrieval to k? | k ∈ {1,3,5,10} | `exp4_topk.csv` |
| 5 | What weights work best? | hybrid α; field-aware weights | `exp5_alpha.csv`, `exp5_field_weights.csv` |

---

## 2. How to run end-to-end

```bash
# 1. install
pip install -e ".[dev]"

# 2. build corpus (HF download with synthetic fallback)
python scripts/download_datasets.py
#   -> data/processed/episodes.jsonl

# 3. build indices
python scripts/build_indices.py
#   -> data/index/{bm25.pkl, dense_state/, dense_state_plan/, dense_plan/}

# 4. tests
pytest -q
#   -> 15 passed

# 5. quick smoke run of experiment 1
python experiments/exp1_methods.py --quick

# 6. full run
python experiments/run_all.py
#   -> results/tables/*.csv, results/figures/*.png
```

`--quick` flags exist on every experiment script and clip queries/tasks to
~30 so a full sweep finishes in seconds for sanity-checking.

---

## 3. Directory tree (annotated)

```
ir-assignment-2/
├── pyproject.toml             # package metadata + dependencies
├── .gitignore                 # ignores data/, indices, generated artifacts
├── README.md                  # short user-facing intro
├── HANDOVER.md                # THIS FILE
│
├── src/episodic/              # library code (importable as `episodic`)
│   ├── __init__.py            # version string only
│   ├── schema.py              # Episode dataclass + JSONL I/O  ← single source of truth
│   │
│   ├── loaders/               # external data → Episode[]
│   │   ├── __init__.py
│   │   ├── ace.py             # ACE benchmark (HF + synthetic fallback)
│   │   └── msc.py             # Multi-Session Chat (HF + synthetic fallback)
│   │
│   ├── indexing/              # storage / index layer
│   │   ├── __init__.py
│   │   ├── tokenize.py        # shared tokenizer (regex + NLTK stopwords)
│   │   ├── bm25_index.py      # rank_bm25 wrapper, pickled to disk
│   │   ├── dense_index.py     # FAISS IndexFlatIP over MiniLM embeddings
│   │   └── metadata_store.py  # episode_id → Episode, plus outcome filter
│   │
│   ├── retrieval/             # 4 retrieval strategies + Retriever protocol
│   │   ├── __init__.py        # lazy re-exports (avoids eager rank_bm25/faiss imports)
│   │   ├── base.py            # Retriever Protocol, RetrievalResult, outcome filter
│   │   ├── bm25.py            # BM25Retriever
│   │   ├── dense.py           # DenseRetriever (state OR state+plan)
│   │   ├── hybrid.py          # alpha * BM25_norm + (1-alpha) * cos_norm
│   │   └── field_aware.py     # weighted state+plan+tools+outcome scoring
│   │
│   ├── rerank/                # composable re-ranking heuristics
│   │   ├── __init__.py
│   │   ├── heuristics.py      # recency, failure_penalty, tool_match_boost, RerankConfig
│   │   └── ltr.py             # stub for future learning-to-rank trainer
│   │
│   ├── agent/                 # downstream measurement instrument
│   │   ├── __init__.py
│   │   ├── simulated_agent.py # rule-based agent that consumes retrieved memories
│   │   ├── tool_bias.py       # adjusts tool prior using success/failure retrievals
│   │   └── prompt_aug.py      # formats memories into a "what worked / avoid" string
│   │
│   └── eval/
│       ├── __init__.py        # re-exports IR metrics
│       ├── ir_metrics.py      # P@k, R@k, MRR, nDCG (from scratch), evaluate()
│       ├── relevance.py       # split, build held-out queries, build qrels (graded)
│       └── downstream.py      # success rate, failure repetition, tool-dist KL
│
├── experiments/               # one file per study, plus shared scaffolding
│   ├── __init__.py            # marker
│   ├── _common.py             # load_setup() + table/plot helpers (path-bootstraps src/)
│   ├── exp1_methods.py        # BM25 vs dense vs dense+plan vs hybrid vs field-aware
│   ├── exp2_representation.py # state / state+plan / full_document on dense
│   ├── exp3_success_failure.py# success-only / failure-only / both (IR + downstream)
│   ├── exp4_topk.py           # k sensitivity
│   ├── exp5_weights.py        # alpha sweep + field-weight grid
│   └── run_all.py             # invokes all five in order
│
├── scripts/                   # entry points for data + index build
│   ├── download_datasets.py   # HF (or synthetic) → data/processed/episodes.jsonl
│   └── build_indices.py       # BM25 + 3 FAISS indices → data/index/
│
├── tests/                     # pytest unit tests (15 cases, all green)
│   ├── conftest.py            # adds src/ to sys.path for tests
│   ├── test_schema.py         # Episode round-trip + full_document construction
│   ├── test_metrics.py        # P@k, R@k, MRR, nDCG correctness
│   ├── test_bm25.py           # BM25 ranks term-overlapping doc first; outcome filter works
│   ├── test_dense.py          # dense retrieval with a deterministic stub encoder
│   ├── test_hybrid.py         # hybrid extremes (α=0, α=1) match component retrievers
│   └── test_agent.py          # tool_bias + simulated agent smoke
│
├── results/                   # generated artifacts (kept empty in git)
│   ├── .gitkeep
│   ├── figures/.gitkeep
│   └── tables/.gitkeep
│
└── data/                      # NOT in git — created at runtime
    ├── processed/episodes.jsonl
    └── index/{bm25.pkl, dense_state/, dense_state_plan/, dense_plan/}
```

---

## 4. File-by-file reference

### Top level

| File | Purpose |
|------|---------|
| **`pyproject.toml`** | setuptools build config. Pins minimum versions of `rank-bm25`, `sentence-transformers`, `faiss-cpu`, `datasets`, numpy/pandas/matplotlib/nltk. `[project.optional-dependencies].dev` adds `pytest`. |
| **`.gitignore`** | Excludes `data/`, generated indices (`*.faiss`, `*.npz`, `*.pkl`), `results/{figures,tables}/*` (but keeps `.gitkeep`), and the usual Python cruft. |
| **`README.md`** | 30-line user-facing intro: stack, layout, quick-start. |
| **`HANDOVER.md`** | This file. |

### `src/episodic/`

| File | What it does | Why it exists |
|------|-------------|---------------|
| **`__init__.py`** | Sets `__version__`. | Makes the directory an importable package. |
| **`schema.py`** | `Episode` dataclass (state, plan, tools, outcome, timestamp, source, task_type) and `full_document` derived field (concatenation used by BM25). `write_jsonl` / `read_jsonl` / `load_episodes` for disk I/O. | Single source of truth — every other module operates on `Episode` objects. Adding a new field here propagates everywhere. |

### `src/episodic/loaders/`

| File | What it does |
|------|-------------|
| **`__init__.py`** | Empty, package marker. |
| **`ace.py`** | Tries 3 candidate HuggingFace dataset paths for ACE; adapts whatever schema it finds via best-effort field aliases (`state` / `task` / `instruction`, `plan` / `trajectory`, `tools` / `actions`, `outcome` / `result` / `label`). If nothing loads, returns a deterministic 800-episode synthetic set whose tool/task affinity is wired to make retrieval evaluation meaningful. |
| **`msc.py`** | Same pattern for MSC: attempts `facebook/msc`, `msc`, `ParlAI/msc`, projects each session into an `Episode` (state = penultimate utterance, plan = last utterance, success = grounded persona-using response). Synthetic fallback of 400 episodes. |

> **Why a synthetic fallback?** ACE/MSC HuggingFace paths drift over releases.
> Without a fallback the entire downstream pipeline would block on dataset
> availability. The fallback is seeded so experiments are reproducible.

### `src/episodic/indexing/`

| File | What it does |
|------|-------------|
| **`__init__.py`** | Package marker. |
| **`tokenize.py`** | `tokenize(text)` returns lowercased word tokens minus stopwords. NLTK stopwords with built-in fallback so first-run failures (no NLTK download) still work. Cached via `lru_cache`. |
| **`bm25_index.py`** | `BM25Index.build(episodes, field=...)` builds an in-memory `BM25Okapi` over tokenized documents and remembers the parallel list of episode IDs. `search(query, k)` returns `(episode_id, score)` pairs. `save/load` use pickle. |
| **`dense_index.py`** | `Encoder` lazily wraps `SentenceTransformer('all-MiniLM-L6-v2')`. `DenseIndex.build(episodes, field)` encodes the chosen field (`"state"` / `"state_plan"` / `"full_document"`), L2-normalizes, and stuffs into `faiss.IndexFlatIP` (inner product over normalized vectors == cosine). `save/load` round-trip both the FAISS index and the `(ids, embeddings, field, model)` tuple via `numpy.savez`. |
| **`metadata_store.py`** | Tiny `dict[str, Episode]` wrapper. Provides `get(id)`, `all_episodes()`, `filter_outcome(label)`, JSON I/O. Used by every retriever to turn an `episode_id` back into a full `Episode`. |

### `src/episodic/retrieval/`

| File | What it does |
|------|-------------|
| **`__init__.py`** | Re-exports the protocol types eagerly and the concrete retrievers via `__getattr__` lazy-import — that way `import episodic.eval.ir_metrics` doesn't pull `rank_bm25` / `faiss` if you only need metrics. |
| **`base.py`** | `Retriever` `Protocol` with `retrieve(query, k, filter_outcome) -> list[RetrievalResult]`. `apply_outcome_filter` is shared by every retriever to implement the success-only / failure-only pipelines (oversample → filter → truncate to `k`). |
| **`bm25.py`** | Thin wrapper over `BM25Index.search` + outcome filtering. |
| **`dense.py`** | Thin wrapper over `DenseIndex.search`; the same class serves both `state` and `state+plan` variants by holding a different `DenseIndex`. |
| **`hybrid.py`** | Runs BM25 and dense over `k * recall_multiplier` candidates each, **min-max normalizes both score lists to [0,1]** so they're comparable, combines as `alpha * bm25_norm + (1-alpha) * cos_norm` over the union, sorts, applies outcome filter. |
| **`field_aware.py`** | Recall pool from BM25 ∪ dense-state. Re-scores each candidate with a weighted sum of: cos(query, episode.state), cos(query, episode.plan), Jaccard tool overlap (query tools extracted by keyword match against the known vocab), and outcome match. Weights are a `FieldWeights` dataclass — defaults `(0.5, 0.2, 0.2, 0.1)`. |

### `src/episodic/rerank/`

| File | What it does |
|------|-------------|
| **`__init__.py`** | Re-exports public API. |
| **`heuristics.py`** | Pure functions: `recency_boost` (exponential decay over age), `failure_penalty` (multiplicative penalty when the episode's outcome contradicts the requested mode), `tool_match_boost` (boosts when query tools intersect episode tools). `RerankConfig` toggles them on/off; `apply_rerank` composes them and re-sorts. |
| **`ltr.py`** | Documented stub: `LTRFeatures` dataclass naming the features a future learning-to-rank reranker would consume. Out of scope for v1 by design — the rest of the system is structured so an LTR reranker plugs in cleanly. |

### `src/episodic/agent/`

| File | What it does |
|------|-------------|
| **`__init__.py`** | Re-exports `SimulatedAgent`, `AgentRollout`, `build_memory_context`, `biased_tool_distribution`. |
| **`simulated_agent.py`** | Deterministic rule-based agent. Per task: optionally retrieves k success episodes and k failure episodes, builds a biased tool distribution, samples `n_tools_per_action` tools without replacement, decides success by intersection with ground-truth tools (with bounded noise), and reports whether it used a known-failure tool set. Returns `AgentRollout` per task. |
| **`tool_bias.py`** | `biased_tool_distribution(vocab, successes, failures, beta_success, beta_failure)`: starts with uniform prior over the tool vocabulary, multiplies by `(1 + beta_success)` per occurrence in successes, by `max(1 - beta_failure, eps)` per occurrence in failures, renormalizes. |
| **`prompt_aug.py`** | `build_memory_context(successes, failures)` formats retrieved episodes into a textual "WHAT HAS WORKED / WHAT TO AVOID" block. Not used by the deterministic agent today (it consumes the tool distribution directly), but ready for a future LLM-based agent. |

### `src/episodic/eval/`

| File | What it does |
|------|-------------|
| **`__init__.py`** | Re-exports the IR metric API. |
| **`ir_metrics.py`** | From-scratch implementations of P@k, R@k, MRR, nDCG@k. `evaluate(results_by_query, qrels, k_values)` aggregates them across a query batch and returns a flat dict (`P@1`, `R@5`, `nDCG@10`, …, plus `MRR`). |
| **`relevance.py`** | `split_episodes` (seeded train/held-out split), `HeldOutQuery` dataclass, `build_queries` (one query per held-out episode), `build_qrels` (graded relevance: 2 if same task_type AND outcome matches the requested mode, 1 if only task_type matches, 0 otherwise). |
| **`downstream.py`** | `aggregate(rollouts) -> DownstreamMetrics`: success rate, failure repetition rate (% of agent failures that re-used a tool set known to have failed before), and the tool-usage distribution. `kl_divergence(p, q)` with epsilon smoothing for tool-distribution-shift comparisons. |

### `experiments/`

All five experiment scripts share `_common.py` and follow the same shape:
load setup → build retrievers → call `evaluate_retriever` → write CSV + bar plot.

| File | What it does |
|------|-------------|
| **`__init__.py`** | Marker; lets `runpy.run_path` resolve module-relative imports cleanly. |
| **`_common.py`** | The boring plumbing. Bootstraps `src/` onto `sys.path` so scripts run without `pip install -e .`. `load_setup()` returns a `Setup` dataclass holding the train/held-out split, queries, qrels, metadata store, BM25 index, three dense indices, encoder, and tool vocab. **Always restricts indices to train at eval time** — when saved indices exist, rebuilds BM25 over train tokens and subsets saved FAISS embeddings by train IDs (no re-encoding). When indices are missing, builds everything in-memory from scratch. Helpers: `evaluate_retriever`, `write_table`, `save_bar_plot` (matplotlib Agg backend → PNG). |
| **`exp1_methods.py`** | Compares BM25, dense (state), dense (state+plan), hybrid, field-aware. Writes `results/tables/exp1_methods.csv` and `results/figures/exp1_methods_ndcg5.png`. |
| **`exp2_representation.py`** | Holds retriever fixed (dense), varies the embedded text (state / state+plan / full_document — the last one is built ad-hoc since `state` and `state+plan` are pre-loaded). |
| **`exp3_success_failure.py`** | Two halves. **IR side:** evaluates hybrid retrieval under three modes (`success`, `failure`, `any`) using mode-aware qrels. **Downstream side:** runs the simulated agent four ways (no retrieval / success-only / failure-only / both) and reports success rate, failure repetition, and tool-distribution KL vs. the no-retrieval baseline. |
| **`exp4_topk.py`** | Sweeps `k ∈ {1, 3, 5, 10}` using the hybrid retriever. Plots nDCG@k curve. |
| **`exp5_weights.py`** | Sweeps hybrid `alpha ∈ {0.3, 0.5, 0.7}` and a small field-weight grid (`FieldWeights(0.5,0.2,0.2,0.1)` etc). Two CSVs, two plots. |
| **`run_all.py`** | Invokes the five scripts in order via `runpy.run_path`, propagating `--quick`. |

### `scripts/`

| File | What it does |
|------|-------------|
| **`download_datasets.py`** | Calls `loaders.ace.load()` and `loaders.msc.load()` (HF first, synthetic fallback), concatenates, writes `data/processed/episodes.jsonl`. CLI flags: `--max-ace`, `--max-msc`, `--no-synthetic`. |
| **`build_indices.py`** | Loads `episodes.jsonl`, builds `BM25Index` over `full_document`, three `DenseIndex` variants (`state`, `state+plan`, `plan`), saves all to `data/index/`. The `dense_plan` index is built ad-hoc here (encoder applied directly to `plan_text`) for the field-aware retriever. |

### `tests/`

| File | Coverage |
|------|---------|
| **`conftest.py`** | Adds `src/` to `sys.path` so tests run without `pip install -e .`. |
| **`test_schema.py`** | `full_document` is auto-built and contains all four sections; JSONL round-trip preserves fields and outcome labels. |
| **`test_metrics.py`** | P@k boundary cases, R@k with empty relevance, MRR (top-rank, no-hit, first-rank), nDCG perfect-ordering = 1.0, nDCG empty cases. |
| **`test_bm25.py`** | BM25 ranks the term-overlapping document first; the outcome filter on `BM25Retriever` returns only documents matching the requested label. |
| **`test_dense.py`** | Dense retriever with a deterministic 1-hot stub encoder (no network, no model download) — covers token-match retrieval and outcome filtering with a 10× oversample. Skips cleanly if `faiss` is missing. |
| **`test_hybrid.py`** | Hybrid finds term-overlapping doc; verifies `α=1.0` ≡ BM25-only and `α=0.0` ≡ dense-only on the same query. |
| **`test_agent.py`** | Tool-bias amplifies success-tool probability (and renormalizes); `SimulatedAgent` runs without a retriever and picks tools from the vocabulary. |

`pytest -q` → **15 passed**.

### `results/`

Generated artifacts. Empty in git (`.gitkeep` placeholders). After
`run_all.py`:

```
results/
├── tables/
│   ├── exp1_methods.csv
│   ├── exp2_representation.csv
│   ├── exp3_success_failure_ir.csv
│   ├── exp3_success_failure_downstream.csv
│   ├── exp4_topk.csv
│   ├── exp5_alpha.csv
│   └── exp5_field_weights.csv
└── figures/
    ├── exp1_methods_ndcg5.png
    ├── exp2_representation_ndcg5.png
    ├── exp3_ir_ndcg5.png
    ├── exp3_ds_success.png
    ├── exp4_topk_ndcg.png
    ├── exp5_alpha_ndcg5.png
    └── exp5_fields_ndcg5.png
```

### `data/`

Not tracked in git. Materialized at runtime:

```
data/
├── processed/episodes.jsonl                # output of download_datasets.py
└── index/                                   # output of build_indices.py
    ├── bm25.pkl
    ├── dense_state/{index.faiss, vectors.npz}
    ├── dense_state_plan/{index.faiss, vectors.npz}
    └── dense_plan/{index.faiss, vectors.npz}
```

---

## 5. Where to plug things in

| If you want to… | Edit this | Notes |
|-----------------|-----------|-------|
| Add a new retrieval strategy | New file in `src/episodic/retrieval/`, implement `Retriever` protocol from `base.py`. | Add a lazy import in `retrieval/__init__.py` and a row in `experiments/exp1_methods.py`. |
| Try a different embedding model | Pass `Encoder(model_name=...)` into `DenseIndex.build` (or via `--model` flag in `build_indices.py`). | The model name is persisted into the saved npz so `DenseIndex.load` recreates the right encoder. |
| Add a re-ranker | New function in `rerank/heuristics.py`, add a flag to `RerankConfig`, wire into `apply_rerank`. | Or implement LTR by filling out `rerank/ltr.py`. |
| Change the relevance rule | `eval/relevance.py::build_qrels`. | Be careful — every IR metric depends on this; rerun all experiments. |
| Use a real LLM agent | New file in `src/episodic/agent/`. Reuse `prompt_aug.build_memory_context` for the prompt. | The current `SimulatedAgent` only consumes the biased tool distribution, but the prompt builder is ready. |
| Add a new dataset | New module under `src/episodic/loaders/` returning `list[Episode]`. Wire into `scripts/download_datasets.py`. | Be honest about the outcome-label derivation — document the rule in the loader. |

---

## 6. Gotchas / things to know before you change anything

1. **Held-out evaluation requires train-only indices.** This was a real
   bug in v0 — `scripts/build_indices.py` writes indices over the **full
   corpus**, but the retrieval-side metadata store is built from the
   **train split only** (correct semantics: a query from the held-out
   set should not be able to retrieve itself or other held-out
   episodes). Loading the saved indices verbatim caused
   `MetadataStore.get(eid) -> KeyError` when BM25 surfaced a held-out
   ID.

   **How `_common.load_setup()` handles it now:** if saved indices are
   present, it rebuilds BM25 from the train corpus (cheap — pure Python
   tokenization) and **subsets the saved dense embeddings by train IDs
   to build fresh FAISS indices** — no re-encoding required. The end
   result: every index used during evaluation contains exactly the
   train episodes, never held-out ones. See
   [`_common._train_dense_from_saved`](experiments/_common.py).

   **If you change the split, the index build, or the metadata store,
   re-audit this path carefully.** Symptoms of regression: `KeyError`
   on `store.get(eid)`, or impossibly perfect IR scores (the retriever
   pulling the query out of its own corpus).

2. **Lazy imports in `retrieval/__init__.py`.** This is intentional. If you
   import retrievers directly there, modules with no IR-runtime needs
   (e.g. `episodic.eval.ir_metrics`) start pulling `rank_bm25` and
   `faiss`, which breaks anyone running just the metrics layer.

3. **NLTK stopwords download.** `tokenize.stopwords()` tries NLTK and falls
   back to a built-in list. First run on a fresh machine may fetch
   stopwords once; offline runs use the fallback automatically. The two
   token sets are similar but not identical — don't be surprised by
   tiny BM25 score deltas across machines.

4. **Synthetic fallback semantics.** If ACE/MSC HuggingFace downloads
   fail, the loaders emit synthetic data with deterministic seeds. This
   is intentional — the pipeline must produce metric numbers. If you
   need the real datasets and don't see them, check connectivity and
   look at the candidate paths in `loaders/ace.py::ACE_HF_CANDIDATES` /
   `loaders/msc.py::MSC_HF_CANDIDATES` — schemas drift; you may need
   to add another adapter.

5. **FAISS = inner product over normalized vectors.** Both
   `dense_index.py` and `field_aware.py` rely on this. If you swap in a
   non-normalized index (e.g. `IndexFlatL2`), the hybrid and field-aware
   score combinations will break silently because they assume cosine.

6. **Hybrid score normalization is per-query min-max.** This is robust
   but loses absolute calibration — don't compare raw hybrid scores
   across queries. For per-query ranking it's fine.

7. **`SimulatedAgent` is a measurement instrument, not the contribution.**
   Its success rule (overlap with ground-truth tools + bounded noise) is
   intentionally simple — the point is that better retrieval should
   produce a more useful tool distribution. Don't tune the agent to
   make numbers look better; tune retrieval.

8. **`pyproject.toml` packages-find points at `src/`.** If you move
   `src/episodic` you must update both `[tool.setuptools.packages.find]`
   and the `_PROJECT_ROOT/_SRC` bootstraps in `experiments/_common.py`,
   `scripts/*.py`, and `tests/conftest.py`.

9. **Results overwrite on every run.** `experiments/*` write into fixed
   paths under `results/`. There is no run-id namespacing. If you want
   to compare two configurations, copy the tables before re-running.

---

## 7. Suggested next steps (v2 ideas)

- Wire a real LLM agent (Ollama / Anthropic SDK / OpenAI) using
  `prompt_aug.build_memory_context`. The retriever / metrics layer
  doesn't need to change.
- Implement `rerank/ltr.py` — the feature shape is documented; train on
  qrels-derived labels.
- Add a cross-encoder reranker (e.g. `ms-marco-MiniLM-L-6-v2`) as a
  third stage after hybrid recall.
- Stream memory updates: let new agent rollouts append to the index
  online so the retrieval distribution evolves during the rollout
  (currently the corpus is static).
- Run-id namespacing in `results/` so multiple experiment configurations
  can coexist.
- Replace `pickle` for BM25 with a JSON-serializable form to harden the
  on-disk format.

---

## 8. One-paragraph TL;DR

`Episode` objects flow from `loaders/` into JSONL, get indexed three ways
(BM25, dense, plan-dense), and are queried by four retrievers behind a
shared `Retriever` protocol. A held-out split + graded qrels in
`eval/relevance.py` feeds `eval/ir_metrics.py` for IR scores. A
deterministic `SimulatedAgent` consumes retrieved successes/failures
through `agent/tool_bias.py` and `eval/downstream.py` reports
success-rate / failure-repetition / tool-KL. Five experiment scripts
explore method, representation, success/failure, k, and weight choices,
emitting CSVs and PNGs under `results/`. Tests cover schema, metrics,
each retriever, and the agent — all green.
