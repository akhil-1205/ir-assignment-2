"""Knowledge-graph index over episodes.

Graph schema (heterogeneous, undirected):

    episode:<id>  --uses-->        tool:<name>     w=1.0
    episode:<id>  --is_a-->        tasktype:<t>    w=1.0
    episode:<id>  --ended-->       outcome:<l>     w=0.5
    episode:<id>  --mentions-->    entity:<tok>    w=idf(tok)

Entity vocab is built from non-stopword content tokens of length >= 4
that appear in at least `min_doc_freq` episodes. Each entity gets an
idf weight so rare-but-shared tokens are stronger evidence than common
ones. The whole graph is stored as a row-normalized sparse adjacency
matrix; retrieval is **Personalized PageRank with restart** seeded by
query entities and tool-vocab keyword hits. This captures multi-hop
relationships (query → entity → episode_A → tool_X → episode_B) that
straight lexical/dense matching can't.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..schema import Episode
from .tokenize import tokenize


def _row_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    row_sums = np.asarray(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sums)
    return (D_inv @ A).tocsr()


@dataclass
class KGIndex:
    A_norm: sp.csr_matrix          # row-normalized adjacency (transition matrix)
    node_to_idx: dict[str, int]
    idx_to_node: list[str]
    episode_ids: list[str]         # ordered, matches episode_node_idx
    episode_node_idx: np.ndarray   # int64 indices into A for episode nodes
    entity_idf: dict[str, float]
    tool_vocab: list[str]

    # ---------- build ----------

    @classmethod
    def build(
        cls,
        episodes: list[Episode],
        min_doc_freq: int = 2,
        min_token_len: int = 4,
        outcome_edge_weight: float = 0.5,
    ) -> "KGIndex":
        # 1. content-token document frequency over state + plan
        ep_tokens: dict[str, set[str]] = {}
        token_df: Counter[str] = Counter()
        for ep in episodes:
            text = f"{ep.state_text} {ep.plan_text}"
            toks = {t for t in tokenize(text) if len(t) >= min_token_len}
            ep_tokens[ep.episode_id] = toks
            for t in toks:
                token_df[t] += 1

        N = max(len(episodes), 1)
        entity_idf: dict[str, float] = {
            tok: math.log((1 + N) / (1 + df)) + 1.0
            for tok, df in token_df.items()
            if df >= min_doc_freq
        }

        # 2. build node index. Episode nodes go first so episode_node_idx is contiguous.
        node_to_idx: dict[str, int] = {}
        idx_to_node: list[str] = []

        def add(name: str) -> int:
            i = node_to_idx.get(name)
            if i is not None:
                return i
            node_to_idx[name] = len(idx_to_node)
            idx_to_node.append(name)
            return node_to_idx[name]

        episode_ids: list[str] = []
        ep_indices: list[int] = []
        for ep in episodes:
            episode_ids.append(ep.episode_id)
            ep_indices.append(add(f"ep:{ep.episode_id}"))

        tool_vocab = sorted({t for ep in episodes for t in ep.tools_used})
        for t in tool_vocab:
            add(f"tool:{t}")
        for tt in sorted({ep.task_type for ep in episodes}):
            add(f"tasktype:{tt}")
        add("outcome:success")
        add("outcome:failure")
        for tok in sorted(entity_idf.keys()):
            add(f"entity:{tok}")

        n_nodes = len(idx_to_node)

        # 3. edges (symmetric)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        def edge(u: int, v: int, w: float) -> None:
            rows.append(u); cols.append(v); data.append(w)
            rows.append(v); cols.append(u); data.append(w)

        for ep in episodes:
            ei = node_to_idx[f"ep:{ep.episode_id}"]
            for t in ep.tools_used:
                edge(ei, node_to_idx[f"tool:{t}"], 1.0)
            edge(ei, node_to_idx[f"tasktype:{ep.task_type}"], 1.0)
            edge(ei, node_to_idx[f"outcome:{ep.outcome_label}"], outcome_edge_weight)
            for tok in ep_tokens[ep.episode_id]:
                if tok in entity_idf:
                    edge(ei, node_to_idx[f"entity:{tok}"], entity_idf[tok])

        A = sp.csr_matrix(
            (np.asarray(data, dtype=np.float64),
             (np.asarray(rows, dtype=np.int64),
              np.asarray(cols, dtype=np.int64))),
            shape=(n_nodes, n_nodes),
        )

        return cls(
            A_norm=_row_normalize(A),
            node_to_idx=node_to_idx,
            idx_to_node=idx_to_node,
            episode_ids=episode_ids,
            episode_node_idx=np.asarray(ep_indices, dtype=np.int64),
            entity_idf=entity_idf,
            tool_vocab=tool_vocab,
        )

    # ---------- search ----------

    def _personalization(self, query: str) -> np.ndarray:
        n_nodes = self.A_norm.shape[0]
        p = np.zeros(n_nodes, dtype=np.float64)

        q_tokens = {t for t in tokenize(query) if len(t) >= 4}
        for tok in q_tokens:
            if tok in self.entity_idf:
                idx = self.node_to_idx.get(f"entity:{tok}")
                if idx is not None:
                    p[idx] += self.entity_idf[tok]

        q_lower = (query or "").lower()
        for t in self.tool_vocab:
            if t.lower() in q_lower:
                idx = self.node_to_idx.get(f"tool:{t}")
                if idx is not None:
                    p[idx] += 1.0

        s = p.sum()
        if s > 0:
            p /= s
        return p

    def ppr_vector(
        self,
        query: str,
        alpha: float = 0.85,
        n_iter: int = 30,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Run PPR with restart and return the full mass vector over all nodes.

        Returns a length-N zero vector if no query token lands on a seed node.
        Callers can index into this vector by `node_to_idx[name]`.
        """
        p = self._personalization(query)
        n = self.A_norm.shape[0]
        if p.sum() == 0:
            return np.zeros(n, dtype=np.float64)
        AT = self.A_norm.T.tocsr()
        r = p.copy()
        for _ in range(n_iter):
            r_new = (1.0 - alpha) * p + alpha * (AT @ r)
            if np.abs(r_new - r).sum() < tol:
                r = r_new
                break
            r = r_new
        return r

    def ppr_mass_for_episodes(
        self,
        query: str,
        episode_ids: list[str],
        alpha: float = 0.85,
        n_iter: int = 30,
    ) -> list[float]:
        """Return PPR mass for the given episode IDs in input order. Missing
        IDs (e.g. held-out episodes filtered out of the index) return 0.0.
        """
        r = self.ppr_vector(query, alpha=alpha, n_iter=n_iter)
        masses: list[float] = []
        for eid in episode_ids:
            idx = self.node_to_idx.get(f"ep:{eid}")
            masses.append(float(r[idx]) if idx is not None else 0.0)
        return masses

    def search(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.85,
        n_iter: int = 30,
        tol: float = 1e-6,
    ) -> list[tuple[str, float]]:
        r = self.ppr_vector(query, alpha=alpha, n_iter=n_iter, tol=tol)
        if r.sum() == 0:
            return []
        ep_scores = r[self.episode_node_idx]
        n = min(k, len(ep_scores))
        if n <= 0:
            return []
        top = np.argpartition(-ep_scores, n - 1)[:n]
        top = top[np.argsort(-ep_scores[top])]
        return [(self.episode_ids[i], float(ep_scores[i])) for i in top]

    # ---------- I/O ----------

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(
                {
                    "A_norm": self.A_norm,
                    "node_to_idx": self.node_to_idx,
                    "idx_to_node": self.idx_to_node,
                    "episode_ids": self.episode_ids,
                    "episode_node_idx": self.episode_node_idx,
                    "entity_idf": self.entity_idf,
                    "tool_vocab": self.tool_vocab,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "KGIndex":
        with Path(path).open("rb") as f:
            d = pickle.load(f)
        return cls(**d)

    def restrict_to_episodes(self, train_id_set: set[str]) -> "KGIndex":
        """Rebuild the index to contain only episode nodes whose ID is in
        `train_id_set`. Used at evaluation time to avoid held-out leakage.

        Drops the held-out episode nodes (and any entity/tool/etc. nodes
        that become orphaned). Re-row-normalizes.
        """
        keep_ep_ids = [eid for eid in self.episode_ids if eid in train_id_set]
        keep_ep_node_idx = {self.node_to_idx[f"ep:{eid}"] for eid in keep_ep_ids}

        # Identify nodes to keep: all non-episode nodes + train episode nodes
        n = self.A_norm.shape[0]
        keep_mask = np.ones(n, dtype=bool)
        for eid in self.episode_ids:
            if eid not in train_id_set:
                keep_mask[self.node_to_idx[f"ep:{eid}"]] = False

        old_idx = np.where(keep_mask)[0]
        new_idx_for_old: dict[int, int] = {old: new for new, old in enumerate(old_idx)}

        A_full = self.A_norm.tocoo(copy=False)
        # We row-normalized once; to restrict cleanly we need raw counts again.
        # Here we accept a slight bias: take submatrix and re-normalize.
        sub = self.A_norm.tocsr()[old_idx, :][:, old_idx]
        sub = _row_normalize(sub)

        new_idx_to_node = [self.idx_to_node[i] for i in old_idx]
        new_node_to_idx = {name: i for i, name in enumerate(new_idx_to_node)}
        new_ep_node_idx = np.array(
            [new_node_to_idx[f"ep:{eid}"] for eid in keep_ep_ids], dtype=np.int64
        )

        return KGIndex(
            A_norm=sub,
            node_to_idx=new_node_to_idx,
            idx_to_node=new_idx_to_node,
            episode_ids=keep_ep_ids,
            episode_node_idx=new_ep_node_idx,
            entity_idf=self.entity_idf,
            tool_vocab=self.tool_vocab,
        )
