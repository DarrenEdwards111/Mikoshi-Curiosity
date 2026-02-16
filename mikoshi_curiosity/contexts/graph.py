"""Network/graph exploration context."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mikoshi_curiosity.space import State, StateSpace


class GraphSpace(StateSpace):
    """Explore a graph structure.

    Parameters
    ----------
    nodes : list
        Node identifiers (strings or dicts with 'id' and optional 'features').
    edges : list
        Pairs (source, target) or dicts with 'source'/'target'.
    embedding_dim : int
        Dimension of random-walk based embeddings.
    """

    def __init__(
        self,
        nodes: List[Any],
        edges: List[Any],
        embedding_dim: int = 64,
    ):
        self._adj: Dict[str, List[str]] = defaultdict(list)
        self._node_features: Dict[str, dict] = {}

        for n in nodes:
            if isinstance(n, dict):
                nid = str(n["id"])
                self._node_features[nid] = {k: v for k, v in n.items() if k != "id"}
            else:
                nid = str(n)
                self._node_features[nid] = {}
            if nid not in self._adj:
                self._adj[nid] = []

        for e in edges:
            if isinstance(e, dict):
                src, tgt = str(e["source"]), str(e["target"])
            elif isinstance(e, (list, tuple)):
                src, tgt = str(e[0]), str(e[1])
            else:
                raise ValueError(f"Edge format not recognised: {e}")
            self._adj[src].append(tgt)
            self._adj[tgt].append(src)

        self._node_ids = list(self._adj.keys())
        self._id_to_idx = {nid: i for i, nid in enumerate(self._node_ids)}
        self._embedding_dim = embedding_dim
        self._embeddings = self._compute_embeddings()

    def _compute_embeddings(self) -> np.ndarray:
        """Simple spectral-ish embedding via random walk features."""
        n = len(self._node_ids)
        if n == 0:
            return np.zeros((0, self._embedding_dim))

        rng = np.random.default_rng(42)
        emb = rng.normal(0, 1, (n, self._embedding_dim))

        # A few rounds of neighbour averaging (like simplified GNN)
        for _ in range(5):
            new_emb = np.zeros_like(emb)
            for i, nid in enumerate(self._node_ids):
                neighbors = self._adj[nid]
                if neighbors:
                    n_indices = [self._id_to_idx[nb] for nb in neighbors if nb in self._id_to_idx]
                    if n_indices:
                        new_emb[i] = 0.5 * emb[i] + 0.5 * np.mean(emb[n_indices], axis=0)
                    else:
                        new_emb[i] = emb[i]
                else:
                    new_emb[i] = emb[i]
            emb = new_emb

        # Normalise
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    def _make_state(self, nid: str) -> State:
        idx = self._id_to_idx[nid]
        degree = len(self._adj[nid])
        features = dict(self._node_features.get(nid, {}))
        features["degree"] = degree
        return State(id=nid, features=features, embedding=self._embeddings[idx])

    def get_neighbors(self, state: State, n: int = 10) -> List[State]:
        adj = self._adj.get(state.id, [])
        # Direct graph neighbours
        result = [self._make_state(nid) for nid in adj[:n]]
        return result

    def get_random(self, n: int = 10) -> List[State]:
        indices = np.random.choice(len(self._node_ids), size=min(n, len(self._node_ids)), replace=False)
        return [self._make_state(self._node_ids[i]) for i in indices]

    def get_state(self, id: str) -> State:
        return self._make_state(id)

    def embed(self, state: State) -> np.ndarray:
        if state.id in self._id_to_idx:
            return self._embeddings[self._id_to_idx[state.id]]
        return np.zeros(self._embedding_dim)

    def size(self) -> int:
        return len(self._node_ids)
