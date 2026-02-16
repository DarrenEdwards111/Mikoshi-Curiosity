"""Generic API exploration context."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from mikoshi_curiosity.space import State, StateSpace


class APISpace(StateSpace):
    """Explore any API that returns items with features.

    Parameters
    ----------
    fetch_fn : callable
        ``fetch_fn(query: str) -> list[dict]`` where each dict has at
        least ``id`` and arbitrary feature keys.
    embed_fn : callable, optional
        ``embed_fn(item: dict) -> np.ndarray``. If *None*, a simple
        hash-based embedding is used.
    """

    def __init__(
        self,
        fetch_fn: Callable[[str], List[Dict]],
        embed_fn: Optional[Callable[[Dict], np.ndarray]] = None,
        embedding_dim: int = 64,
    ):
        self.fetch_fn = fetch_fn
        self.embed_fn = embed_fn
        self.embedding_dim = embedding_dim
        self._cache: Dict[str, State] = {}
        self._rng = np.random.default_rng(0)

    def _default_embed(self, item: Dict) -> np.ndarray:
        """Simple deterministic embedding from feature values."""
        rng = np.random.default_rng(hash(str(sorted(item.items()))) % (2**31))
        return rng.normal(0, 1, self.embedding_dim).astype(np.float64)

    def _item_to_state(self, item: Dict) -> State:
        sid = str(item.get("id", id(item)))
        if sid in self._cache:
            return self._cache[sid]
        features = {k: v for k, v in item.items() if k != "id"}
        if self.embed_fn:
            emb = self.embed_fn(item)
        else:
            emb = self._default_embed(item)
        state = State(id=sid, features=features, embedding=emb, metadata=item)
        self._cache[sid] = state
        return state

    def get_neighbors(self, state: State, n: int = 10) -> List[State]:
        # Use state features as query
        query = " ".join(str(v) for v in list(state.features.values())[:5])
        items = self.fetch_fn(query)
        return [self._item_to_state(item) for item in items[:n]]

    def get_random(self, n: int = 10) -> List[State]:
        items = self.fetch_fn("")
        return [self._item_to_state(item) for item in items[:n]]

    def get_state(self, id: str) -> State:
        if id in self._cache:
            return self._cache[id]
        raise KeyError(f"State {id!r} not in cache; fetch it first")

    def embed(self, state: State) -> np.ndarray:
        if state.embedding is not None:
            return state.embedding
        return self._default_embed(state.features)

    def size(self) -> int:
        return len(self._cache)
