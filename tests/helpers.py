"""Shared test helpers."""

import numpy as np
from mikoshi_curiosity.space import State, StateSpace


def make_state(id: str = "s1", dim: int = 8, features: dict = None, rng=None) -> State:
    if rng is None:
        rng = np.random.default_rng(hash(id) % (2**31))
    return State(
        id=id,
        features=features or {"val": rng.random()},
        embedding=rng.normal(0, 1, dim),
    )


def make_states(n: int, dim: int = 8) -> list:
    return [make_state(f"s{i}", dim) for i in range(n)]


class SimpleSpace(StateSpace):
    """Deterministic test space."""

    def __init__(self, states: list[State] = None, n: int = 50, dim: int = 8):
        if states is None:
            states = make_states(n, dim)
        self._states = {s.id: s for s in states}
        self._list = states
        self._dim = dim

    def get_neighbors(self, state: State, n: int = 10) -> list[State]:
        if state.embedding is None:
            return self._list[:n]
        dists = [(s, float(np.linalg.norm(state.embedding - s.embedding)))
                 for s in self._list if s.id != state.id and s.embedding is not None]
        dists.sort(key=lambda x: x[1])
        return [s for s, _ in dists[:n]]

    def get_random(self, n: int = 10) -> list[State]:
        rng = np.random.default_rng()
        idx = rng.choice(len(self._list), size=min(n, len(self._list)), replace=False)
        return [self._list[i] for i in idx]

    def get_state(self, id: str) -> State:
        return self._states[id]

    def embed(self, state: State) -> np.ndarray:
        if state.id in self._states:
            return self._states[state.id].embedding
        return np.zeros(self._dim)

    def size(self) -> int:
        return len(self._list)
