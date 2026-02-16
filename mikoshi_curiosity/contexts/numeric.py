"""Continuous numeric/parameter space exploration."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from mikoshi_curiosity.space import State, StateSpace


class NumericSpace(StateSpace):
    """Explore a continuous parameter space.

    Parameters
    ----------
    dimensions : dict[str, tuple]
        Mapping of dimension name → (min, max).
    eval_fn : callable
        Evaluates a parameter dict and returns a score.
    step_size : float
        Fraction of range to perturb when generating neighbours.
    """

    def __init__(
        self,
        dimensions: Dict[str, Tuple[float, float]],
        eval_fn: Callable[[Dict[str, float]], float],
        step_size: float = 0.1,
    ):
        self.dimensions = dimensions
        self.dim_names = list(dimensions.keys())
        self.bounds = np.array([dimensions[d] for d in self.dim_names])
        self.eval_fn = eval_fn
        self.step_size = step_size
        self._rng = np.random.default_rng(42)
        self._counter = 0

    def _params_to_embedding(self, params: Dict[str, float]) -> np.ndarray:
        vals = np.array([params[d] for d in self.dim_names])
        # Normalise to [0, 1]
        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        ranges[ranges == 0] = 1.0
        return (vals - self.bounds[:, 0]) / ranges

    def _embedding_to_params(self, emb: np.ndarray) -> Dict[str, float]:
        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        vals = emb * ranges + self.bounds[:, 0]
        return {d: float(vals[i]) for i, d in enumerate(self.dim_names)}

    def _make_state(self, params: Dict[str, float], state_id: Optional[str] = None) -> State:
        if state_id is None:
            self._counter += 1
            state_id = f"p{self._counter}"
        emb = self._params_to_embedding(params)
        score = self.eval_fn(params)
        return State(
            id=state_id,
            features=dict(params),
            embedding=emb,
            metadata={"eval_score": score},
        )

    def get_neighbors(self, state: State, n: int = 10) -> List[State]:
        params = {d: state.features[d] for d in self.dim_names}
        neighbors = []
        for _ in range(n):
            new_params = {}
            for d in self.dim_names:
                lo, hi = self.dimensions[d]
                rng = hi - lo
                delta = self._rng.normal(0, self.step_size * rng)
                new_params[d] = float(np.clip(params[d] + delta, lo, hi))
            neighbors.append(self._make_state(new_params))
        return neighbors

    def get_random(self, n: int = 10) -> List[State]:
        states = []
        for _ in range(n):
            params = {}
            for d in self.dim_names:
                lo, hi = self.dimensions[d]
                params[d] = float(self._rng.uniform(lo, hi))
            states.append(self._make_state(params))
        return states

    def get_state(self, id: str) -> State:
        raise NotImplementedError("NumericSpace states are generated, not stored")

    def embed(self, state: State) -> np.ndarray:
        if state.embedding is not None:
            return state.embedding
        params = {d: state.features[d] for d in self.dim_names}
        return self._params_to_embedding(params)

    def size(self) -> int:
        return -1  # Continuous/infinite
