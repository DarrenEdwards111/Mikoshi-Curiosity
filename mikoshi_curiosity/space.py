"""State and StateSpace abstractions for exploration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class State:
    """A point in exploration space."""

    id: str
    features: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.asarray(self.embedding, dtype=np.float64)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, State):
            return self.id == other.id
        return NotImplemented


class StateSpace:
    """Abstract state space to explore.

    Subclass this and implement the abstract methods to define a domain.
    """

    def get_neighbors(self, state: State, n: int = 10) -> List[State]:
        """Return up to *n* neighboring states."""
        raise NotImplementedError

    def get_random(self, n: int = 10) -> List[State]:
        """Return *n* random states from the space."""
        raise NotImplementedError

    def get_state(self, id: str) -> State:
        """Retrieve a state by its id."""
        raise NotImplementedError

    def embed(self, state: State) -> np.ndarray:
        """Return a vector embedding for *state*."""
        raise NotImplementedError

    def distance(self, a: State, b: State) -> float:
        """Return distance between two states (default: Euclidean on embeddings)."""
        ea = a.embedding if a.embedding is not None else self.embed(a)
        eb = b.embedding if b.embedding is not None else self.embed(b)
        return float(np.linalg.norm(ea - eb))

    def size(self) -> int:
        """Return the (approximate) number of states in the space."""
        raise NotImplementedError
