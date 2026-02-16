"""Go-Explore style exploration memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from mikoshi_curiosity.space import State


@dataclass
class MemoryEntry:
    """A remembered state with exploration metadata."""

    state: State
    score: float = 0.0
    visit_count: int = 0
    source: Optional[str] = None
    children_explored: int = 0
    is_frontier: bool = True

    @property
    def id(self) -> str:
        return self.state.id


class ExplorationMemory:
    """Stores visited states and supports nearest-neighbour queries."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.states: Dict[str, MemoryEntry] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_ids: List[str] = []
        self._dirty = True

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, state: State, score: float = 0.0, source: Optional[str] = None) -> MemoryEntry:
        """Add or update a state in memory."""
        if state.id in self.states:
            entry = self.states[state.id]
            entry.visit_count += 1
            if score > entry.score:
                entry.score = score
            return entry

        entry = MemoryEntry(state=state, score=score, visit_count=1, source=source)
        self.states[state.id] = entry
        self._dirty = True

        if len(self.states) > self.max_size:
            self.prune()

        return entry

    def has_visited(self, state: State, threshold: float = 0.95) -> bool:
        """Check if *state* (or something very similar) has been visited."""
        if state.id in self.states:
            return True
        if threshold < 1.0 and state.embedding is not None and len(self.states) > 0:
            neighbors = self.nearest_neighbors(state, k=1)
            if neighbors:
                dist = float(np.linalg.norm(state.embedding - neighbors[0].state.embedding))
                # Similarity = 1 / (1 + dist); if >= threshold → visited
                similarity = 1.0 / (1.0 + dist)
                return similarity >= threshold
        return False

    def nearest_neighbors(self, state: State, k: int = 5) -> List[MemoryEntry]:
        """Return the *k* nearest remembered states by embedding distance."""
        if state.embedding is None or len(self.states) == 0:
            return []

        self._rebuild_index()
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        dists = np.linalg.norm(self._embeddings - state.embedding, axis=1)
        # Exclude self
        mask = np.array([eid != state.id for eid in self._embedding_ids])
        dists_masked = np.where(mask, dists, np.inf)
        k_actual = min(k, int(mask.sum()))
        if k_actual == 0:
            return []
        # argpartition kth is 0-indexed, so use k_actual-1
        indices = np.argpartition(dists_masked, min(k_actual - 1, len(dists_masked) - 1))[:k_actual]
        indices = indices[np.argsort(dists_masked[indices])]
        return [self.states[self._embedding_ids[i]] for i in indices]

    def get_frontier(self, n: int = 10) -> List[MemoryEntry]:
        """Return the most promising frontier states (high score, still explorable)."""
        frontier = [e for e in self.states.values() if e.is_frontier]
        frontier.sort(key=lambda e: e.score, reverse=True)
        return frontier[:n]

    def get_archive(self) -> List[MemoryEntry]:
        """Return all remembered states."""
        return list(self.states.values())

    def visit_count(self, state: State) -> int:
        entry = self.states.get(state.id)
        return entry.visit_count if entry else 0

    def mark_explored(self, state_id: str):
        """Mark a state as fully explored (no longer frontier)."""
        if state_id in self.states:
            self.states[state_id].is_frontier = False
            self.states[state_id].children_explored += 1

    def prune(self):
        """Remove least interesting states if over capacity."""
        if len(self.states) <= self.max_size:
            return
        entries = sorted(self.states.values(), key=lambda e: e.score)
        to_remove = len(self.states) - self.max_size
        for entry in entries[:to_remove]:
            del self.states[entry.id]
        self._dirty = True

    def __len__(self) -> int:
        return len(self.states)

    def __contains__(self, state_id: str) -> bool:
        return state_id in self.states

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_index(self):
        if not self._dirty:
            return
        ids = []
        vecs = []
        for sid, entry in self.states.items():
            if entry.state.embedding is not None:
                ids.append(sid)
                vecs.append(entry.state.embedding)
        if vecs:
            self._embeddings = np.array(vecs)
            self._embedding_ids = ids
        else:
            self._embeddings = None
            self._embedding_ids = []
        self._dirty = False
