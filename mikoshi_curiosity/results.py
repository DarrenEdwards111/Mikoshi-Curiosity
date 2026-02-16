"""Discovery results and exploration statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mikoshi_curiosity.space import State
from mikoshi_curiosity.memory import ExplorationMemory


@dataclass
class Discovery:
    """A single interesting finding."""

    state: State
    score: float
    reason: str = ""
    strategy_scores: Dict[str, float] = field(default_factory=dict)
    path: List[State] = field(default_factory=list)

    def __repr__(self):
        return f"Discovery(id={self.state.id!r}, score={self.score:.3f}, reason={self.reason!r})"


@dataclass
class ExplorationStats:
    """Statistics about an exploration run."""

    steps: int = 0
    states_visited: int = 0
    states_scored: int = 0
    frontier_size: int = 0
    discoveries_found: int = 0
    coverage: float = 0.0  # fraction of space explored (if known)


class ExplorationResult:
    """Container for exploration outcomes."""

    def __init__(
        self,
        discoveries: Optional[List[Discovery]] = None,
        stats: Optional[ExplorationStats] = None,
        memory: Optional[ExplorationMemory] = None,
    ):
        self.discoveries = discoveries or []
        self.stats = stats or ExplorationStats()
        self.memory = memory

    def top(self, n: int = 10) -> List[Discovery]:
        """Return top-*n* discoveries by score."""
        return sorted(self.discoveries, key=lambda d: d.score, reverse=True)[:n]

    def by_strategy(self, strategy: str) -> List[Discovery]:
        """Return discoveries sorted by a specific strategy score."""
        return sorted(
            self.discoveries,
            key=lambda d: d.strategy_scores.get(strategy, 0),
            reverse=True,
        )

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Exploration complete: {self.stats.steps} steps, {self.stats.states_visited} states visited",
            f"Discoveries: {len(self.discoveries)}",
        ]
        for i, d in enumerate(self.top(5), 1):
            lines.append(f"  {i}. [{d.score:.3f}] {d.state.id} — {d.reason}")
        return "\n".join(lines)

    def to_dataframe(self):
        """Convert discoveries to a pandas DataFrame."""
        import pandas as pd

        rows = []
        for d in self.discoveries:
            row = {"id": d.state.id, "score": d.score, "reason": d.reason}
            row.update({f"score_{k}": v for k, v in d.strategy_scores.items()})
            row.update(d.state.features)
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self):
        return f"ExplorationResult(discoveries={len(self.discoveries)}, steps={self.stats.steps})"
