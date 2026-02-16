"""Core curiosity-driven exploration engine."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np

from mikoshi_curiosity.space import State, StateSpace
from mikoshi_curiosity.memory import ExplorationMemory
from mikoshi_curiosity.prediction import PredictionModel
from mikoshi_curiosity.results import Discovery, ExplorationResult, ExplorationStats
from mikoshi_curiosity import scoring as sc

STRATEGIES = ("novelty", "surprise", "diversity", "serendipity", "balanced")


class CuriosityEngine:
    """Domain-agnostic curiosity engine.

    Parameters
    ----------
    space : StateSpace
        The state space to explore.
    strategy : str
        Scoring strategy: novelty | surprise | diversity | serendipity | balanced.
    memory_size : int
        Maximum states to keep in exploration memory.
    user_profile : np.ndarray, optional
        Embedding representing user interests (used by serendipity).
    score_weights : dict, optional
        Custom weights for balanced strategy.
    """

    def __init__(
        self,
        space: StateSpace,
        strategy: str = "novelty",
        memory_size: int = 10000,
        user_profile: Optional[np.ndarray] = None,
        score_weights: Optional[Dict[str, float]] = None,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}. Choose from {STRATEGIES}")
        self.space = space
        self.strategy = strategy
        self.memory = ExplorationMemory(memory_size)
        self.prediction_model = PredictionModel()
        self.user_profile = user_profile
        self.score_weights = score_weights
        self._discoveries: List[Discovery] = []

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, state: State, neighbors: Optional[List[State]] = None) -> float:
        """Score a state's interestingness using the current strategy."""
        discovery_states = [d.state for d in self._discoveries]
        if self.strategy == "novelty":
            return sc.novelty_score(state, self.memory)
        elif self.strategy == "surprise":
            return sc.surprise_score(state, self.prediction_model, neighbors)
        elif self.strategy == "diversity":
            return sc.diversity_score(state, discovery_states)
        elif self.strategy == "serendipity":
            return sc.serendipity_score(state, self.user_profile, self.memory)
        else:  # balanced
            return sc.balanced_score(
                state,
                self.memory,
                self.prediction_model,
                discovery_states,
                profile=self.user_profile,
                neighbors=neighbors,
                weights=self.score_weights,
            )

    def _strategy_scores(self, state: State, neighbors: Optional[List[State]] = None) -> Dict[str, float]:
        discovery_states = [d.state for d in self._discoveries]
        return {
            "novelty": sc.novelty_score(state, self.memory),
            "surprise": sc.surprise_score(state, self.prediction_model, neighbors),
            "diversity": sc.diversity_score(state, discovery_states),
            "serendipity": sc.serendipity_score(state, self.user_profile, self.memory),
            "diminishing": sc.diminishing_returns(state, self.memory.visit_count(state)),
        }

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def explore(
        self,
        seed: Union[State, List[State]],
        budget: int = 100,
        neighbors_per_step: int = 10,
        discovery_threshold: float = 0.0,
        callback: Optional[Callable] = None,
    ) -> ExplorationResult:
        """Run the exploration loop.

        Parameters
        ----------
        seed : State or list[State]
            Starting point(s).
        budget : int
            Maximum exploration steps.
        neighbors_per_step : int
            How many neighbours to fetch per step.
        discovery_threshold : float
            Minimum score to record as a discovery.
        callback : callable, optional
            Called with (step, discovery_or_none) each iteration.

        Returns
        -------
        ExplorationResult
        """
        seeds = [seed] if isinstance(seed, State) else list(seed)
        self._discoveries = []

        # Ensure embeddings
        for s in seeds:
            if s.embedding is None:
                s.embedding = self.space.embed(s)
            self.memory.add(s, score=0.0, source="seed")

        stats = ExplorationStats()

        for step in range(budget):
            # Pick best frontier state
            frontier = self.memory.get_frontier(n=20)
            if not frontier:
                break

            # Select state with highest score
            best_entry = max(frontier, key=lambda e: e.score + 1.0 / (1.0 + e.children_explored))
            current = best_entry.state

            # Get neighbours
            neighbors = self.space.get_neighbors(current, n=neighbors_per_step)
            stats.states_visited += 1

            # Update prediction model
            self.prediction_model.update(current, neighbors)
            self.memory.mark_explored(current.id)

            discovery_this_step = None
            for nb in neighbors:
                if nb.embedding is None:
                    nb.embedding = self.space.embed(nb)
                s = self.score(nb, neighbors=neighbors)
                ss = self._strategy_scores(nb, neighbors=neighbors)
                stats.states_scored += 1

                entry = self.memory.add(nb, score=s)

                if s > discovery_threshold:
                    # Determine reason
                    top_strategy = max(ss, key=ss.get)
                    reason = self._reason(top_strategy, ss[top_strategy])
                    disc = Discovery(
                        state=nb,
                        score=s,
                        reason=reason,
                        strategy_scores=ss,
                        path=[current, nb],
                    )
                    self._discoveries.append(disc)
                    discovery_this_step = disc

            stats.steps = step + 1
            stats.frontier_size = len(self.memory.get_frontier())

            if callback is not None:
                callback(step, discovery_this_step)

        stats.discoveries_found = len(self._discoveries)
        space_size = 0
        try:
            space_size = self.space.size()
        except (NotImplementedError, Exception):
            pass
        if space_size > 0:
            stats.coverage = len(self.memory) / space_size

        return ExplorationResult(
            discoveries=sorted(self._discoveries, key=lambda d: d.score, reverse=True),
            stats=stats,
            memory=self.memory,
        )

    def get_discoveries(self, top_n: int = 20) -> List[Discovery]:
        """Return current top discoveries."""
        return sorted(self._discoveries, key=lambda d: d.score, reverse=True)[:top_n]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reason(strategy: str, value: float) -> str:
        templates = {
            "novelty": "High novelty — unlike anything seen before",
            "surprise": "Surprising — violates predictions",
            "diversity": "Diverse — expands the discovery set",
            "serendipity": "Serendipitous — unexpected yet relevant",
            "diminishing": "Fresh territory — rarely visited",
        }
        return templates.get(strategy, f"Interesting ({strategy}={value:.2f})")
