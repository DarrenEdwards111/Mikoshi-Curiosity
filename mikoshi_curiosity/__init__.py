"""Mikoshi Curiosity — domain-agnostic exploration engine."""

__version__ = "0.1.0"

from mikoshi_curiosity.space import State, StateSpace
from mikoshi_curiosity.engine import CuriosityEngine
from mikoshi_curiosity.memory import ExplorationMemory, MemoryEntry
from mikoshi_curiosity.prediction import PredictionModel
from mikoshi_curiosity.results import Discovery, ExplorationResult, ExplorationStats
from mikoshi_curiosity.scoring import (
    novelty_score,
    surprise_score,
    diversity_score,
    serendipity_score,
    diminishing_returns,
    balanced_score,
)

__all__ = [
    "State",
    "StateSpace",
    "CuriosityEngine",
    "ExplorationMemory",
    "MemoryEntry",
    "PredictionModel",
    "Discovery",
    "ExplorationResult",
    "ExplorationStats",
    "novelty_score",
    "surprise_score",
    "diversity_score",
    "serendipity_score",
    "diminishing_returns",
    "balanced_score",
]
