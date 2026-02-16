"""Scoring functions for state interestingness."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from mikoshi_curiosity.space import State
from mikoshi_curiosity.memory import ExplorationMemory
from mikoshi_curiosity.prediction import PredictionModel


def novelty_score(state: State, memory: ExplorationMemory, k: int = 5) -> float:
    """Mean distance to k-nearest neighbours in memory. Higher = more novel."""
    if state.embedding is None or len(memory) == 0:
        return 1.0
    neighbors = memory.nearest_neighbors(state, k=k)
    if not neighbors:
        return 1.0
    dists = [
        float(np.linalg.norm(state.embedding - n.state.embedding))
        for n in neighbors
        if n.state.embedding is not None
    ]
    if not dists:
        return 1.0
    return float(np.mean(dists))


def surprise_score(state: State, prediction_model: PredictionModel, neighbors: Optional[List[State]] = None) -> float:
    """Prediction error — how surprising is this state's neighbourhood?"""
    if state.embedding is None:
        return 0.0
    if neighbors is None:
        return 0.0
    return prediction_model.prediction_error(state, neighbors)


def diversity_score(state: State, current_discoveries: List[State]) -> float:
    """Minimum distance to any current discovery. Higher = more diverse."""
    if state.embedding is None or not current_discoveries:
        return 1.0
    dists = []
    for d in current_discoveries:
        if d.embedding is not None:
            dists.append(float(np.linalg.norm(state.embedding - d.embedding)))
    if not dists:
        return 1.0
    return float(np.min(dists))


def serendipity_score(
    state: State,
    user_profile: Optional[np.ndarray],
    memory: ExplorationMemory,
    k: int = 5,
) -> float:
    """Unexpected AND relevant: novelty × relevance to user profile."""
    nov = novelty_score(state, memory, k=k)
    if user_profile is None or state.embedding is None:
        return nov
    # Relevance = cosine similarity to profile
    dot = float(np.dot(state.embedding, user_profile))
    norm_s = float(np.linalg.norm(state.embedding))
    norm_p = float(np.linalg.norm(user_profile))
    if norm_s == 0 or norm_p == 0:
        return nov
    relevance = max(0.0, dot / (norm_s * norm_p))
    return nov * relevance


def diminishing_returns(state: State, visit_count: int) -> float:
    """Decay factor for repeatedly visited states. Returns value in (0, 1]."""
    if visit_count <= 0:
        return 1.0
    return 1.0 / (1.0 + visit_count)


def balanced_score(
    state: State,
    memory: ExplorationMemory,
    prediction_model: PredictionModel,
    discoveries: List[State],
    profile: Optional[np.ndarray] = None,
    neighbors: Optional[List[State]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Weighted combination of all scoring signals."""
    w = weights or {
        "novelty": 0.3,
        "surprise": 0.2,
        "diversity": 0.3,
        "serendipity": 0.1,
        "diminishing": 0.1,
    }
    visit_ct = memory.visit_count(state)
    scores = {
        "novelty": novelty_score(state, memory),
        "surprise": surprise_score(state, prediction_model, neighbors),
        "diversity": diversity_score(state, discoveries),
        "serendipity": serendipity_score(state, profile, memory),
        "diminishing": diminishing_returns(state, visit_ct),
    }
    total = sum(w.get(k, 0) * v for k, v in scores.items())
    return total
