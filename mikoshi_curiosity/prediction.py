"""Simple online prediction model for surprise detection."""

from __future__ import annotations

from typing import List

import numpy as np

from mikoshi_curiosity.space import State


class PredictionModel:
    """Learns to predict neighbour embeddings from a state embedding.

    Uses a simple linear model updated via online gradient descent.
    Prediction error signals *surprise*.
    """

    def __init__(self, embedding_dim: int = 128, learning_rate: float = 0.01):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self._fitted = False

    def _ensure_weights(self, dim: int):
        if self.weights is None or self.weights.shape[0] != dim:
            self.embedding_dim = dim
            rng = np.random.default_rng(42)
            self.weights = rng.normal(0, 0.01, (dim, dim))
            self.bias = np.zeros(dim)
            self._fitted = False

    def predict_neighbors(self, state: State) -> np.ndarray:
        """Predict the mean embedding of neighbours."""
        if state.embedding is None:
            raise ValueError("State has no embedding")
        self._ensure_weights(len(state.embedding))
        return self.weights @ state.embedding + self.bias

    def prediction_error(self, state: State, actual_neighbors: List[State]) -> float:
        """Return MSE between predicted and actual mean neighbour embedding."""
        if not actual_neighbors:
            return 0.0
        embs = [n.embedding for n in actual_neighbors if n.embedding is not None]
        if not embs:
            return 0.0
        actual_mean = np.mean(embs, axis=0)
        predicted = self.predict_neighbors(state)
        return float(np.mean((predicted - actual_mean) ** 2))

    def update(self, state: State, actual_neighbors: List[State]):
        """Online gradient descent update."""
        if state.embedding is None:
            return
        embs = [n.embedding for n in actual_neighbors if n.embedding is not None]
        if not embs:
            return
        self._ensure_weights(len(state.embedding))
        actual_mean = np.mean(embs, axis=0)
        predicted = self.predict_neighbors(state)
        error = predicted - actual_mean
        # Gradient descent
        self.weights -= self.learning_rate * np.outer(error, state.embedding)
        self.bias -= self.learning_rate * error
        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._fitted
