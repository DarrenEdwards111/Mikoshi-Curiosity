"""Tabular dataset exploration context."""

from __future__ import annotations

from typing import List, Union

import numpy as np

from mikoshi_curiosity.space import State, StateSpace


class DatasetSpace(StateSpace):
    """Explore rows/columns/patterns in a DataFrame or CSV.

    Each row becomes a State; numeric columns are used for embeddings.
    Neighbours are the most similar rows by Euclidean distance.
    """

    def __init__(self, data: Union["pd.DataFrame", str]):
        import pandas as pd

        if isinstance(data, str):
            data = pd.read_csv(data)
        self.data: pd.DataFrame = data.reset_index(drop=True)
        self._numeric_cols = list(self.data.select_dtypes(include=[np.number]).columns)
        # Pre-compute normalised numeric matrix for embeddings
        num = self.data[self._numeric_cols].values.astype(np.float64)
        self._means = np.nanmean(num, axis=0)
        self._stds = np.nanstd(num, axis=0)
        self._stds[self._stds == 0] = 1.0
        self._matrix = (np.nan_to_num(num, nan=0.0) - self._means) / self._stds

    def _row_to_state(self, idx: int) -> State:
        row = self.data.iloc[idx]
        features = row.to_dict()
        embedding = self._matrix[idx]
        return State(id=str(idx), features=features, embedding=embedding, metadata={"row_index": idx})

    def get_neighbors(self, state: State, n: int = 10) -> List[State]:
        dists = np.linalg.norm(self._matrix - state.embedding, axis=1)
        idx = int(state.id) if state.id.isdigit() else -1
        if 0 <= idx < len(dists):
            dists[idx] = np.inf
        nearest = np.argpartition(dists, min(n, len(dists) - 1))[:n]
        nearest = nearest[np.argsort(dists[nearest])]
        return [self._row_to_state(int(i)) for i in nearest]

    def get_random(self, n: int = 10) -> List[State]:
        indices = np.random.choice(len(self.data), size=min(n, len(self.data)), replace=False)
        return [self._row_to_state(int(i)) for i in indices]

    def get_state(self, id: str) -> State:
        return self._row_to_state(int(id))

    def embed(self, state: State) -> np.ndarray:
        if state.embedding is not None:
            return state.embedding
        return self._row_to_state(int(state.id)).embedding

    def size(self) -> int:
        return len(self.data)
