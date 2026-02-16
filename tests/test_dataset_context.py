"""Tests for DatasetSpace context."""

import numpy as np
import pytest

from mikoshi_curiosity.space import State

pd = pytest.importorskip("pandas")
from mikoshi_curiosity.contexts.dataset import DatasetSpace


def sample_df(n=50, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(5, 2, n),
        "c": rng.normal(-3, 0.5, n),
        "label": [f"row{i}" for i in range(n)],
    })


class TestDatasetInit:
    def test_from_dataframe(self):
        df = sample_df()
        space = DatasetSpace(df)
        assert space.size() == 50

    def test_from_csv(self, tmp_path):
        df = sample_df()
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        space = DatasetSpace(str(path))
        assert space.size() == 50

    def test_numeric_cols_detected(self):
        df = sample_df()
        space = DatasetSpace(df)
        assert "a" in space._numeric_cols
        assert "label" not in space._numeric_cols


class TestDatasetNeighbors:
    def test_get_neighbors(self):
        space = DatasetSpace(sample_df())
        s = space.get_state("0")
        nb = space.get_neighbors(s, n=5)
        assert len(nb) == 5
        assert all(isinstance(x, State) for x in nb)

    def test_neighbors_exclude_self(self):
        space = DatasetSpace(sample_df())
        s = space.get_state("0")
        nb = space.get_neighbors(s, n=5)
        assert all(x.id != "0" for x in nb)

    def test_neighbors_have_embeddings(self):
        space = DatasetSpace(sample_df())
        s = space.get_state("0")
        nb = space.get_neighbors(s, n=3)
        assert all(x.embedding is not None for x in nb)

    def test_neighbors_are_close(self):
        space = DatasetSpace(sample_df())
        s = space.get_state("0")
        nb = space.get_neighbors(s, n=5)
        dists = [float(np.linalg.norm(s.embedding - x.embedding)) for x in nb]
        assert dists == sorted(dists)


class TestDatasetRandom:
    def test_get_random(self):
        space = DatasetSpace(sample_df())
        r = space.get_random(n=10)
        assert len(r) == 10

    def test_random_unique(self):
        space = DatasetSpace(sample_df())
        r = space.get_random(n=10)
        ids = [s.id for s in r]
        assert len(set(ids)) == 10


class TestDatasetState:
    def test_get_state(self):
        space = DatasetSpace(sample_df())
        s = space.get_state("5")
        assert s.id == "5"
        assert "a" in s.features
        assert s.embedding is not None

    def test_embed(self):
        space = DatasetSpace(sample_df())
        s = space.get_state("3")
        emb = space.embed(s)
        assert emb.shape == (3,)  # 3 numeric cols

    def test_distance(self):
        space = DatasetSpace(sample_df())
        s1 = space.get_state("0")
        s2 = space.get_state("1")
        d = space.distance(s1, s2)
        assert d >= 0

    def test_features_match_row(self):
        df = sample_df()
        space = DatasetSpace(df)
        s = space.get_state("0")
        assert abs(s.features["a"] - df.iloc[0]["a"]) < 1e-10


class TestDatasetAnomaly:
    def test_outlier_is_distant(self):
        df = sample_df(n=50)
        # Add outlier
        df.loc[50] = [100, 100, 100, "outlier"]
        space = DatasetSpace(df)
        outlier = space.get_state("50")
        normal = space.get_state("0")
        # Outlier should be far from normal states
        nb_out = space.get_neighbors(outlier, n=5)
        nb_norm = space.get_neighbors(normal, n=5)
        dist_out = np.mean([float(np.linalg.norm(outlier.embedding - x.embedding)) for x in nb_out])
        dist_norm = np.mean([float(np.linalg.norm(normal.embedding - x.embedding)) for x in nb_norm])
        assert dist_out > dist_norm


class TestDatasetEdgeCases:
    def test_single_row(self):
        df = pd.DataFrame({"x": [1.0]})
        space = DatasetSpace(df)
        assert space.size() == 1

    def test_no_numeric_cols(self):
        df = pd.DataFrame({"name": ["a", "b", "c"]})
        space = DatasetSpace(df)
        s = space.get_state("0")
        assert s.embedding is not None
        assert len(s.embedding) == 0

    def test_with_nans(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [4.0, 5.0, np.nan]})
        space = DatasetSpace(df)
        s = space.get_state("1")
        assert not np.any(np.isnan(s.embedding))
