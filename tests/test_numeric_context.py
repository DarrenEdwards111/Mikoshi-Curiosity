"""Tests for NumericSpace context."""

import numpy as np
import pytest

from mikoshi_curiosity.space import State
from mikoshi_curiosity.contexts.numeric import NumericSpace


def simple_eval(params):
    return sum(v ** 2 for v in params.values())


def make_space():
    return NumericSpace(
        dimensions={"x": (-10, 10), "y": (-10, 10)},
        eval_fn=simple_eval,
    )


class TestNumericInit:
    def test_basic_init(self):
        space = make_space()
        assert len(space.dim_names) == 2

    def test_size_infinite(self):
        space = make_space()
        assert space.size() == -1

    def test_custom_step_size(self):
        space = NumericSpace({"x": (0, 1)}, eval_fn=lambda p: p["x"], step_size=0.5)
        assert space.step_size == 0.5


class TestNumericNeighbors:
    def test_get_neighbors(self):
        space = make_space()
        s = space._make_state({"x": 0, "y": 0}, "origin")
        nb = space.get_neighbors(s, n=5)
        assert len(nb) == 5

    def test_neighbors_have_embeddings(self):
        space = make_space()
        s = space._make_state({"x": 0, "y": 0}, "origin")
        nb = space.get_neighbors(s, n=3)
        assert all(n.embedding is not None for n in nb)

    def test_neighbors_within_bounds(self):
        space = make_space()
        s = space._make_state({"x": 9.9, "y": 9.9}, "corner")
        nb = space.get_neighbors(s, n=20)
        for n in nb:
            assert -10 <= n.features["x"] <= 10
            assert -10 <= n.features["y"] <= 10

    def test_neighbors_near_parent(self):
        space = NumericSpace({"x": (0, 100)}, eval_fn=lambda p: p["x"], step_size=0.01)
        s = space._make_state({"x": 50}, "mid")
        nb = space.get_neighbors(s, n=10)
        for n in nb:
            assert abs(n.features["x"] - 50) < 20  # within ~20% of range


class TestNumericRandom:
    def test_get_random(self):
        space = make_space()
        r = space.get_random(10)
        assert len(r) == 10

    def test_random_within_bounds(self):
        space = make_space()
        r = space.get_random(50)
        for s in r:
            assert -10 <= s.features["x"] <= 10
            assert -10 <= s.features["y"] <= 10

    def test_random_unique_ids(self):
        space = make_space()
        r = space.get_random(10)
        ids = [s.id for s in r]
        assert len(set(ids)) == 10


class TestNumericState:
    def test_make_state(self):
        space = make_space()
        s = space._make_state({"x": 1, "y": 2})
        assert s.features["x"] == 1
        assert s.features["y"] == 2

    def test_eval_score_in_metadata(self):
        space = make_space()
        s = space._make_state({"x": 3, "y": 4})
        assert s.metadata["eval_score"] == 25  # 9 + 16

    def test_embedding_normalized(self):
        space = make_space()
        s = space._make_state({"x": -10, "y": -10})
        assert np.allclose(s.embedding, [0, 0])
        s2 = space._make_state({"x": 10, "y": 10})
        assert np.allclose(s2.embedding, [1, 1])

    def test_embed(self):
        space = make_space()
        s = space._make_state({"x": 0, "y": 0})
        emb = space.embed(s)
        assert emb.shape == (2,)

    def test_distance(self):
        space = make_space()
        s1 = space._make_state({"x": 0, "y": 0})
        s2 = space._make_state({"x": 10, "y": 10})
        d = space.distance(s1, s2)
        assert d > 0

    def test_get_state_raises(self):
        space = make_space()
        with pytest.raises(NotImplementedError):
            space.get_state("p1")


class TestNumericHighDim:
    def test_many_dimensions(self):
        dims = {f"d{i}": (0, 1) for i in range(20)}
        space = NumericSpace(dims, eval_fn=lambda p: sum(p.values()))
        r = space.get_random(5)
        assert len(r) == 5
        assert r[0].embedding.shape == (20,)

    def test_single_dimension(self):
        space = NumericSpace({"x": (0, 1)}, eval_fn=lambda p: p["x"])
        r = space.get_random(5)
        assert r[0].embedding.shape == (1,)


class TestNumericEval:
    def test_eval_called(self):
        calls = []
        def my_eval(params):
            calls.append(params)
            return 0.0
        space = NumericSpace({"x": (0, 1)}, eval_fn=my_eval)
        space.get_random(3)
        assert len(calls) == 3

    def test_interesting_landscape(self):
        # Eval with a peak
        def peaked(p):
            x = p["x"]
            return np.exp(-((x - 5) ** 2))
        space = NumericSpace({"x": (0, 10)}, eval_fn=peaked)
        r = space.get_random(100)
        scores = [s.metadata["eval_score"] for s in r]
        assert max(scores) > 0.5  # should find near peak
