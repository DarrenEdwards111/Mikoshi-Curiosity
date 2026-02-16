"""Tests for State and StateSpace base classes."""

import numpy as np
import pytest

from mikoshi_curiosity.space import State, StateSpace


class TestState:
    def test_create(self):
        s = State(id="a")
        assert s.id == "a"
        assert s.features == {}
        assert s.embedding is None

    def test_with_embedding(self):
        s = State(id="a", embedding=np.array([1.0, 2.0]))
        assert s.embedding.shape == (2,)

    def test_embedding_from_list(self):
        s = State(id="a", embedding=[1.0, 2.0])
        assert isinstance(s.embedding, np.ndarray)

    def test_hash(self):
        s1 = State(id="a")
        s2 = State(id="a")
        assert hash(s1) == hash(s2)

    def test_eq(self):
        assert State(id="a") == State(id="a")
        assert State(id="a") != State(id="b")

    def test_eq_non_state(self):
        assert State(id="a").__eq__("a") is NotImplemented

    def test_features(self):
        s = State(id="a", features={"x": 1})
        assert s.features["x"] == 1

    def test_metadata(self):
        s = State(id="a", metadata={"src": "test"})
        assert s.metadata["src"] == "test"

    def test_set_membership(self):
        s = {State(id="a"), State(id="b"), State(id="a")}
        assert len(s) == 2


class TestStateSpace:
    def test_abstract_methods(self):
        ss = StateSpace()
        with pytest.raises(NotImplementedError):
            ss.get_neighbors(State(id="a"))
        with pytest.raises(NotImplementedError):
            ss.get_random()
        with pytest.raises(NotImplementedError):
            ss.get_state("a")
        with pytest.raises(NotImplementedError):
            ss.embed(State(id="a"))
        with pytest.raises(NotImplementedError):
            ss.size()

    def test_distance_default(self):
        ss = StateSpace()
        a = State(id="a", embedding=np.array([0.0, 0.0]))
        b = State(id="b", embedding=np.array([3.0, 4.0]))
        assert abs(ss.distance(a, b) - 5.0) < 1e-10
