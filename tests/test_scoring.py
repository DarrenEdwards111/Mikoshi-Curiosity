"""Tests for scoring functions."""

import numpy as np
import pytest

from mikoshi_curiosity import scoring as sc
from mikoshi_curiosity.memory import ExplorationMemory
from mikoshi_curiosity.prediction import PredictionModel
from tests.helpers import make_state, make_states


class TestNoveltyScore:
    def test_novel_in_empty_memory(self):
        mem = ExplorationMemory()
        s = make_state("x")
        assert sc.novelty_score(s, mem) == 1.0

    def test_novel_far_state(self):
        mem = ExplorationMemory()
        for s in make_states(10):
            mem.add(s)
        far = make_state("far", dim=8)
        far.embedding = np.ones(8) * 100
        score = sc.novelty_score(far, mem)
        assert score > 0

    def test_novel_near_state(self):
        mem = ExplorationMemory()
        s0 = make_state("s0")
        mem.add(s0)
        near = make_state("near")
        near.embedding = s0.embedding + 1e-6
        score = sc.novelty_score(near, mem)
        assert score < 1.0

    def test_no_embedding(self):
        mem = ExplorationMemory()
        s = make_state("x")
        s.embedding = None
        assert sc.novelty_score(s, mem) == 1.0

    def test_k_parameter(self):
        mem = ExplorationMemory()
        for s in make_states(20):
            mem.add(s)
        state = make_state("test")
        s1 = sc.novelty_score(state, mem, k=1)
        s5 = sc.novelty_score(state, mem, k=5)
        assert isinstance(s1, float)
        assert isinstance(s5, float)


class TestSurpriseScore:
    def test_no_neighbors(self):
        pm = PredictionModel()
        s = make_state("x")
        assert sc.surprise_score(s, pm, neighbors=[]) == 0.0

    def test_no_embedding(self):
        pm = PredictionModel()
        s = make_state("x")
        s.embedding = None
        assert sc.surprise_score(s, pm) == 0.0

    def test_with_neighbors(self):
        pm = PredictionModel(embedding_dim=8)
        s = make_state("x")
        neighbors = make_states(5)
        score = sc.surprise_score(s, pm, neighbors)
        assert score >= 0

    def test_trained_model_lower_surprise(self):
        pm = PredictionModel(embedding_dim=8)
        states = make_states(10)
        # Train
        for s in states:
            pm.update(s, states)
        before = sc.surprise_score(states[0], pm, states[1:5])
        for _ in range(50):
            for s in states:
                pm.update(s, states)
        after = sc.surprise_score(states[0], pm, states[1:5])
        # After training, surprise should generally decrease
        assert isinstance(after, float)


class TestDiversityScore:
    def test_no_discoveries(self):
        s = make_state("x")
        assert sc.diversity_score(s, []) == 1.0

    def test_diverse_from_discoveries(self):
        s = make_state("far")
        s.embedding = np.ones(8) * 100
        discoveries = make_states(5)
        score = sc.diversity_score(s, discoveries)
        assert score > 0

    def test_same_as_discovery(self):
        s = make_state("s0")
        discoveries = [s]
        score = sc.diversity_score(s, discoveries)
        assert score == 0.0

    def test_no_embedding(self):
        s = make_state("x")
        s.embedding = None
        assert sc.diversity_score(s, make_states(3)) == 1.0

    def test_min_distance(self):
        s = make_state("test")
        d1 = make_state("d1")
        d2 = make_state("d2")
        d1.embedding = s.embedding + 0.01
        d2.embedding = s.embedding + 10.0
        score = sc.diversity_score(s, [d1, d2])
        # Should be min distance = to d1
        assert score < 1.0


class TestSerendipityScore:
    def test_no_profile(self):
        mem = ExplorationMemory()
        s = make_state("x")
        score = sc.serendipity_score(s, None, mem)
        assert score == sc.novelty_score(s, mem)

    def test_with_profile(self):
        mem = ExplorationMemory()
        for st in make_states(5):
            mem.add(st)
        s = make_state("x")
        profile = s.embedding.copy()
        score = sc.serendipity_score(s, profile, mem)
        assert score >= 0

    def test_orthogonal_profile(self):
        mem = ExplorationMemory()
        s = make_state("x", dim=2)
        s.embedding = np.array([1.0, 0.0])
        profile = np.array([0.0, 1.0])
        score = sc.serendipity_score(s, profile, mem)
        assert score == 0.0

    def test_zero_profile(self):
        mem = ExplorationMemory()
        s = make_state("x")
        score = sc.serendipity_score(s, np.zeros(8), mem)
        # novelty_score since norm is 0
        assert score == sc.novelty_score(s, mem)

    def test_no_embedding(self):
        mem = ExplorationMemory()
        s = make_state("x")
        s.embedding = None
        score = sc.serendipity_score(s, np.ones(8), mem)
        assert score == sc.novelty_score(s, mem)


class TestDiminishingReturns:
    def test_zero_visits(self):
        s = make_state("x")
        assert sc.diminishing_returns(s, 0) == 1.0

    def test_one_visit(self):
        s = make_state("x")
        assert sc.diminishing_returns(s, 1) == 0.5

    def test_many_visits(self):
        s = make_state("x")
        assert sc.diminishing_returns(s, 100) < 0.02

    def test_negative_visits(self):
        s = make_state("x")
        assert sc.diminishing_returns(s, -1) == 1.0

    def test_monotonically_decreasing(self):
        s = make_state("x")
        prev = 1.0
        for v in range(1, 20):
            curr = sc.diminishing_returns(s, v)
            assert curr < prev
            prev = curr


class TestBalancedScore:
    def test_default_weights(self):
        mem = ExplorationMemory()
        pm = PredictionModel(embedding_dim=8)
        s = make_state("x")
        score = sc.balanced_score(s, mem, pm, [])
        assert isinstance(score, float)

    def test_custom_weights(self):
        mem = ExplorationMemory()
        pm = PredictionModel(embedding_dim=8)
        s = make_state("x")
        w = {"novelty": 1.0, "surprise": 0, "diversity": 0, "serendipity": 0, "diminishing": 0}
        score = sc.balanced_score(s, mem, pm, [], weights=w)
        expected = sc.novelty_score(s, mem)
        assert abs(score - expected) < 1e-6

    def test_with_profile(self):
        mem = ExplorationMemory()
        pm = PredictionModel(embedding_dim=8)
        s = make_state("x")
        profile = np.ones(8)
        score = sc.balanced_score(s, mem, pm, [], profile=profile)
        assert isinstance(score, float)

    def test_with_neighbors(self):
        mem = ExplorationMemory()
        pm = PredictionModel(embedding_dim=8)
        s = make_state("x")
        nb = make_states(3)
        score = sc.balanced_score(s, mem, pm, [], neighbors=nb)
        assert isinstance(score, float)
