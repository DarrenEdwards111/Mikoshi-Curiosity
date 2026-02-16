"""Tests for the CuriosityEngine."""

import numpy as np
import pytest

from mikoshi_curiosity.engine import CuriosityEngine, STRATEGIES
from mikoshi_curiosity.results import Discovery, ExplorationResult
from tests.helpers import SimpleSpace, make_state, make_states


class TestEngineInit:
    def test_default_init(self):
        space = SimpleSpace()
        engine = CuriosityEngine(space)
        assert engine.strategy == "novelty"
        assert len(engine.memory) == 0

    def test_all_strategies(self):
        space = SimpleSpace()
        for s in STRATEGIES:
            engine = CuriosityEngine(space, strategy=s)
            assert engine.strategy == s

    def test_invalid_strategy(self):
        space = SimpleSpace()
        with pytest.raises(ValueError):
            CuriosityEngine(space, strategy="invalid")

    def test_custom_memory_size(self):
        space = SimpleSpace()
        engine = CuriosityEngine(space, memory_size=50)
        assert engine.memory.max_size == 50

    def test_user_profile(self):
        space = SimpleSpace()
        profile = np.ones(8)
        engine = CuriosityEngine(space, user_profile=profile)
        assert engine.user_profile is not None


class TestExplore:
    def test_basic_explore(self):
        space = SimpleSpace(n=30)
        engine = CuriosityEngine(space, strategy="novelty")
        seed = space.get_state("s0")
        result = engine.explore(seed, budget=10)
        assert isinstance(result, ExplorationResult)
        assert result.stats.steps > 0

    def test_explore_with_list_seed(self):
        space = SimpleSpace(n=30)
        engine = CuriosityEngine(space)
        seeds = [space.get_state("s0"), space.get_state("s1")]
        result = engine.explore(seeds, budget=5)
        assert result.stats.steps > 0

    def test_budget_respected(self):
        space = SimpleSpace(n=30)
        engine = CuriosityEngine(space)
        result = engine.explore(space.get_state("s0"), budget=5)
        assert result.stats.steps <= 5

    def test_budget_zero(self):
        space = SimpleSpace(n=30)
        engine = CuriosityEngine(space)
        result = engine.explore(space.get_state("s0"), budget=0)
        assert result.stats.steps == 0

    def test_callback_called(self):
        space = SimpleSpace(n=30)
        engine = CuriosityEngine(space)
        calls = []
        def cb(step, disc):
            calls.append(step)
        engine.explore(space.get_state("s0"), budget=5, callback=cb)
        assert len(calls) > 0

    def test_discoveries_ranked(self):
        space = SimpleSpace(n=50)
        engine = CuriosityEngine(space, strategy="novelty")
        result = engine.explore(space.get_state("s0"), budget=20)
        scores = [d.score for d in result.discoveries]
        assert scores == sorted(scores, reverse=True)

    def test_explore_surprise(self):
        space = SimpleSpace(n=30)
        engine = CuriosityEngine(space, strategy="surprise")
        result = engine.explore(space.get_state("s0"), budget=10)
        assert isinstance(result, ExplorationResult)

    def test_explore_diversity(self):
        space = SimpleSpace(n=30)
        engine = CuriosityEngine(space, strategy="diversity")
        result = engine.explore(space.get_state("s0"), budget=10)
        assert isinstance(result, ExplorationResult)

    def test_explore_serendipity(self):
        space = SimpleSpace(n=30)
        profile = np.ones(8)
        engine = CuriosityEngine(space, strategy="serendipity", user_profile=profile)
        result = engine.explore(space.get_state("s0"), budget=10)
        assert isinstance(result, ExplorationResult)

    def test_explore_balanced(self):
        space = SimpleSpace(n=30)
        engine = CuriosityEngine(space, strategy="balanced")
        result = engine.explore(space.get_state("s0"), budget=10)
        assert isinstance(result, ExplorationResult)

    def test_get_discoveries(self):
        space = SimpleSpace(n=30)
        engine = CuriosityEngine(space)
        engine.explore(space.get_state("s0"), budget=10)
        disc = engine.get_discoveries(top_n=5)
        assert len(disc) <= 5

    def test_discovery_has_path(self):
        space = SimpleSpace(n=30)
        engine = CuriosityEngine(space)
        result = engine.explore(space.get_state("s0"), budget=10)
        if result.discoveries:
            assert len(result.discoveries[0].path) > 0

    def test_discovery_has_strategy_scores(self):
        space = SimpleSpace(n=30)
        engine = CuriosityEngine(space)
        result = engine.explore(space.get_state("s0"), budget=10)
        if result.discoveries:
            assert "novelty" in result.discoveries[0].strategy_scores

    def test_stats_populated(self):
        space = SimpleSpace(n=30)
        engine = CuriosityEngine(space)
        result = engine.explore(space.get_state("s0"), budget=10)
        assert result.stats.states_visited > 0
        assert result.stats.states_scored > 0

    def test_memory_populated(self):
        space = SimpleSpace(n=30)
        engine = CuriosityEngine(space)
        result = engine.explore(space.get_state("s0"), budget=10)
        assert len(result.memory) > 1

    def test_explore_coverage(self):
        space = SimpleSpace(n=20)
        engine = CuriosityEngine(space)
        result = engine.explore(space.get_state("s0"), budget=50)
        assert result.stats.coverage > 0

    def test_no_embedding_seed(self):
        space = SimpleSpace(n=20)
        seed = make_state("noEmb", dim=8)
        seed.embedding = None
        engine = CuriosityEngine(space)
        result = engine.explore(seed, budget=3)
        assert result.stats.steps > 0

    def test_score_method(self):
        space = SimpleSpace(n=20)
        engine = CuriosityEngine(space, strategy="novelty")
        s = space.get_state("s0")
        score = engine.score(s)
        assert isinstance(score, float)

    def test_custom_weights(self):
        space = SimpleSpace(n=20)
        engine = CuriosityEngine(space, strategy="balanced", score_weights={"novelty": 1.0})
        result = engine.explore(space.get_state("s0"), budget=5)
        assert isinstance(result, ExplorationResult)

    def test_discovery_threshold(self):
        space = SimpleSpace(n=30)
        engine = CuriosityEngine(space)
        result = engine.explore(space.get_state("s0"), budget=10, discovery_threshold=9999)
        assert len(result.discoveries) == 0

    def test_neighbors_per_step(self):
        space = SimpleSpace(n=30)
        engine = CuriosityEngine(space)
        result = engine.explore(space.get_state("s0"), budget=5, neighbors_per_step=3)
        assert result.stats.steps > 0
