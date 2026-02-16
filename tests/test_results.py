"""Tests for results module."""

import numpy as np
import pytest

from mikoshi_curiosity.space import State
from mikoshi_curiosity.memory import ExplorationMemory
from mikoshi_curiosity.results import Discovery, ExplorationResult, ExplorationStats
from tests.helpers import make_state


def make_discovery(id="d1", score=0.5, reason="test", **kwargs):
    return Discovery(state=make_state(id), score=score, reason=reason, **kwargs)


class TestDiscovery:
    def test_basic(self):
        d = make_discovery()
        assert d.score == 0.5
        assert d.reason == "test"

    def test_repr(self):
        d = make_discovery()
        assert "d1" in repr(d)
        assert "0.500" in repr(d)

    def test_strategy_scores(self):
        d = make_discovery(strategy_scores={"novelty": 0.9, "surprise": 0.3})
        assert d.strategy_scores["novelty"] == 0.9

    def test_path(self):
        path = [make_state("a"), make_state("b")]
        d = make_discovery(path=path)
        assert len(d.path) == 2

    def test_default_fields(self):
        d = Discovery(state=make_state("x"), score=0.0)
        assert d.reason == ""
        assert d.strategy_scores == {}
        assert d.path == []


class TestExplorationStats:
    def test_defaults(self):
        s = ExplorationStats()
        assert s.steps == 0
        assert s.coverage == 0.0

    def test_custom(self):
        s = ExplorationStats(steps=10, states_visited=50, coverage=0.5)
        assert s.steps == 10
        assert s.coverage == 0.5


class TestExplorationResult:
    def test_empty(self):
        r = ExplorationResult()
        assert len(r.discoveries) == 0
        assert r.stats.steps == 0

    def test_top(self):
        discs = [make_discovery(f"d{i}", score=float(i)) for i in range(10)]
        r = ExplorationResult(discoveries=discs)
        top = r.top(3)
        assert len(top) == 3
        assert top[0].score == 9.0

    def test_top_more_than_available(self):
        discs = [make_discovery(f"d{i}", score=float(i)) for i in range(3)]
        r = ExplorationResult(discoveries=discs)
        top = r.top(10)
        assert len(top) == 3

    def test_by_strategy(self):
        discs = [
            make_discovery("a", score=1, strategy_scores={"novelty": 0.1}),
            make_discovery("b", score=0.5, strategy_scores={"novelty": 0.9}),
        ]
        r = ExplorationResult(discoveries=discs)
        by_nov = r.by_strategy("novelty")
        assert by_nov[0].state.id == "b"

    def test_by_strategy_missing_key(self):
        discs = [make_discovery("a", strategy_scores={})]
        r = ExplorationResult(discoveries=discs)
        result = r.by_strategy("novelty")
        assert len(result) == 1

    def test_summary(self):
        discs = [make_discovery(f"d{i}", score=float(i), reason=f"r{i}") for i in range(3)]
        stats = ExplorationStats(steps=10, states_visited=50)
        r = ExplorationResult(discoveries=discs, stats=stats)
        s = r.summary()
        assert "10 steps" in s
        assert "50 states" in s

    def test_summary_empty(self):
        r = ExplorationResult()
        s = r.summary()
        assert "0 steps" in s

    def test_repr(self):
        r = ExplorationResult(
            discoveries=[make_discovery()],
            stats=ExplorationStats(steps=5),
        )
        assert "1" in repr(r)
        assert "5" in repr(r)

    def test_with_memory(self):
        mem = ExplorationMemory()
        mem.add(make_state("s1"))
        r = ExplorationResult(memory=mem)
        assert r.memory is not None
        assert len(r.memory) == 1


class TestToDataframe:
    def test_to_dataframe(self):
        pd = pytest.importorskip("pandas")
        discs = [
            make_discovery("a", score=1.0, strategy_scores={"novelty": 0.9}),
            make_discovery("b", score=0.5, strategy_scores={"novelty": 0.3}),
        ]
        r = ExplorationResult(discoveries=discs)
        df = r.to_dataframe()
        assert len(df) == 2
        assert "score" in df.columns
        assert "score_novelty" in df.columns

    def test_to_dataframe_empty(self):
        pd = pytest.importorskip("pandas")
        r = ExplorationResult()
        df = r.to_dataframe()
        assert len(df) == 0

    def test_to_dataframe_features(self):
        pd = pytest.importorskip("pandas")
        s = make_state("x", features={"color": "red"})
        d = Discovery(state=s, score=1.0)
        r = ExplorationResult(discoveries=[d])
        df = r.to_dataframe()
        assert "color" in df.columns
