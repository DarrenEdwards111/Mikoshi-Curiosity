"""Tests for ExplorationMemory."""

import numpy as np
import pytest

from mikoshi_curiosity.memory import ExplorationMemory, MemoryEntry
from tests.helpers import make_state, make_states


class TestMemoryAdd:
    def test_add_state(self):
        mem = ExplorationMemory()
        s = make_state("s1")
        entry = mem.add(s, score=0.5)
        assert entry.state.id == "s1"
        assert len(mem) == 1

    def test_add_duplicate_increments_visit(self):
        mem = ExplorationMemory()
        s = make_state("s1")
        mem.add(s, score=0.5)
        mem.add(s, score=0.3)
        assert mem.states["s1"].visit_count == 2

    def test_add_duplicate_keeps_best_score(self):
        mem = ExplorationMemory()
        s = make_state("s1")
        mem.add(s, score=0.5)
        mem.add(s, score=0.8)
        assert mem.states["s1"].score == 0.8

    def test_add_duplicate_doesnt_raise_score(self):
        mem = ExplorationMemory()
        s = make_state("s1")
        mem.add(s, score=0.8)
        mem.add(s, score=0.3)
        assert mem.states["s1"].score == 0.8

    def test_add_with_source(self):
        mem = ExplorationMemory()
        s = make_state("s1")
        entry = mem.add(s, score=0.5, source="seed")
        assert entry.source == "seed"

    def test_add_many(self):
        mem = ExplorationMemory()
        for s in make_states(100):
            mem.add(s)
        assert len(mem) == 100

    def test_contains(self):
        mem = ExplorationMemory()
        s = make_state("s1")
        mem.add(s)
        assert "s1" in mem
        assert "s999" not in mem


class TestMemoryVisit:
    def test_has_visited_by_id(self):
        mem = ExplorationMemory()
        s = make_state("s1")
        mem.add(s)
        assert mem.has_visited(s)

    def test_not_visited(self):
        mem = ExplorationMemory()
        s = make_state("s1")
        assert not mem.has_visited(s)

    def test_has_visited_by_similarity(self):
        mem = ExplorationMemory()
        s = make_state("s1")
        mem.add(s)
        near = make_state("near")
        near.embedding = s.embedding + 1e-10
        assert mem.has_visited(near, threshold=0.5)

    def test_not_visited_distant(self):
        mem = ExplorationMemory()
        s = make_state("s1")
        mem.add(s)
        far = make_state("far")
        far.embedding = np.ones(8) * 1000
        assert not mem.has_visited(far, threshold=0.99)

    def test_visit_count_zero(self):
        mem = ExplorationMemory()
        s = make_state("s1")
        assert mem.visit_count(s) == 0

    def test_visit_count_after_adds(self):
        mem = ExplorationMemory()
        s = make_state("s1")
        mem.add(s)
        mem.add(s)
        mem.add(s)
        assert mem.visit_count(s) == 3


class TestMemoryNeighbors:
    def test_nearest_neighbors_empty(self):
        mem = ExplorationMemory()
        s = make_state("s1")
        assert mem.nearest_neighbors(s) == []

    def test_nearest_neighbors(self):
        mem = ExplorationMemory()
        states = make_states(20)
        for s in states:
            mem.add(s)
        query = states[0]
        nn = mem.nearest_neighbors(query, k=5)
        assert len(nn) == 5
        assert all(isinstance(e, MemoryEntry) for e in nn)

    def test_nearest_neighbors_excludes_self(self):
        mem = ExplorationMemory()
        states = make_states(10)
        for s in states:
            mem.add(s)
        nn = mem.nearest_neighbors(states[0], k=5)
        assert all(e.state.id != states[0].id for e in nn)

    def test_nearest_neighbors_k_larger_than_memory(self):
        mem = ExplorationMemory()
        states = make_states(3)
        for s in states:
            mem.add(s)
        nn = mem.nearest_neighbors(states[0], k=10)
        assert len(nn) == 2  # excluding self

    def test_no_embedding(self):
        mem = ExplorationMemory()
        s = make_state("s1")
        s.embedding = None
        assert mem.nearest_neighbors(s) == []

    def test_neighbors_sorted_by_distance(self):
        mem = ExplorationMemory()
        base = make_state("base")
        base.embedding = np.zeros(8)
        mem.add(base)
        for i in range(10):
            s = make_state(f"n{i}")
            s.embedding = np.ones(8) * (i + 1)
            mem.add(s)
        nn = mem.nearest_neighbors(base, k=5)
        dists = [float(np.linalg.norm(e.state.embedding - base.embedding)) for e in nn]
        assert dists == sorted(dists)


class TestMemoryFrontier:
    def test_frontier_empty(self):
        mem = ExplorationMemory()
        assert mem.get_frontier() == []

    def test_frontier_returns_frontier_states(self):
        mem = ExplorationMemory()
        for s in make_states(5):
            mem.add(s, score=np.random.random())
        frontier = mem.get_frontier()
        assert len(frontier) == 5
        assert all(e.is_frontier for e in frontier)

    def test_frontier_sorted_by_score(self):
        mem = ExplorationMemory()
        for i, s in enumerate(make_states(10)):
            mem.add(s, score=float(i))
        frontier = mem.get_frontier(n=5)
        scores = [e.score for e in frontier]
        assert scores == sorted(scores, reverse=True)

    def test_mark_explored(self):
        mem = ExplorationMemory()
        s = make_state("s1")
        mem.add(s)
        mem.mark_explored("s1")
        assert not mem.states["s1"].is_frontier

    def test_frontier_excludes_explored(self):
        mem = ExplorationMemory()
        states = make_states(5)
        for s in states:
            mem.add(s, score=1.0)
        mem.mark_explored("s0")
        frontier = mem.get_frontier()
        assert all(e.state.id != "s0" for e in frontier)


class TestMemoryPrune:
    def test_prune_under_capacity(self):
        mem = ExplorationMemory(max_size=100)
        for s in make_states(10):
            mem.add(s)
        mem.prune()
        assert len(mem) == 10

    def test_prune_over_capacity(self):
        mem = ExplorationMemory(max_size=5)
        for i, s in enumerate(make_states(10)):
            mem.add(s, score=float(i))
        assert len(mem) <= 5

    def test_prune_keeps_highest_scores(self):
        mem = ExplorationMemory(max_size=5)
        for i, s in enumerate(make_states(10)):
            mem.add(s, score=float(i))
        scores = [e.score for e in mem.states.values()]
        assert min(scores) >= 5.0

    def test_auto_prune_on_add(self):
        mem = ExplorationMemory(max_size=3)
        for i in range(10):
            s = make_state(f"s{i}")
            mem.add(s, score=float(i))
        assert len(mem) <= 3


class TestMemoryArchive:
    def test_archive_empty(self):
        mem = ExplorationMemory()
        assert mem.get_archive() == []

    def test_archive_all_states(self):
        mem = ExplorationMemory()
        states = make_states(10)
        for s in states:
            mem.add(s)
        archive = mem.get_archive()
        assert len(archive) == 10

    def test_len(self):
        mem = ExplorationMemory()
        assert len(mem) == 0
        for s in make_states(5):
            mem.add(s)
        assert len(mem) == 5
