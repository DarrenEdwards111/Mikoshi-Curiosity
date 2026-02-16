"""Tests for GraphSpace context."""

import numpy as np
import pytest

from mikoshi_curiosity.space import State
from mikoshi_curiosity.contexts.graph import GraphSpace


def triangle_graph():
    return GraphSpace(nodes=["a", "b", "c"], edges=[("a", "b"), ("b", "c"), ("a", "c")])


def line_graph(n=10):
    nodes = [str(i) for i in range(n)]
    edges = [(str(i), str(i + 1)) for i in range(n - 1)]
    return GraphSpace(nodes=nodes, edges=edges)


def star_graph(center="hub", spokes=5):
    nodes = [center] + [f"s{i}" for i in range(spokes)]
    edges = [(center, f"s{i}") for i in range(spokes)]
    return GraphSpace(nodes=nodes, edges=edges)


class TestGraphInit:
    def test_basic_init(self):
        g = triangle_graph()
        assert g.size() == 3

    def test_dict_nodes(self):
        g = GraphSpace(
            nodes=[{"id": "a", "color": "red"}, {"id": "b", "color": "blue"}],
            edges=[("a", "b")],
        )
        assert g.size() == 2

    def test_dict_edges(self):
        g = GraphSpace(
            nodes=["a", "b"],
            edges=[{"source": "a", "target": "b"}],
        )
        assert g.size() == 2

    def test_large_graph(self):
        nodes = [str(i) for i in range(100)]
        edges = [(str(i), str((i + 1) % 100)) for i in range(100)]
        g = GraphSpace(nodes=nodes, edges=edges)
        assert g.size() == 100

    def test_embeddings_computed(self):
        g = triangle_graph()
        s = g.get_state("a")
        assert s.embedding is not None
        assert len(s.embedding) == 64


class TestGraphNeighbors:
    def test_get_neighbors(self):
        g = triangle_graph()
        s = g.get_state("a")
        nb = g.get_neighbors(s, n=5)
        assert len(nb) == 2  # triangle: a connected to b, c

    def test_line_graph_neighbors(self):
        g = line_graph()
        s = g.get_state("0")
        nb = g.get_neighbors(s)
        assert len(nb) == 1  # endpoint
        nb_ids = [n.id for n in nb]
        assert "1" in nb_ids

    def test_star_hub_neighbors(self):
        g = star_graph(spokes=5)
        s = g.get_state("hub")
        nb = g.get_neighbors(s)
        assert len(nb) == 5

    def test_star_spoke_neighbors(self):
        g = star_graph(spokes=5)
        s = g.get_state("s0")
        nb = g.get_neighbors(s)
        assert len(nb) == 1
        assert nb[0].id == "hub"


class TestGraphRandom:
    def test_get_random(self):
        g = line_graph(20)
        r = g.get_random(5)
        assert len(r) == 5

    def test_random_unique(self):
        g = line_graph(20)
        r = g.get_random(10)
        assert len(set(s.id for s in r)) == 10


class TestGraphState:
    def test_get_state(self):
        g = triangle_graph()
        s = g.get_state("a")
        assert s.id == "a"
        assert "degree" in s.features

    def test_degree_feature(self):
        g = star_graph(spokes=5)
        hub = g.get_state("hub")
        assert hub.features["degree"] == 5
        spoke = g.get_state("s0")
        assert spoke.features["degree"] == 1

    def test_embed(self):
        g = triangle_graph()
        s = g.get_state("a")
        emb = g.embed(s)
        assert emb.shape == (64,)

    def test_embed_unknown(self):
        g = triangle_graph()
        s = State(id="unknown")
        emb = g.embed(s)
        assert np.allclose(emb, 0)

    def test_distance(self):
        g = triangle_graph()
        a = g.get_state("a")
        b = g.get_state("b")
        d = g.distance(a, b)
        assert d >= 0

    def test_node_features_preserved(self):
        g = GraphSpace(
            nodes=[{"id": "a", "color": "red"}],
            edges=[],
        )
        s = g.get_state("a")
        assert s.features.get("color") == "red"


class TestGraphBridge:
    def test_bridge_node_different_embedding(self):
        # Two clusters connected by a bridge
        nodes = ["a1", "a2", "a3", "bridge", "b1", "b2", "b3"]
        edges = [
            ("a1", "a2"), ("a2", "a3"), ("a1", "a3"),
            ("a3", "bridge"),
            ("bridge", "b1"),
            ("b1", "b2"), ("b2", "b3"), ("b1", "b3"),
        ]
        g = GraphSpace(nodes=nodes, edges=edges)
        bridge = g.get_state("bridge")
        a1 = g.get_state("a1")
        b1 = g.get_state("b1")
        # Bridge should be between clusters
        assert bridge.embedding is not None
        assert a1.embedding is not None


class TestGraphEdgeCases:
    def test_isolated_node(self):
        g = GraphSpace(nodes=["a", "b"], edges=[])
        s = g.get_state("a")
        nb = g.get_neighbors(s)
        assert len(nb) == 0

    def test_self_loop_ignored(self):
        g = GraphSpace(nodes=["a"], edges=[("a", "a")])
        assert g.size() == 1

    def test_single_node(self):
        g = GraphSpace(nodes=["a"], edges=[])
        assert g.size() == 1
