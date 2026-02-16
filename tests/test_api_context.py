"""Tests for APISpace context."""

import numpy as np
import pytest

from mikoshi_curiosity.space import State
from mikoshi_curiosity.contexts.api import APISpace


def mock_fetch(query):
    return [
        {"id": f"item{i}", "name": f"Item {i}", "score": i * 0.1}
        for i in range(5)
    ]


class TestAPIInit:
    def test_basic(self):
        space = APISpace(fetch_fn=mock_fetch)
        assert space.size() == 0

    def test_with_embed_fn(self):
        space = APISpace(fetch_fn=mock_fetch, embed_fn=lambda x: np.ones(8))
        r = space.get_random(3)
        assert np.allclose(r[0].embedding, np.ones(8))


class TestAPINeighbors:
    def test_get_neighbors(self):
        space = APISpace(fetch_fn=mock_fetch)
        s = State(id="item0", features={"name": "Item 0"})
        nb = space.get_neighbors(s, n=3)
        assert len(nb) == 3

    def test_neighbors_cached(self):
        space = APISpace(fetch_fn=mock_fetch)
        s = State(id="item0", features={"name": "Item 0"})
        space.get_neighbors(s, n=5)
        assert space.size() == 5


class TestAPIRandom:
    def test_get_random(self):
        space = APISpace(fetch_fn=mock_fetch)
        r = space.get_random(3)
        assert len(r) == 3


class TestAPIState:
    def test_get_cached(self):
        space = APISpace(fetch_fn=mock_fetch)
        space.get_random(5)
        s = space.get_state("item0")
        assert s.id == "item0"

    def test_get_uncached_raises(self):
        space = APISpace(fetch_fn=mock_fetch)
        with pytest.raises(KeyError):
            space.get_state("nonexistent")

    def test_embed(self):
        space = APISpace(fetch_fn=mock_fetch, embedding_dim=16)
        s = State(id="x", features={"a": 1})
        emb = space.embed(s)
        assert emb.shape == (16,)
