"""Tests for TextSpace context."""

import numpy as np
import pytest

from mikoshi_curiosity.space import State
from mikoshi_curiosity.contexts.text import TextSpace


DOCS = [
    {"id": "d1", "text": "The cat sat on the mat", "metadata": {"topic": "animals"}},
    {"id": "d2", "text": "The dog chased the cat around the garden", "metadata": {"topic": "animals"}},
    {"id": "d3", "text": "Quantum computing uses qubits for computation", "metadata": {"topic": "science"}},
    {"id": "d4", "text": "Machine learning algorithms process large datasets", "metadata": {"topic": "tech"}},
    {"id": "d5", "text": "The cat and dog played together in the sun", "metadata": {"topic": "animals"}},
    {"id": "d6", "text": "Neural networks learn representations from data", "metadata": {"topic": "tech"}},
    {"id": "d7", "text": "Cooking pasta requires boiling water and salt", "metadata": {"topic": "food"}},
    {"id": "d8", "text": "The stock market crashed due to economic uncertainty", "metadata": {"topic": "finance"}},
    {"id": "d9", "text": "Deep learning models use multiple layers of neurons", "metadata": {"topic": "tech"}},
    {"id": "d10", "text": "The garden was full of beautiful flowers and trees", "metadata": {"topic": "nature"}},
]


class TestTextInit:
    def test_basic_init(self):
        space = TextSpace(DOCS)
        assert space.size() == 10

    def test_custom_max_features(self):
        space = TextSpace(DOCS, max_features=10)
        assert len(space._vocab) <= 10

    def test_embeddings_normalized(self):
        space = TextSpace(DOCS)
        for i in range(len(DOCS)):
            norm = np.linalg.norm(space._matrix[i])
            if norm > 0:
                assert abs(norm - 1.0) < 1e-6


class TestTextNeighbors:
    def test_get_neighbors(self):
        space = TextSpace(DOCS)
        s = space.get_state("d1")
        nb = space.get_neighbors(s, n=3)
        assert len(nb) == 3

    def test_animal_docs_similar(self):
        space = TextSpace(DOCS)
        s = space.get_state("d1")  # cat on mat
        nb = space.get_neighbors(s, n=3)
        nb_ids = [n.id for n in nb]
        # d2 and d5 also mention cat/dog
        assert "d2" in nb_ids or "d5" in nb_ids

    def test_neighbors_exclude_self(self):
        space = TextSpace(DOCS)
        s = space.get_state("d1")
        nb = space.get_neighbors(s, n=5)
        assert all(n.id != "d1" for n in nb)

    def test_neighbors_have_embeddings(self):
        space = TextSpace(DOCS)
        s = space.get_state("d1")
        nb = space.get_neighbors(s, n=3)
        assert all(n.embedding is not None for n in nb)


class TestTextRandom:
    def test_get_random(self):
        space = TextSpace(DOCS)
        r = space.get_random(5)
        assert len(r) == 5

    def test_random_unique(self):
        space = TextSpace(DOCS)
        r = space.get_random(5)
        assert len(set(s.id for s in r)) == 5


class TestTextState:
    def test_get_state(self):
        space = TextSpace(DOCS)
        s = space.get_state("d3")
        assert s.id == "d3"
        assert "text_preview" in s.features

    def test_embed_known(self):
        space = TextSpace(DOCS)
        s = space.get_state("d1")
        emb = space.embed(s)
        assert isinstance(emb, np.ndarray)

    def test_embed_unknown(self):
        space = TextSpace(DOCS)
        s = State(id="new", features={"text": "cat dog animal"})
        emb = space.embed(s)
        assert isinstance(emb, np.ndarray)

    def test_distance(self):
        space = TextSpace(DOCS)
        s1 = space.get_state("d1")
        s2 = space.get_state("d3")
        d = space.distance(s1, s2)
        assert d >= 0

    def test_similar_docs_closer(self):
        space = TextSpace(DOCS)
        d1 = space.get_state("d1")  # cat
        d2 = space.get_state("d2")  # cat dog
        d3 = space.get_state("d3")  # quantum
        assert space.distance(d1, d2) < space.distance(d1, d3)

    def test_metadata_preserved(self):
        space = TextSpace(DOCS)
        s = space.get_state("d1")
        assert s.metadata.get("topic") == "animals"


class TestTextEdgeCases:
    def test_single_doc(self):
        space = TextSpace([{"id": "only", "text": "hello world"}])
        assert space.size() == 1

    def test_empty_text(self):
        docs = [{"id": "e1", "text": ""}, {"id": "e2", "text": "hello"}]
        space = TextSpace(docs)
        s = space.get_state("e1")
        assert s.embedding is not None

    def test_duplicate_words(self):
        docs = [{"id": "d1", "text": "cat cat cat cat cat"}]
        space = TextSpace(docs)
        assert space.size() == 1
