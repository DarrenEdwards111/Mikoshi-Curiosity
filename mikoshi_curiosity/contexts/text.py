"""Text corpus exploration context using TF-IDF."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from mikoshi_curiosity.space import State, StateSpace


class TextSpace(StateSpace):
    """Explore a collection of documents using TF-IDF embeddings.

    Parameters
    ----------
    documents : list[dict]
        Each dict must have 'id' and 'text'; 'metadata' is optional.
    max_features : int
        Maximum vocabulary size for TF-IDF.
    """

    def __init__(self, documents, max_features: int = 500):
        # Accept plain strings or dicts
        normalized = []
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                normalized.append({"id": str(i), "text": doc, "metadata": {}})
            else:
                normalized.append(doc)
        documents = normalized
        self.documents = {doc["id"]: doc for doc in documents}
        self._doc_ids = [doc["id"] for doc in documents]
        self.max_features = max_features
        self._build_tfidf(documents)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z]+", text.lower())

    def _build_tfidf(self, documents: List[Dict]):
        # Build vocabulary
        doc_freq: Counter = Counter()
        term_freqs: List[Counter] = []
        for doc in documents:
            tokens = self._tokenize(doc["text"])
            tf = Counter(tokens)
            term_freqs.append(tf)
            for term in set(tokens):
                doc_freq[term] += 1

        # Select top features by document frequency
        vocab = [w for w, _ in doc_freq.most_common(self.max_features)]
        self._vocab = {w: i for i, w in enumerate(vocab)}
        n_docs = len(documents)

        # Build TF-IDF matrix
        matrix = np.zeros((n_docs, len(vocab)), dtype=np.float64)
        for i, tf in enumerate(term_freqs):
            total = sum(tf.values()) or 1
            for term, idx in self._vocab.items():
                if term in tf:
                    tfidf = (tf[term] / total) * math.log((n_docs + 1) / (doc_freq[term] + 1))
                    matrix[i, idx] = tfidf

        # Normalise rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms
        self._id_to_idx = {did: i for i, did in enumerate(self._doc_ids)}

    def _make_state(self, doc_id: str) -> State:
        doc = self.documents[doc_id]
        idx = self._id_to_idx[doc_id]
        return State(
            id=doc_id,
            features={"text_preview": doc["text"][:200]},
            embedding=self._matrix[idx],
            metadata=doc.get("metadata", {}),
        )

    def get_neighbors(self, state: State, n: int = 10) -> List[State]:
        if state.embedding is None:
            state = self._make_state(state.id)
        sims = self._matrix @ state.embedding
        idx = self._id_to_idx.get(state.id, -1)
        if idx >= 0:
            sims[idx] = -np.inf
        top_n = np.argsort(sims)[-n:][::-1]
        return [self._make_state(self._doc_ids[i]) for i in top_n]

    def get_random(self, n: int = 10) -> List[State]:
        indices = np.random.choice(len(self._doc_ids), size=min(n, len(self._doc_ids)), replace=False)
        return [self._make_state(self._doc_ids[i]) for i in indices]

    def get_state(self, id: str) -> State:
        return self._make_state(id)

    def embed(self, state: State) -> np.ndarray:
        if state.id in self._id_to_idx:
            return self._matrix[self._id_to_idx[state.id]]
        # Embed unseen text
        tokens = self._tokenize(state.features.get("text", ""))
        tf = Counter(tokens)
        total = sum(tf.values()) or 1
        vec = np.zeros(len(self._vocab))
        for term, idx in self._vocab.items():
            if term in tf:
                vec[idx] = tf[term] / total
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def size(self) -> int:
        return len(self.documents)
