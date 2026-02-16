"""Tests for PredictionModel."""

import numpy as np
import pytest

from mikoshi_curiosity.prediction import PredictionModel
from tests.helpers import make_state, make_states


class TestPredictionInit:
    def test_default_init(self):
        pm = PredictionModel()
        assert pm.embedding_dim == 128
        assert not pm.is_fitted

    def test_custom_dim(self):
        pm = PredictionModel(embedding_dim=16)
        assert pm.embedding_dim == 16

    def test_custom_lr(self):
        pm = PredictionModel(learning_rate=0.1)
        assert pm.learning_rate == 0.1


class TestPredict:
    def test_predict_shape(self):
        pm = PredictionModel(embedding_dim=8)
        s = make_state("x")
        pred = pm.predict_neighbors(s)
        assert pred.shape == (8,)

    def test_predict_no_embedding_raises(self):
        pm = PredictionModel()
        s = make_state("x")
        s.embedding = None
        with pytest.raises(ValueError):
            pm.predict_neighbors(s)

    def test_weights_auto_init(self):
        pm = PredictionModel()
        s = make_state("x", dim=16)
        pm.predict_neighbors(s)
        assert pm.weights is not None
        assert pm.weights.shape == (16, 16)

    def test_weights_reinit_on_dim_change(self):
        pm = PredictionModel(embedding_dim=8)
        s8 = make_state("a", dim=8)
        pm.predict_neighbors(s8)
        assert pm.weights.shape == (8, 8)
        s16 = make_state("b", dim=16)
        pm.predict_neighbors(s16)
        assert pm.weights.shape == (16, 16)


class TestPredictionError:
    def test_no_neighbors(self):
        pm = PredictionModel(embedding_dim=8)
        s = make_state("x")
        assert pm.prediction_error(s, []) == 0.0

    def test_positive_error(self):
        pm = PredictionModel(embedding_dim=8)
        s = make_state("x")
        neighbors = make_states(5)
        err = pm.prediction_error(s, neighbors)
        assert err >= 0

    def test_error_decreases_with_training(self):
        pm = PredictionModel(embedding_dim=8, learning_rate=0.05)
        states = make_states(10)
        s = states[0]
        nb = states[1:6]
        err_before = pm.prediction_error(s, nb)
        for _ in range(100):
            pm.update(s, nb)
        err_after = pm.prediction_error(s, nb)
        assert err_after < err_before

    def test_no_embedding_neighbors(self):
        pm = PredictionModel(embedding_dim=8)
        s = make_state("x")
        nb = make_state("y")
        nb.embedding = None
        assert pm.prediction_error(s, [nb]) == 0.0


class TestUpdate:
    def test_update_sets_fitted(self):
        pm = PredictionModel(embedding_dim=8)
        s = make_state("x")
        nb = make_states(3)
        pm.update(s, nb)
        assert pm.is_fitted

    def test_update_no_embedding(self):
        pm = PredictionModel(embedding_dim=8)
        s = make_state("x")
        s.embedding = None
        pm.update(s, make_states(3))
        assert not pm.is_fitted

    def test_update_empty_neighbors(self):
        pm = PredictionModel(embedding_dim=8)
        s = make_state("x")
        pm.update(s, [])
        assert not pm.is_fitted

    def test_multiple_updates(self):
        pm = PredictionModel(embedding_dim=8)
        states = make_states(10)
        for s in states:
            pm.update(s, states)
        assert pm.is_fitted

    def test_weights_change(self):
        pm = PredictionModel(embedding_dim=8)
        s = make_state("x")
        nb = make_states(3)
        pm.predict_neighbors(s)  # init weights
        w_before = pm.weights.copy()
        pm.update(s, nb)
        assert not np.allclose(pm.weights, w_before)
