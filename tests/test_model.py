"""LSTM Autoencoder reconstruction-shape test."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from duksan_lstm_ae.model import build_model


def test_model_output_shape_matches_input(config):
    model = build_model(config, "cpu")
    inputs = torch.randn(
        4,
        config["window"]["sequence_length"],
        len(config["data"]["feature_cols"]),
    )
    outputs = model(inputs)
    assert outputs.shape == inputs.shape
