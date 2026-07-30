"""Threshold and optional STATUS-metric behavior tests."""

from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("torch")

from duksan_lstm_ae.evaluation import calculate_metrics, select_threshold


def test_threshold_never_requires_test_scores(config):
    train_scores = np.array([1.0, 2.0, 3.0])
    validation_scores = np.array([2.0, 4.0, 6.0])
    threshold, details = select_threshold(
        train_scores,
        validation_scores,
        config,
    )
    assert threshold == np.percentile(
        validation_scores,
        config["threshold"]["percentile"],
    )
    assert details["source"] == "validation"


def test_label_metrics_are_not_reported_without_mapping():
    labels = np.array([-1, -1, -1])
    scores = np.array([0.1, 0.2, 0.3])
    predictions = np.array([0, 0, 1])
    metrics = calculate_metrics(
        labels,
        scores,
        predictions,
        labels_configured=False,
    )
    assert metrics["label_metrics_available"] is False
    assert "f1" not in metrics
