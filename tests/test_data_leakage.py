"""Time-split, purge-gap, train-scaler, and feature-order tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from duksan_lstm_ae.data import (
    chronological_frame_split,
    fit_train_scaler,
    transform_frame,
)


def _long_frame(config) -> pd.DataFrame:
    size = 200
    times = pd.date_range("2024-01-01", periods=size, freq="1min")
    frame = pd.DataFrame(
        {
            "GA_DT": times,
            "SEGMENT_ID": 1,
            "STATUS": pd.Series(["unknown"] * size, dtype="string"),
            "SOURCE_ROW_INDEX": np.arange(size),
            "_ROW_INDEX": np.arange(size),
        }
    )
    for index, feature in enumerate(config["data"]["feature_cols"]):
        values = np.arange(size, dtype=float) + index
        values[-30:] += 100_000
        frame[feature] = values
    return frame


def test_split_is_chronological_and_scaler_fits_train_only(config):
    frame = _long_frame(config)
    train, validation, test, info = chronological_frame_split(frame, config)
    scaler = fit_train_scaler(train, config)

    assert train["GA_DT"].max() < validation["GA_DT"].min()
    assert validation["GA_DT"].max() < test["GA_DT"].min()
    assert info["purged_after_train_rows"] == config["split"]["purge_gap_steps"]
    assert info["purged_after_validation_rows"] == config["split"]["purge_gap_steps"]
    assert scaler.fitted_split == "train"

    expected_train_median = train[config["data"]["feature_cols"][0]].median()
    assert scaler.estimator.center_[0] == expected_train_median


def test_feature_order_mismatch_fails(config):
    frame = _long_frame(config)
    train, _, _, _ = chronological_frame_split(frame, config)
    scaler = fit_train_scaler(train, config)
    scaler.feature_cols = list(reversed(scaler.feature_cols))
    with pytest.raises(ValueError, match="Feature order mismatch"):
        transform_frame(train, scaler, config)
