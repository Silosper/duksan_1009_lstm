"""Window shape, continuity, metadata, and STATUS isolation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from duksan_lstm_ae.data import make_windows, validate_window_bundle


def _frame(config) -> pd.DataFrame:
    feature_cols = config["data"]["feature_cols"]
    first = pd.date_range("2024-01-01", periods=6, freq="1min")
    second = pd.date_range("2024-01-01 01:00", periods=5, freq="1min")
    times = first.append(second)
    frame = pd.DataFrame(
        {
            "GA_DT": times,
            "SEGMENT_ID": [1] * len(first) + [2] * len(second),
            "STATUS": ["unmapped"] * len(times),
            "SOURCE_ROW_INDEX": np.arange(len(times)),
            "_ROW_INDEX": np.arange(len(times)),
            "_SPLIT": "train",
        }
    )
    for index, feature in enumerate(feature_cols):
        frame[feature] = np.arange(len(times), dtype=float) + index
        frame[f"INTERPOLATED_{feature}"] = 0
    return frame


def test_windows_never_cross_segment_or_long_gap(config):
    bundle = make_windows(
        _frame(config),
        config,
        normal_only=False,
        max_windows=None,
        split_name="train",
    )
    validate_window_bundle(bundle, config)

    assert bundle.X.shape == (7, 3, 5)
    assert np.isfinite(bundle.X).all()
    assert "STATUS" not in bundle.feature_cols
    assert set(bundle.segment_ids) == {1, 2}
    durations = pd.to_datetime(bundle.end_times) - pd.to_datetime(
        bundle.start_times
    )
    assert (durations == pd.Timedelta(minutes=2)).all()


def test_nonfinite_window_is_excluded(config):
    frame = _frame(config)
    frame.loc[2, "CURRENT1"] = np.nan
    bundle = make_windows(
        frame,
        config,
        normal_only=False,
        max_windows=None,
        split_name="train",
    )
    assert np.isfinite(bundle.X).all()
    assert bundle.exclusion_counts["windows_nonfinite"] > 0
