"""Synthetic tests for duplicate, short-gap, and long-gap behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from duksan_lstm_ae.preprocessing import (
    preprocess_frame,
    resolve_duplicate_timestamps,
)


FEATURES = ["CURRENT1", "VOLTAGE", "TEMP_CUR", "ANALOGUE", "GROUND"]


def _raw_with_gaps() -> pd.DataFrame:
    times = pd.to_datetime(
        [
            "2024-01-01 00:00",
            "2024-01-01 00:01",
            "2024-01-01 00:01",
            "2024-01-01 00:02",
            "2024-01-01 00:04",
            "2024-01-01 00:10",
            "2024-01-01 00:11",
            "2024-01-01 00:12",
        ]
    )
    frame = pd.DataFrame({"GA_DT": times, "STATUS": [" A "] * len(times)})
    for offset, feature in enumerate(FEATURES):
        frame[feature] = np.arange(len(times), dtype=float) + offset
    return frame


def test_short_gap_is_interpolated_and_long_gap_splits_segment(config):
    processed, report, segments = preprocess_frame(_raw_with_gaps(), config)

    short_gap = processed.loc[
        processed["GA_DT"].eq(pd.Timestamp("2024-01-01 00:03"))
    ]
    assert len(short_gap) == 1
    assert short_gap["INTERPOLATED_CURRENT1"].iloc[0] == 1

    long_gap_times = pd.date_range(
        "2024-01-01 00:05",
        "2024-01-01 00:09",
        freq="1min",
    )
    assert not processed["GA_DT"].isin(long_gap_times).any()
    assert processed["SEGMENT_ID"].nunique() == 2
    assert report["short_missing_timestamp_runs"] == 1
    assert report["long_missing_timestamp_runs"] == 1
    assert report["scaling_applied_during_preprocessing"] is False
    assert len(segments) == 2
    assert processed["STATUS"].dropna().eq("A").all()
    assert "STATUS_ORIGINAL" in processed


def test_duplicate_policy_first_last_and_mean(config):
    raw = _raw_with_gaps().iloc[:4].copy()
    raw["STATUS_ORIGINAL"] = raw["STATUS"]
    raw["SOURCE_ROW_INDEX"] = np.arange(len(raw))
    raw["STATUS"] = raw["STATUS"].astype("string").str.strip()
    duplicate_time = pd.Timestamp("2024-01-01 00:01")
    duplicate_values = raw.loc[raw["GA_DT"].eq(duplicate_time), "CURRENT1"]

    means = resolve_duplicate_timestamps(
        raw,
        time_col="GA_DT",
        feature_cols=FEATURES,
        status_col="STATUS",
        status_original_col="STATUS_ORIGINAL",
        source_index_col="SOURCE_ROW_INDEX",
        policy="mean",
    )
    first = resolve_duplicate_timestamps(
        raw,
        time_col="GA_DT",
        feature_cols=FEATURES,
        status_col="STATUS",
        status_original_col="STATUS_ORIGINAL",
        source_index_col="SOURCE_ROW_INDEX",
        policy="first",
    )
    last = resolve_duplicate_timestamps(
        raw,
        time_col="GA_DT",
        feature_cols=FEATURES,
        status_col="STATUS",
        status_original_col="STATUS_ORIGINAL",
        source_index_col="SOURCE_ROW_INDEX",
        policy="last",
    )

    assert means.loc[means["GA_DT"].eq(duplicate_time), "CURRENT1"].iloc[0] == (
        duplicate_values.mean()
    )
    assert first.loc[first["GA_DT"].eq(duplicate_time), "CURRENT1"].iloc[0] == (
        duplicate_values.iloc[0]
    )
    assert last.loc[last["GA_DT"].eq(duplicate_time), "CURRENT1"].iloc[0] == (
        duplicate_values.iloc[-1]
    )
