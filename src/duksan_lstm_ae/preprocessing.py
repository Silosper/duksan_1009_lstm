"""Time-axis cleaning, bounded interpolation, segmentation, and reporting."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import frequency_delta, resolve_path


def preprocess_csv(
    config: dict[str, Any],
    *,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Preprocess the configured raw CSV without fitting a model scaler."""
    source = resolve_path(config, input_path or config["data"]["raw_path"])
    destination = resolve_path(
        config,
        output_path or config["data"]["processed_path"],
    )
    raw = pd.read_csv(source, low_memory=False)
    processed, report, segment_summary = preprocess_frame(raw, config)

    destination.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(destination, index=False, encoding="utf-8-sig")

    report_path = resolve_path(
        config,
        config["data"]["preprocessing_report_path"],
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    segment_path = resolve_path(config, config["data"]["segment_summary_path"])
    segment_path.parent.mkdir(parents=True, exist_ok=True)
    segment_summary.to_csv(segment_path, index=False, encoding="utf-8-sig")

    print(f"Processed CSV: {destination}")
    print(f"Preprocessing report: {report_path}")
    print(f"Segment summary: {segment_path}")
    return processed, report, segment_summary


def preprocess_frame(
    raw_frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Return a model-ready frame, JSON-safe report, and segment summary."""
    data_cfg = config["data"]
    missing_cfg = config["missing"]
    time_col = data_cfg["time_col"]
    status_col = data_cfg["status_col"]
    status_original_col = data_cfg["status_original_col"]
    source_index_col = data_cfg["source_index_col"]
    segment_col = data_cfg["segment_col"]
    feature_cols = list(data_cfg["feature_cols"])
    required = [time_col, *feature_cols]
    missing_columns = [column for column in required if column not in raw_frame]
    if missing_columns:
        raise ValueError(f"Required raw columns are missing: {missing_columns}")

    frame = raw_frame.copy(deep=True).reset_index(drop=True)
    raw_rows = len(frame)
    frame[source_index_col] = np.arange(raw_rows, dtype=np.int64)
    if status_col not in frame:
        warnings.warn(
            f"{status_col} is absent. Label-based filtering and metrics are disabled.",
            stacklevel=2,
        )
        frame[status_col] = pd.NA
    frame[status_original_col] = frame[status_col].copy()
    frame[status_col] = normalize_status(frame[status_col])
    input_status_counts = {
        str(key): int(value)
        for key, value in frame[status_col]
        .value_counts(dropna=False)
        .items()
    }

    parsed_time = pd.to_datetime(frame[time_col], errors="coerce")
    invalid_time_rows = int(parsed_time.isna().sum())
    frame[time_col] = parsed_time
    frame = frame.dropna(subset=[time_col]).copy()

    for column in feature_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    infinite_values = int(
        np.isinf(frame[feature_cols].to_numpy(dtype=float)).sum()
    )
    frame[feature_cols] = frame[feature_cols].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    frame = frame[
        [
            time_col,
            *feature_cols,
            status_col,
            status_original_col,
            source_index_col,
        ]
    ].sort_values(time_col).reset_index(drop=True)

    duplicate_excess = int(frame.duplicated(time_col, keep="first").sum())
    frame = resolve_duplicate_timestamps(
        frame,
        time_col=time_col,
        feature_cols=feature_cols,
        status_col=status_col,
        status_original_col=status_original_col,
        source_index_col=source_index_col,
        policy=data_cfg["duplicate_policy"],
    )
    interval_report = analyze_time_intervals(frame[time_col], config)

    aligned = align_to_frequency(frame, config)
    missing_before = {
        column: int(aligned[column].isna().sum()) for column in feature_cols
    }
    time_gap_stats = summarize_missing_timestamp_runs(
        aligned["OBSERVED_COUNT"].eq(0),
        int(missing_cfg["max_interpolation_gap_steps"]),
    )
    sensor_gap_stats = summarize_missing_timestamp_runs(
        aligned[feature_cols].isna().any(axis=1),
        int(missing_cfg["max_interpolation_gap_steps"]),
    )

    interpolated, interpolation_masks, long_missing_mask = (
        interpolate_short_missing_runs(aligned, config)
    )
    for column, mask in interpolation_masks.items():
        if bool(missing_cfg["add_interpolation_mask"]):
            prefix = missing_cfg["interpolation_mask_prefix"]
            interpolated[f"{prefix}{column}"] = mask.astype(np.int8)

    remaining_before_drop = {
        column: int(interpolated[column].isna().sum())
        for column in feature_cols
    }
    model_ready_mask = np.isfinite(
        interpolated[feature_cols].to_numpy(dtype=float)
    ).all(axis=1)
    excluded_missing_rows = int((~model_ready_mask).sum())
    processed = interpolated.loc[model_ready_mask].copy()
    if processed.empty:
        raise ValueError(
            "No model-ready rows remain after bounded interpolation. "
            "Review source data, frequency, and missing-gap settings."
        )
    processed = assign_segment_ids(processed, config)
    processed = processed.reset_index().rename(columns={"index": time_col})
    processed[source_index_col] = (
        processed[source_index_col].fillna(-1).astype(np.int64)
    )

    segment_summary = summarize_segments(processed, config)
    sequence_length = int(config["window"]["sequence_length"])
    excluded_short_segments = int(
        (segment_summary["rows"] < sequence_length).sum()
    )
    interpolation_total = int(
        sum(mask.sum() for mask in interpolation_masks.values())
    )
    status_counts = {
        str(key): int(value)
        for key, value in processed[status_col]
        .value_counts(dropna=False)
        .items()
    }
    report = {
        "raw_rows": raw_rows,
        "invalid_time_rows": invalid_time_rows,
        "duplicate_timestamp_excess_rows": duplicate_excess,
        "duplicate_policy": data_cfg["duplicate_policy"],
        "rows_after_time_and_duplicate_cleaning": len(frame),
        "configured_frequency": data_cfg["frequency"],
        "interval_analysis": interval_report,
        "short_missing_timestamp_runs": time_gap_stats["short_runs"],
        "long_missing_timestamp_runs": time_gap_stats["long_runs"],
        "short_sensor_missing_runs": sensor_gap_stats["short_runs"],
        "long_sensor_missing_runs": sensor_gap_stats["long_runs"],
        "long_missing_rows": int(long_missing_mask.sum()),
        "longest_missing_timestamp_run_steps": time_gap_stats["longest_run_steps"],
        "missing_values_before_interpolation": missing_before,
        "interpolated_values": interpolation_total,
        "missing_values_after_interpolation_before_drop": remaining_before_drop,
        "infinite_values_converted_to_missing": infinite_values,
        "rows_excluded_for_remaining_missing_or_inf": excluded_missing_rows,
        "processed_rows": len(processed),
        "segment_count": int(processed[segment_col].nunique()),
        "segments_shorter_than_sequence_length": excluded_short_segments,
        "feature_cols": feature_cols,
        "input_status_value_counts": input_status_counts,
        "status_value_counts": status_counts,
        "scaling_applied_during_preprocessing": False,
    }
    _print_preprocessing_report(report)
    return processed, report, segment_summary


def normalize_status(series: pd.Series) -> pd.Series:
    """Normalize representation only; never assign semantic meaning."""
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""))


def resolve_duplicate_timestamps(
    frame: pd.DataFrame,
    *,
    time_col: str,
    feature_cols: list[str],
    status_col: str,
    status_original_col: str,
    source_index_col: str,
    policy: str,
) -> pd.DataFrame:
    """Resolve exact duplicate timestamps with the configured policy."""
    policy = str(policy).lower()
    ordered = frame.sort_values(time_col)
    if policy in {"first", "last"}:
        return (
            ordered.drop_duplicates(time_col, keep=policy)
            .sort_values(time_col)
            .reset_index(drop=True)
        )
    if policy != "mean":
        raise ValueError(f"Unsupported duplicate policy: {policy}")

    aggregations: dict[str, Any] = {
        **{column: "mean" for column in feature_cols},
        status_col: _last_non_missing,
        status_original_col: _last_non_missing,
        source_index_col: "min",
    }
    return (
        ordered.groupby(time_col, as_index=False, sort=True)
        .agg(aggregations)
        .reset_index(drop=True)
    )


def analyze_time_intervals(
    timestamps: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Summarize observed intervals before frequency alignment."""
    diffs = timestamps.sort_values().diff().dropna()
    configured = frequency_delta(config)
    if diffs.empty:
        return {
            "summary_seconds": {},
            "most_frequent_interval": None,
            "configured_frequency_mismatch_count": 0,
            "longest_observed_gap": None,
            "gap_buckets": {},
        }

    most_frequent = diffs.value_counts().index[0]
    mismatch_count = int(diffs.ne(configured).sum())
    if most_frequent != configured:
        warnings.warn(
            "Configured frequency differs from the most frequent observed "
            f"interval: configured={configured}, observed_mode={most_frequent}.",
            stacklevel=2,
        )

    gap_steps = np.floor(
        diffs.loc[diffs > configured] / configured
    ).astype(int) - 1
    bucket_edges = sorted(
        {int(value) for value in config["missing"]["gap_bucket_steps"] if int(value) > 0}
    )
    gap_buckets: dict[str, int] = {}
    lower = 1
    for upper in bucket_edges:
        gap_buckets[f"{lower}-{upper}_steps"] = int(
            ((gap_steps >= lower) & (gap_steps <= upper)).sum()
        )
        lower = upper + 1
    gap_buckets[f"{lower}+_steps"] = int((gap_steps >= lower).sum())

    summary = diffs.dt.total_seconds().describe()
    return {
        "summary_seconds": {
            str(key): _json_number(value) for key, value in summary.items()
        },
        "most_frequent_interval": str(most_frequent),
        "configured_frequency_mismatch_count": mismatch_count,
        "longest_observed_gap": str(diffs.max()),
        "gap_buckets": gap_buckets,
    }


def align_to_frequency(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Resample to the configured fixed frequency and expose missing timestamps."""
    data_cfg = config["data"]
    time_col = data_cfg["time_col"]
    status_col = data_cfg["status_col"]
    status_original_col = data_cfg["status_original_col"]
    source_index_col = data_cfg["source_index_col"]
    feature_cols = list(data_cfg["feature_cols"])
    frequency = data_cfg["frequency"]
    indexed = frame.set_index(time_col)

    sensor_values = indexed[feature_cols].resample(frequency).mean()
    status = indexed[status_col].resample(frequency).last()
    original_status = indexed[status_original_col].resample(frequency).last()
    source_index = indexed[source_index_col].resample(frequency).first()
    observed_count = indexed.resample(frequency).size().rename("OBSERVED_COUNT")
    aligned = pd.concat(
        [
            sensor_values,
            status.rename(status_col),
            original_status.rename(status_original_col),
            source_index.rename(source_index_col),
            observed_count,
        ],
        axis=1,
    )
    aligned.index.name = time_col
    aligned[status_col] = normalize_status(aligned[status_col])
    aligned["OBSERVED_COUNT"] = aligned["OBSERVED_COUNT"].astype(np.int64)
    return aligned


def interpolate_short_missing_runs(
    aligned: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.Series]:
    """Interpolate only bounded runs measured in sampling-frequency steps."""
    feature_cols = list(config["data"]["feature_cols"])
    missing_cfg = config["missing"]
    max_steps = int(missing_cfg["max_interpolation_gap_steps"])
    result = aligned.copy(deep=True)
    masks = {
        column: pd.Series(False, index=result.index) for column in feature_cols
    }
    if max_steps == 0:
        return result, masks, result[feature_cols].isna().any(axis=1)

    any_missing = result[feature_cols].isna().any(axis=1)
    run_id = any_missing.ne(any_missing.shift(fill_value=False)).cumsum()
    run_length = any_missing.groupby(run_id).transform("sum")
    long_missing_mask = any_missing & run_length.gt(max_steps)

    # 각 long-gap 행을 경계로 사용하여 보간이 공백 반대편까지 도달하지 않게 함
    block_id = long_missing_mask.astype(np.int64).cumsum()
    valid_positions = ~long_missing_mask
    for _, index in result.loc[valid_positions].groupby(
        block_id.loc[valid_positions],
        sort=False,
    ).groups.items():
        block = result.loc[index, feature_cols]
        for column in feature_cols:
            before_missing = block[column].isna()
            if missing_cfg["interpolation_method"] == "time":
                candidate = block[column].interpolate(
                    method="time",
                    limit_area="inside",
                )
            else:
                candidate = block[column].interpolate(
                    method="linear",
                    limit_area="inside",
                )
            filled = before_missing & candidate.notna()
            result.loc[index, column] = candidate
            masks[column].loc[index] = filled
    return result, masks, long_missing_mask


def assign_segment_ids(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Assign a new segment whenever the allowed time interval is exceeded."""
    ordered = frame.sort_index().copy()
    allowed = frequency_delta(config) * int(
        config["missing"]["max_segment_gap_steps"]
    )
    time_diff = ordered.index.to_series().diff()
    segment_break = time_diff.isna() | time_diff.gt(allowed)
    ordered[config["data"]["segment_col"]] = (
        segment_break.cumsum().astype(np.int64)
    )
    return ordered


def summarize_segments(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Summarize continuity and window capacity for each segment."""
    data_cfg = config["data"]
    segment_col = data_cfg["segment_col"]
    time_col = data_cfg["time_col"]
    feature_cols = list(data_cfg["feature_cols"])
    sequence_length = int(config["window"]["sequence_length"])
    stride = int(config["window"]["stride"])
    rows: list[dict[str, Any]] = []
    for segment_id, group in frame.groupby(segment_col, sort=True):
        start = group[time_col].min()
        end = group[time_col].max()
        row_count = len(group)
        possible_windows = (
            0
            if row_count < sequence_length
            else 1 + (row_count - sequence_length) // stride
        )
        rows.append(
            {
                "segment_id": int(segment_id),
                "start": start,
                "end": end,
                "rows": row_count,
                "duration": str(end - start),
                "remaining_sensor_missing_values": int(
                    group[feature_cols].isna().sum().sum()
                ),
                "remaining_status_missing_values": int(
                    group[data_cfg["status_col"]].isna().sum()
                ),
                "possible_windows": possible_windows,
                "excluded_too_short": row_count < sequence_length,
            }
        )
    return pd.DataFrame(rows)


def summarize_missing_timestamp_runs(
    missing_timestamp_mask: pd.Series,
    max_short_steps: int,
) -> dict[str, int]:
    if not missing_timestamp_mask.any():
        return {"short_runs": 0, "long_runs": 0, "longest_run_steps": 0}
    run_id = missing_timestamp_mask.ne(
        missing_timestamp_mask.shift(fill_value=False)
    ).cumsum()
    lengths = missing_timestamp_mask.groupby(run_id).sum()
    lengths = lengths[lengths > 0].astype(int)
    return {
        "short_runs": int((lengths <= max_short_steps).sum()),
        "long_runs": int((lengths > max_short_steps).sum()),
        "longest_run_steps": int(lengths.max()),
    }


def _last_non_missing(series: pd.Series) -> Any:
    values = series.dropna()
    return values.iloc[-1] if len(values) else pd.NA


def _json_number(value: Any) -> int | float | None:
    if pd.isna(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def _print_preprocessing_report(report: dict[str, Any]) -> None:
    print(f"Raw rows: {report['raw_rows']:,}")
    print(f"Invalid timestamps: {report['invalid_time_rows']:,}")
    print(
        "Duplicate timestamp excess rows: "
        f"{report['duplicate_timestamp_excess_rows']:,} "
        f"(policy={report['duplicate_policy']})"
    )
    print(f"Configured frequency: {report['configured_frequency']}")
    interval = report["interval_analysis"]
    print(f"Observed interval mode: {interval['most_frequent_interval']}")
    print(
        "Configured-frequency mismatch count: "
        f"{interval['configured_frequency_mismatch_count']:,}"
    )
    print(f"Longest observed gap: {interval['longest_observed_gap']}")
    print(f"Interval summary (seconds): {interval['summary_seconds']}")
    print(f"Gap buckets: {interval['gap_buckets']}")
    print(
        "Short/long missing timestamp runs: "
        f"{report['short_missing_timestamp_runs']:,}/"
        f"{report['long_missing_timestamp_runs']:,}"
    )
    print(
        "Short/long sensor-missing runs: "
        f"{report['short_sensor_missing_runs']:,}/"
        f"{report['long_sensor_missing_runs']:,}"
    )
    print(f"Interpolated sensor values: {report['interpolated_values']:,}")
    print(
        "Missing before interpolation: "
        f"{report['missing_values_before_interpolation']}"
    )
    print(
        "Missing after interpolation: "
        f"{report['missing_values_after_interpolation_before_drop']}"
    )
    print(
        "Rows excluded for remaining missing/inf: "
        f"{report['rows_excluded_for_remaining_missing_or_inf']:,}"
    )
    print(
        f"Processed rows/segments: {report['processed_rows']:,}/"
        f"{report['segment_count']:,}"
    )
    print(
        "Segments excluded for being shorter than sequence_length: "
        f"{report['segments_shorter_than_sequence_length']:,}"
    )
    print(f"Features: {report['feature_cols']}")
    print(f"Input STATUS counts: {report['input_status_value_counts']}")
    print(f"Processed STATUS counts: {report['status_value_counts']}")
    print(
        "NOTICE: preprocessing does not fit a model scaler; "
        "scaling is fit on the train split later."
    )
