"""Leakage-safe splitting, scaling, and segment-safe window generation."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import (
    canonical_status_values,
    frequency_delta,
    resolve_path,
)
from .preprocessing import normalize_status
from .scaling import FeatureScaler, fit_feature_scaler


UNKNOWN_LABEL = -1
NORMAL_LABEL = 0
ANOMALY_LABEL = 1


@dataclass
class WindowBundle:
    """Fixed-length model inputs plus traceable per-window metadata."""

    X: np.ndarray
    labels: np.ndarray
    start_times: np.ndarray
    end_times: np.ndarray
    segment_ids: np.ndarray
    row_start_indices: np.ndarray
    row_end_indices: np.ndarray
    interpolation_ratios: np.ndarray
    status_references: np.ndarray
    split_names: np.ndarray
    feature_cols: list[str]
    exclusion_counts: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.X)

    def subset(self, mask: np.ndarray) -> "WindowBundle":
        """Return a metadata-preserving subset."""
        return WindowBundle(
            X=self.X[mask],
            labels=self.labels[mask],
            start_times=self.start_times[mask],
            end_times=self.end_times[mask],
            segment_ids=self.segment_ids[mask],
            row_start_indices=self.row_start_indices[mask],
            row_end_indices=self.row_end_indices[mask],
            interpolation_ratios=self.interpolation_ratios[mask],
            status_references=self.status_references[mask],
            split_names=self.split_names[mask],
            feature_cols=self.feature_cols,
            exclusion_counts=dict(self.exclusion_counts),
        )


@dataclass
class PreparedData:
    """Train, validation, and test windows with a train-fitted scaler."""

    train: WindowBundle
    validation: WindowBundle
    test: WindowBundle
    scaler: FeatureScaler
    split_info: dict[str, Any]

    # 이전 코드가 사용하던 속성명을 호환용으로 유지
    @property
    def train_normal(self) -> WindowBundle:
        return self.train

    @property
    def validation_normal(self) -> WindowBundle:
        return self.validation

    @property
    def test_all(self) -> WindowBundle:
        return self.test

    @property
    def split_times(self) -> dict[str, str]:
        return {
            key: str(value)
            for key, value in self.split_info.items()
            if key.endswith("_start") or key.endswith("_end")
        }


def load_sensor_frame(
    config: dict[str, Any],
    data_path: str | Path | None = None,
    *,
    require_status: bool = True,
) -> pd.DataFrame:
    """Load a processed frame without silently assigning STATUS semantics."""
    data_cfg = config["data"]
    path = resolve_path(config, data_path or data_cfg["processed_path"])
    time_col = data_cfg["time_col"]
    segment_col = data_cfg["segment_col"]
    status_col = data_cfg["status_col"]
    status_original_col = data_cfg["status_original_col"]
    source_index_col = data_cfg["source_index_col"]
    feature_cols = list(data_cfg["feature_cols"])
    required = [time_col, *feature_cols]

    frame = pd.read_csv(path, low_memory=False)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")
    legacy_scaler_path = resolve_path(
        config,
        data_cfg["preprocessing_scaler_path"],
    )
    looks_like_legacy_processed = (
        status_original_col not in frame.columns
        and "OBSERVED_COUNT" in frame.columns
        and legacy_scaler_path.exists()
    )
    if looks_like_legacy_processed and not bool(
        data_cfg.get("input_is_scaled", False)
    ):
        raise RuntimeError(
            "Legacy whole-data-scaled processed CSV detected. Run "
            "`python processing/preprocess.py --config config/settings.yaml` "
            "to regenerate raw-scale processed data. For temporary backward "
            "compatibility only, set data.input_is_scaled=true."
        )
    if require_status and status_col not in frame:
        warnings.warn(
            f"{status_col} is missing; STATUS filtering and label metrics "
            "will be unavailable.",
            stacklevel=2,
        )

    parsed = pd.to_datetime(frame[time_col], errors="coerce")
    invalid_time_count = int(parsed.isna().sum())
    if invalid_time_count:
        print(f"Invalid timestamps removed during load: {invalid_time_count:,}")
    frame[time_col] = parsed
    frame = frame.dropna(subset=[time_col]).copy()
    if status_col not in frame:
        frame[status_col] = pd.NA
    if status_original_col not in frame:
        frame[status_original_col] = frame[status_col].copy()
    frame[status_col] = normalize_status(frame[status_col])

    for column in feature_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[feature_cols] = frame[feature_cols].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    if source_index_col not in frame:
        frame[source_index_col] = np.arange(len(frame), dtype=np.int64)

    frame = frame.sort_values(time_col).reset_index(drop=True)
    duplicate_count = int(frame.duplicated(time_col, keep="first").sum())
    if duplicate_count:
        print(
            f"Duplicate timestamp excess rows during load: {duplicate_count:,} "
            f"(policy={data_cfg['duplicate_policy']})"
        )
        frame = _resolve_loaded_duplicates(frame, config)

    if bool(data_cfg.get("input_is_scaled", False)):
        scaler_path = resolve_path(
            config,
            data_cfg["preprocessing_scaler_path"],
        )
        legacy_scaler = joblib.load(scaler_path)
        if getattr(legacy_scaler, "n_features_in_", len(feature_cols)) != len(
            feature_cols
        ):
            raise ValueError(
                "Legacy preprocessing scaler feature count does not match "
                "data.feature_cols."
            )
        frame[feature_cols] = legacy_scaler.inverse_transform(
            frame[feature_cols]
        )
        warnings.warn(
            "A legacy whole-data preprocessing scaler was inverted. "
            "Re-run preprocessing with data.input_is_scaled=false.",
            stacklevel=2,
        )

    if segment_col not in frame:
        configured_interval = frequency_delta(config)
        frame[segment_col] = (
            frame[time_col]
            .diff()
            .ne(configured_interval)
            .cumsum()
            .astype(np.int64)
        )
    frame["_ROW_INDEX"] = np.arange(len(frame), dtype=np.int64)
    print(f"Loaded rows: {len(frame):,} | time-sorted: True")
    print(f"Features: {feature_cols}")
    print(
        "STATUS counts: "
        f"{frame[status_col].value_counts(dropna=False).to_dict()}"
    )
    return frame


def chronological_frame_split(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split rows chronologically and purge rows after each split boundary."""
    split_cfg = config["split"]
    time_col = config["data"]["time_col"]
    segment_col = config["data"]["segment_col"]
    ordered = frame.sort_values(time_col).reset_index(drop=True).copy()
    row_count = len(ordered)
    train_end = int(row_count * float(split_cfg["train_ratio"]))
    validation_end = train_end + int(
        row_count * float(split_cfg["validation_ratio"])
    )
    purge_steps = int(split_cfg["purge_gap_steps"])
    validation_start = min(train_end + purge_steps, validation_end)
    test_start = min(validation_end + purge_steps, row_count)

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[validation_start:validation_end].copy()
    test = ordered.iloc[test_start:].copy()
    if min(len(train), len(validation), len(test)) <= 0:
        raise ValueError(
            "A chronological split is empty after applying purge_gap_steps. "
            "Reduce the purge gap or adjust split ratios."
        )

    train["_SPLIT"] = "train"
    validation["_SPLIT"] = "validation"
    test["_SPLIT"] = "test"
    validate_split_time_disjoint(train, validation, test, time_col)

    info: dict[str, Any] = {
        "train_start": str(train[time_col].min()),
        "train_end": str(train[time_col].max()),
        "validation_start": str(validation[time_col].min()),
        "validation_end": str(validation[time_col].max()),
        "test_start": str(test[time_col].min()),
        "test_end": str(test[time_col].max()),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_segments": int(train[segment_col].nunique()),
        "validation_segments": int(validation[segment_col].nunique()),
        "test_segments": int(test[segment_col].nunique()),
        "purge_gap_steps": purge_steps,
        "purged_after_train_rows": validation_start - train_end,
        "purged_after_validation_rows": test_start - validation_end,
    }
    print("Chronological split:")
    for split_name in ("train", "validation", "test"):
        print(
            f"  {split_name}: {info[f'{split_name}_start']} -> "
            f"{info[f'{split_name}_end']} | "
            f"rows={info[f'{split_name}_rows']:,} | "
            f"segments={info[f'{split_name}_segments']:,}"
        )
    print(
        "  purged boundary rows: "
        f"{info['purged_after_train_rows'] + info['purged_after_validation_rows']:,}"
    )
    return train, validation, test, info


def fit_train_scaler(
    train_frame: pd.DataFrame,
    config: dict[str, Any],
) -> FeatureScaler:
    """Backward-compatible name for train-only configurable scaling."""
    return fit_feature_scaler(train_frame, config)


def transform_frame(
    frame: pd.DataFrame,
    scaler: FeatureScaler,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply the train-fitted scaler without modifying metadata columns."""
    expected = list(config["data"]["feature_cols"])
    scaler.validate_feature_order(expected)
    return scaler.transform_frame(frame)


def make_windows(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    normal_only: bool = False,
    max_windows: int | None,
    split_name: str | None = None,
) -> WindowBundle:
    """Create valid fixed windows that never cross a gap or segment boundary."""
    data_cfg = config["data"]
    window_cfg = config["window"]
    status_cfg = config["status"]
    time_col = data_cfg["time_col"]
    segment_col = data_cfg["segment_col"]
    status_col = data_cfg["status_col"]
    source_index_col = data_cfg["source_index_col"]
    feature_cols = list(data_cfg["feature_cols"])
    sequence_length = int(window_cfg["sequence_length"])
    stride = int(window_cfg["stride"])
    configured_interval = frequency_delta(config)
    max_interpolation_ratio = float(
        window_cfg["max_interpolation_ratio"]
    )
    prefix = config["missing"]["interpolation_mask_prefix"]
    interpolation_cols = [
        f"{prefix}{column}"
        for column in feature_cols
        if f"{prefix}{column}" in frame.columns
    ]
    normal_values = canonical_status_values(status_cfg["normal_values"])
    anomaly_values = canonical_status_values(status_cfg["anomaly_values"])
    effective_split = split_name or (
        str(frame["_SPLIT"].iloc[0]) if "_SPLIT" in frame and len(frame) else "unknown"
    )

    chunks: dict[str, list[np.ndarray]] = {
        "X": [],
        "labels": [],
        "start_times": [],
        "end_times": [],
        "segment_ids": [],
        "row_start_indices": [],
        "row_end_indices": [],
        "interpolation_ratios": [],
        "status_references": [],
        "split_names": [],
    }
    exclusions = {
        "segments_too_short": 0,
        "windows_nonfinite": 0,
        "windows_interpolation_ratio": 0,
        "windows_status_filter": 0,
        "windows_unknown_status": 0,
        "windows_capped": 0,
    }
    created = 0

    ordered = frame.sort_values(time_col).copy()
    segment_break = ordered[segment_col].ne(ordered[segment_col].shift())
    time_break = ordered[time_col].diff().ne(configured_interval)
    ordered["_WINDOW_SEGMENT"] = (segment_break | time_break).cumsum()

    for window_segment_id, group in ordered.groupby(
        "_WINDOW_SEGMENT",
        sort=False,
    ):
        if len(group) < sequence_length:
            exclusions["segments_too_short"] += 1
            continue

        features = group[feature_cols].to_numpy(dtype=np.float32, copy=True)
        finite_row = np.isfinite(features).all(axis=1)
        invalid_cumulative = np.concatenate(
            ([0], np.cumsum(~finite_row, dtype=np.int64))
        )
        times = group[time_col].to_numpy(dtype="datetime64[ns]")
        source_indices = group[source_index_col].to_numpy(dtype=np.int64)
        statuses = group[status_col].astype("string").to_numpy(dtype=object)
        starts = np.arange(
            0,
            len(group) - sequence_length + 1,
            stride,
            dtype=np.int64,
        )
        ends = starts + sequence_length

        invalid_counts = invalid_cumulative[ends] - invalid_cumulative[starts]
        finite_windows = invalid_counts == 0
        exclusions["windows_nonfinite"] += int((~finite_windows).sum())
        starts = starts[finite_windows]
        if len(starts) == 0:
            continue

        if interpolation_cols:
            interpolation = group[interpolation_cols].to_numpy(
                dtype=np.float32
            )
            row_interpolation_count = interpolation.sum(axis=1)
            interpolation_cumulative = np.concatenate(
                ([0.0], np.cumsum(row_interpolation_count))
            )
            ratios = (
                interpolation_cumulative[starts + sequence_length]
                - interpolation_cumulative[starts]
            ) / (sequence_length * len(feature_cols))
        else:
            ratios = np.zeros(len(starts), dtype=np.float32)
        accepted_ratio = ratios <= max_interpolation_ratio
        exclusions["windows_interpolation_ratio"] += int(
            (~accepted_ratio).sum()
        )
        starts = starts[accepted_ratio]
        ratios = ratios[accepted_ratio]
        if len(starts) == 0:
            continue

        labels, references = _window_status_metadata(
            statuses,
            starts,
            sequence_length,
            normal_values,
            anomaly_values,
            status_cfg["window_label_mode"],
        )
        keep = np.ones(len(starts), dtype=bool)
        if normal_only:
            keep &= labels == NORMAL_LABEL
            exclusions["windows_status_filter"] += int((~keep).sum())
        elif bool(status_cfg["exclude_unknown"]):
            known = labels != UNKNOWN_LABEL
            exclusions["windows_unknown_status"] += int((~known).sum())
            keep &= known
        starts = starts[keep]
        ratios = ratios[keep]
        labels = labels[keep]
        references = references[keep]
        if len(starts) == 0:
            continue

        if max_windows is not None:
            remaining = int(max_windows) - created
            if remaining <= 0:
                exclusions["windows_capped"] += len(starts)
                break
            if len(starts) > remaining:
                exclusions["windows_capped"] += len(starts) - remaining
                starts = starts[:remaining]
                ratios = ratios[:remaining]
                labels = labels[:remaining]
                references = references[:remaining]

        for offset in range(0, len(starts), 10_000):
            start_chunk = starts[offset : offset + 10_000]
            end_chunk = start_chunk + sequence_length - 1
            indices = start_chunk[:, None] + np.arange(sequence_length)
            chunks["X"].append(features[indices])
            chunks["labels"].append(
                labels[offset : offset + len(start_chunk)]
            )
            chunks["start_times"].append(times[start_chunk])
            chunks["end_times"].append(times[end_chunk])
            chunks["segment_ids"].append(
                np.full(len(start_chunk), int(window_segment_id), dtype=np.int64)
            )
            chunks["row_start_indices"].append(source_indices[start_chunk])
            chunks["row_end_indices"].append(source_indices[end_chunk])
            chunks["interpolation_ratios"].append(
                ratios[offset : offset + len(start_chunk)]
            )
            chunks["status_references"].append(
                references[offset : offset + len(start_chunk)]
            )
            chunks["split_names"].append(
                np.full(len(start_chunk), effective_split, dtype=object)
            )
        created += len(starts)

    bundle = _concatenate_window_chunks(
        chunks,
        sequence_length,
        feature_cols,
        exclusions,
    )
    validate_window_bundle(bundle, config)
    print(
        f"{effective_split} windows: {len(bundle):,} | "
        f"shape={bundle.X.shape} | exclusions={exclusions}"
    )
    return bundle


def prepare_training_data(
    config: dict[str, Any],
    data_path: str | Path | None = None,
    scaler: FeatureScaler | None = None,
) -> PreparedData:
    """Prepare split-first, train-scaled, segment-safe model windows."""
    frame = load_sensor_frame(config, data_path)
    train_frame, validation_frame, test_frame, split_info = (
        chronological_frame_split(frame, config)
    )
    if not bool(config["status"]["use_for_training_filter"]):
        warnings.warn(
            "STATUS training filter is disabled. All otherwise valid train "
            "windows form an unsupervised baseline; STATUS is metadata only.",
            stacklevel=2,
        )
    if scaler is None:
        scaler = fit_train_scaler(train_frame, config)
    else:
        scaler.validate_feature_order(list(config["data"]["feature_cols"]))

    train_frame = transform_frame(train_frame, scaler, config)
    validation_frame = transform_frame(validation_frame, scaler, config)
    test_frame = transform_frame(test_frame, scaler, config)
    window_cfg = config["window"]
    filter_training = bool(config["status"]["use_for_training_filter"])

    train_windows = make_windows(
        train_frame,
        config,
        normal_only=filter_training,
        max_windows=_optional_int(window_cfg.get("max_train_windows")),
        split_name="train",
    )
    validation_windows = make_windows(
        validation_frame,
        config,
        normal_only=filter_training,
        max_windows=_optional_int(window_cfg.get("max_validation_windows")),
        split_name="validation",
    )
    test_windows = make_windows(
        test_frame,
        config,
        normal_only=False,
        max_windows=_optional_int(window_cfg.get("max_test_windows")),
        split_name="test",
    )
    if min(len(train_windows), len(validation_windows), len(test_windows)) == 0:
        raise ValueError(
            "At least one prepared window split is empty. Review sequence "
            "length, purge gap, missing values, STATUS filter, and split ratios."
        )
    split_info.update(
        {
            "train_windows": len(train_windows),
            "validation_windows": len(validation_windows),
            "test_windows": len(test_windows),
            "train_window_exclusions": train_windows.exclusion_counts,
            "validation_window_exclusions": validation_windows.exclusion_counts,
            "test_window_exclusions": test_windows.exclusion_counts,
        }
    )
    validate_window_split_disjoint(
        train_windows,
        validation_windows,
        test_windows,
    )
    return PreparedData(
        train=train_windows,
        validation=validation_windows,
        test=test_windows,
        scaler=scaler,
        split_info=split_info,
    )


def validate_split_time_disjoint(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    time_col: str,
) -> None:
    """Ensure source timestamps cannot appear in more than one split."""
    if not (
        train[time_col].max() < validation[time_col].min()
        and validation[time_col].max() < test[time_col].min()
    ):
        raise AssertionError("Chronological split time ranges overlap.")


def validate_window_split_disjoint(
    train: WindowBundle,
    validation: WindowBundle,
    test: WindowBundle,
) -> None:
    """Ensure no model window time range overlaps a different split."""
    if not (
        pd.Timestamp(train.end_times.max())
        < pd.Timestamp(validation.start_times.min())
        and pd.Timestamp(validation.end_times.max())
        < pd.Timestamp(test.start_times.min())
    ):
        raise AssertionError("Window time ranges overlap across splits.")


def validate_window_bundle(
    bundle: WindowBundle,
    config: dict[str, Any],
) -> None:
    """Assert core model-input and continuity invariants."""
    expected_shape = (
        len(bundle),
        int(config["window"]["sequence_length"]),
        len(config["data"]["feature_cols"]),
    )
    if bundle.X.shape != expected_shape:
        raise AssertionError(
            f"Window tensor shape mismatch: {bundle.X.shape} != {expected_shape}"
        )
    if not np.isfinite(bundle.X).all():
        raise AssertionError("Window tensor contains NaN or infinite values.")
    if bundle.feature_cols != list(config["data"]["feature_cols"]):
        raise AssertionError("Window feature order differs from configuration.")
    if config["data"]["status_col"] in bundle.feature_cols:
        raise AssertionError("STATUS must not be included in model features.")
    if len(bundle):
        expected_duration = frequency_delta(config) * (
            int(config["window"]["sequence_length"]) - 1
        )
        durations = pd.to_datetime(bundle.end_times) - pd.to_datetime(
            bundle.start_times
        )
        if not np.all(durations == expected_duration):
            raise AssertionError(
                "At least one window has an irregular internal time interval."
            )


def _resolve_loaded_duplicates(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    data_cfg = config["data"]
    time_col = data_cfg["time_col"]
    feature_cols = list(data_cfg["feature_cols"])
    policy = str(data_cfg["duplicate_policy"]).lower()
    if policy in {"first", "last"}:
        return (
            frame.drop_duplicates(time_col, keep=policy)
            .sort_values(time_col)
            .reset_index(drop=True)
        )

    aggregations: dict[str, Any] = {
        column: "mean" for column in feature_cols
    }
    for column in frame.columns:
        if column in aggregations or column == time_col:
            continue
        aggregations[column] = "max" if column.startswith(
            config["missing"]["interpolation_mask_prefix"]
        ) else "last"
    return (
        frame.groupby(time_col, as_index=False, sort=True)
        .agg(aggregations)
        .reset_index(drop=True)
    )


def _window_status_metadata(
    statuses: np.ndarray,
    starts: np.ndarray,
    sequence_length: int,
    normal_values: set[str],
    anomaly_values: set[str],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full(len(starts), UNKNOWN_LABEL, dtype=np.int8)
    references = np.empty(len(starts), dtype=object)
    for index, start in enumerate(starts):
        raw_values = statuses[start : start + sequence_length]
        cleaned = [
            str(value).strip()
            for value in raw_values
            if not pd.isna(value) and str(value).strip()
        ]
        reference_values = (
            cleaned if mode == "all" else list(dict.fromkeys(cleaned))
        )
        references[index] = json.dumps(reference_values, ensure_ascii=False)
        if not normal_values and not anomaly_values:
            continue
        if mode == "last":
            last = cleaned[-1] if cleaned else None
            if last in normal_values:
                labels[index] = NORMAL_LABEL
            elif last in anomaly_values:
                labels[index] = ANOMALY_LABEL
        elif mode == "any_anomaly":
            if any(value in anomaly_values for value in cleaned):
                labels[index] = ANOMALY_LABEL
            elif cleaned and all(value in normal_values for value in cleaned):
                labels[index] = NORMAL_LABEL
        elif mode == "majority":
            normal_count = sum(value in normal_values for value in cleaned)
            anomaly_count = sum(value in anomaly_values for value in cleaned)
            if normal_count > anomaly_count:
                labels[index] = NORMAL_LABEL
            elif anomaly_count > normal_count:
                labels[index] = ANOMALY_LABEL
        elif mode == "all":
            if cleaned and all(value in normal_values for value in cleaned):
                labels[index] = NORMAL_LABEL
            elif cleaned and all(value in anomaly_values for value in cleaned):
                labels[index] = ANOMALY_LABEL
    return labels, references


def _concatenate_window_chunks(
    chunks: dict[str, list[np.ndarray]],
    sequence_length: int,
    feature_cols: list[str],
    exclusions: dict[str, int],
) -> WindowBundle:
    if not chunks["X"]:
        return WindowBundle(
            X=np.empty(
                (0, sequence_length, len(feature_cols)),
                dtype=np.float32,
            ),
            labels=np.empty(0, dtype=np.int8),
            start_times=np.empty(0, dtype="datetime64[ns]"),
            end_times=np.empty(0, dtype="datetime64[ns]"),
            segment_ids=np.empty(0, dtype=np.int64),
            row_start_indices=np.empty(0, dtype=np.int64),
            row_end_indices=np.empty(0, dtype=np.int64),
            interpolation_ratios=np.empty(0, dtype=np.float32),
            status_references=np.empty(0, dtype=object),
            split_names=np.empty(0, dtype=object),
            feature_cols=feature_cols,
            exclusion_counts=exclusions,
        )
    return WindowBundle(
        X=np.concatenate(chunks["X"]),
        labels=np.concatenate(chunks["labels"]),
        start_times=np.concatenate(chunks["start_times"]),
        end_times=np.concatenate(chunks["end_times"]),
        segment_ids=np.concatenate(chunks["segment_ids"]),
        row_start_indices=np.concatenate(chunks["row_start_indices"]),
        row_end_indices=np.concatenate(chunks["row_end_indices"]),
        interpolation_ratios=np.concatenate(chunks["interpolation_ratios"]),
        status_references=np.concatenate(chunks["status_references"]),
        split_names=np.concatenate(chunks["split_names"]),
        feature_cols=feature_cols,
        exclusion_counts=exclusions,
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    return None if parsed <= 0 else parsed
