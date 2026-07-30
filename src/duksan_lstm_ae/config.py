"""Central configuration loading, migration, validation, and path helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/settings.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load, migrate legacy keys, and validate a project YAML configuration."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")

    config = _migrate_legacy_config(config)
    validate_config(config)
    config["_config_path"] = str(config_path.resolve())
    return config


def save_config(
    config: dict[str, Any],
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> Path:
    """Validate and save configuration values to YAML."""
    clean_config = deepcopy(config)
    clean_config.pop("_config_path", None)
    validate_config(clean_config)

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            clean_config,
            file,
            allow_unicode=True,
            sort_keys=False,
        )
    return config_path


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def artifact_path(config: dict[str, Any], file_key: str) -> Path:
    """Return an artifact file path from the configured run directory."""
    run_dir = resolve_path(config, config["artifacts"]["run_dir"])
    return run_dir / config["artifacts"][file_key]


def frequency_delta(config: dict[str, Any]) -> pd.Timedelta:
    """Return the configured fixed sampling interval as Timedelta."""
    try:
        delta = pd.to_timedelta(config["data"]["frequency"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "data.frequency must be a fixed pandas duration such as '1min'."
        ) from exc
    if delta <= pd.Timedelta(0):
        raise ValueError("data.frequency must be positive.")
    return delta


def validate_config(config: dict[str, Any]) -> None:
    """Raise ValueError when a required setting is invalid."""
    required_sections = {
        "project",
        "data",
        "missing",
        "status",
        "split",
        "scaling",
        "window",
        "model",
        "training",
        "threshold",
        "artifacts",
        "inference",
        "streamlit",
    }
    missing = sorted(required_sections - set(config))
    if missing:
        raise ValueError(f"Missing config sections: {missing}")

    feature_cols = list(config["data"].get("feature_cols", []))
    status_col = config["data"]["status_col"]
    if not feature_cols:
        raise ValueError("data.feature_cols must not be empty.")
    if status_col in feature_cols:
        raise ValueError("STATUS/meta column must not be included in feature_cols.")
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("data.feature_cols contains duplicates.")
    frequency_delta(config)

    duplicate_policy = str(config["data"]["duplicate_policy"]).lower()
    if duplicate_policy not in {"first", "last", "mean"}:
        raise ValueError("data.duplicate_policy must be first, last, or mean.")

    method = str(config["missing"]["interpolation_method"]).lower()
    if method not in {"time", "linear"}:
        raise ValueError(
            "missing.interpolation_method must be time or linear "
            "(both disable endpoint extrapolation)."
        )
    if int(config["missing"]["max_interpolation_gap_steps"]) < 0:
        raise ValueError("max_interpolation_gap_steps must be non-negative.")
    if int(config["missing"]["max_segment_gap_steps"]) < 1:
        raise ValueError("max_segment_gap_steps must be at least 1.")

    status_cfg = config["status"]
    label_mode = status_cfg["window_label_mode"]
    if label_mode not in {"last", "any_anomaly", "majority", "all"}:
        raise ValueError(
            "status.window_label_mode must be last, any_anomaly, majority, or all."
        )
    normal_values = {_canonical_status(v) for v in status_cfg["normal_values"]}
    anomaly_values = {_canonical_status(v) for v in status_cfg["anomaly_values"]}
    overlap = normal_values & anomaly_values
    if overlap:
        raise ValueError(f"STATUS normal/anomaly values overlap: {sorted(overlap)}")
    if bool(status_cfg["use_for_training_filter"]) and not normal_values:
        raise ValueError(
            "status.use_for_training_filter=true requires status.normal_values."
        )

    split_cfg = config["split"]
    ratios = [
        float(split_cfg["train_ratio"]),
        float(split_cfg["validation_ratio"]),
        float(split_cfg["test_ratio"]),
    ]
    if any(value <= 0 for value in ratios):
        raise ValueError("All split ratios must be positive.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0.")
    if int(split_cfg["purge_gap_steps"]) < 0:
        raise ValueError("split.purge_gap_steps must be non-negative.")

    scaling_method = str(config["scaling"]["method"]).lower()
    if scaling_method not in {"standard", "robust", "minmax", "none"}:
        raise ValueError("scaling.method must be standard, robust, minmax, or none.")

    if int(config["window"]["sequence_length"]) <= 0:
        raise ValueError("window.sequence_length must be positive.")
    if int(config["window"]["stride"]) <= 0:
        raise ValueError("window.stride must be positive.")
    if not bool(config["window"]["require_complete_window"]):
        raise ValueError(
            "window.require_complete_window must be true for fixed-shape LSTM input."
        )
    interpolation_ratio = float(config["window"]["max_interpolation_ratio"])
    if not 0 <= interpolation_ratio <= 1:
        raise ValueError("window.max_interpolation_ratio must be in [0, 1].")

    if int(config["training"]["batch_size"]) <= 0:
        raise ValueError("training.batch_size must be positive.")
    if int(config["training"]["epochs"]) <= 0:
        raise ValueError("training.epochs must be positive.")

    threshold_cfg = config["threshold"]
    if threshold_cfg["method"] not in {
        "train_percentile",
        "validation_percentile",
        "mean_std",
    }:
        raise ValueError(
            "threshold.method must be train_percentile, "
            "validation_percentile, or mean_std."
        )
    percentile = float(threshold_cfg["percentile"])
    if not 0 < percentile < 100:
        raise ValueError("threshold.percentile must be between 0 and 100.")
    if threshold_cfg["mean_std_source"] not in {"train", "validation"}:
        raise ValueError("threshold.mean_std_source must be train or validation.")
    if threshold_cfg["score_aggregation"] != "window_mean":
        raise ValueError(
            "Only threshold.score_aggregation=window_mean is currently supported."
        )


def canonical_status_values(values: list[Any]) -> set[str]:
    """Normalize configured STATUS values without assigning their meaning."""
    return {_canonical_status(value) for value in values}


def _canonical_status(value: Any) -> str:
    return str(value).strip()


def _migrate_legacy_config(config: dict[str, Any]) -> dict[str, Any]:
    """Add safe defaults for snapshots created by the previous project version."""
    migrated = deepcopy(config)
    data = migrated.setdefault("data", {})
    project = migrated.setdefault("project", {})

    migrated.setdefault(
        "missing",
        {
            "interpolation_method": "time",
            "max_interpolation_gap_steps": 5,
            "max_segment_gap_steps": 1,
            "add_interpolation_mask": True,
            "interpolation_mask_prefix": "INTERPOLATED_",
            "gap_bucket_steps": [1, 5, 30, 60, 360, 1440],
        },
    )
    status = migrated.setdefault(
        "status",
        {
            "use_for_training_filter": False,
            "normal_values": [],
            "anomaly_values": [],
            "exclude_unknown": False,
            "window_label_mode": data.pop("label_mode", "last"),
        },
    )
    if "normal_label" in data and not status["normal_values"]:
        status["normal_values"] = [data["normal_label"]]
    if "fault_label" in data and not status["anomaly_values"]:
        status["anomaly_values"] = [data["fault_label"]]

    migrated.setdefault(
        "split",
        {
            "train_ratio": data.pop("train_ratio", 0.70),
            "validation_ratio": data.pop("validation_ratio", 0.15),
            "test_ratio": 0.15,
            "purge_gap_steps": 0,
        },
    )
    migrated.setdefault("scaling", {"method": "standard"})

    data.setdefault("raw_path", "data/raw/duksanind_VW_EQUIPMENT_RAW_DATA_1009.csv")
    data.setdefault("preprocessing_report_path", "data/processed/preprocessing_report.json")
    data.setdefault("segment_summary_path", "data/processed/segment_summary.csv")
    data.setdefault("status_original_col", "STATUS_ORIGINAL")
    data.setdefault("source_index_col", "SOURCE_ROW_INDEX")
    data.setdefault("frequency", "1min")
    data.setdefault("duplicate_policy", "mean")
    data.setdefault("input_is_scaled", False)

    window = migrated.setdefault("window", {})
    if "label_mode" in window:
        status["window_label_mode"] = window.pop("label_mode")
    window.setdefault("require_complete_window", True)
    window.setdefault("max_interpolation_ratio", 1.0)

    training = migrated.setdefault("training", {})
    training.setdefault("seed", project.pop("seed", 42))
    training.setdefault("gradient_clip_norm", training.pop("gradient_clip", 1.0))

    threshold = migrated.setdefault("threshold", {})
    if threshold.get("method") == "quantile":
        threshold["method"] = "validation_percentile"
        threshold["percentile"] = float(threshold.pop("quantile", 0.995)) * 100
    threshold.setdefault("method", "validation_percentile")
    threshold.setdefault("percentile", 99.5)
    threshold.setdefault("mean_std_k", 3.0)
    threshold.setdefault("mean_std_source", "validation")
    threshold.setdefault("score_aggregation", "window_mean")
    return migrated
