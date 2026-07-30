"""Shared synthetic configuration for data-pipeline tests."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


BASE_CONFIG = {
    "project": {"name": "test_lstm_ae"},
    "data": {
        "raw_path": "unused.csv",
        "processed_path": "unused.csv",
        "preprocessing_report_path": "unused.json",
        "segment_summary_path": "unused_segments.csv",
        "time_col": "GA_DT",
        "segment_col": "SEGMENT_ID",
        "status_col": "STATUS",
        "status_original_col": "STATUS_ORIGINAL",
        "source_index_col": "SOURCE_ROW_INDEX",
        "frequency": "1min",
        "duplicate_policy": "mean",
        "input_is_scaled": False,
        "preprocessing_scaler_path": "unused.pkl",
        "feature_cols": [
            "CURRENT1",
            "VOLTAGE",
            "TEMP_CUR",
            "ANALOGUE",
            "GROUND",
        ],
    },
    "missing": {
        "interpolation_method": "time",
        "max_interpolation_gap_steps": 2,
        "max_segment_gap_steps": 1,
        "add_interpolation_mask": True,
        "interpolation_mask_prefix": "INTERPOLATED_",
        "gap_bucket_steps": [1, 2, 5, 30],
    },
    "status": {
        "use_for_training_filter": False,
        "normal_values": [],
        "anomaly_values": [],
        "exclude_unknown": False,
        "window_label_mode": "last",
    },
    "split": {
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
        "purge_gap_steps": 2,
    },
    "scaling": {"method": "robust"},
    "window": {
        "sequence_length": 3,
        "stride": 1,
        "require_complete_window": True,
        "max_interpolation_ratio": 1.0,
        "max_train_windows": None,
        "max_validation_windows": None,
        "max_test_windows": None,
    },
    "model": {
        "hidden_size": 8,
        "latent_size": 3,
        "num_layers": 1,
        "dropout": 0.0,
    },
    "training": {
        "seed": 42,
        "batch_size": 4,
        "epochs": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "patience": 1,
        "gradient_clip_norm": 1.0,
        "num_workers": 0,
        "device": "cpu",
    },
    "threshold": {
        "method": "validation_percentile",
        "percentile": 99.0,
        "mean_std_k": 3.0,
        "mean_std_source": "validation",
        "score_aggregation": "window_mean",
    },
    "artifacts": {
        "run_dir": "artifacts/test",
        "model_file": "model.pt",
        "scaler_file": "scaler.pkl",
        "history_file": "history.csv",
        "threshold_file": "threshold.json",
        "metrics_file": "metrics.json",
        "evaluation_file": "evaluation.csv",
        "config_snapshot_file": "settings.yaml",
    },
    "inference": {"batch_size": 4, "output_path": "unused_output.csv"},
    "streamlit": {"host": "127.0.0.1", "port": 8501, "max_timeline_points": 100},
}


@pytest.fixture
def config() -> dict:
    return deepcopy(BASE_CONFIG)
