"""Batch anomaly inference for fixed-frequency sensor CSV files."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import load_run
from .config import resolve_path
from .data import load_sensor_frame, make_windows, transform_frame
from .evaluation import score_windows
from .runtime import select_device


def run_inference(
    config: dict[str, Any],
    *,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    input_is_scaled: bool | None = None,
) -> pd.DataFrame:
    """Load trained artifacts, validate features, score, and save predictions."""
    device = select_device(config["training"]["device"])
    model, scaler, threshold, run_config = load_run(config, device)
    inference_config = copy.deepcopy(run_config)
    inference_config["inference"] = config["inference"]
    inference_config["training"]["device"] = config["training"]["device"]
    if input_is_scaled is not None:
        inference_config["data"]["input_is_scaled"] = bool(input_is_scaled)

    scaler.validate_feature_order(
        list(inference_config["data"]["feature_cols"])
    )
    frame = load_sensor_frame(
        inference_config,
        data_path=input_path,
        require_status=False,
    )
    frame = transform_frame(frame, scaler, inference_config)
    windows = make_windows(
        frame,
        inference_config,
        normal_only=False,
        max_windows=None,
        split_name="inference",
    )
    if len(windows) == 0:
        raise ValueError(
            "No valid inference windows. Check frequency, missing values, "
            "segment lengths, and sequence_length."
        )
    result = score_windows(
        model,
        windows,
        inference_config,
        device,
    )
    result["prediction"] = (
        result["anomaly_score"] >= threshold
    ).astype(int)
    result["threshold"] = threshold
    result["detection"] = result["prediction"].map(
        {0: "not_detected", 1: "detected_anomaly"}
    )

    destination = resolve_path(
        inference_config,
        output_path or inference_config["inference"]["output_path"],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, encoding="utf-8-sig")
    return result
