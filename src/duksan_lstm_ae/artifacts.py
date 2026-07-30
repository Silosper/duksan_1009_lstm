"""Loading of trained model, scaler, threshold, and run configuration."""

from __future__ import annotations

import json
from typing import Any

import torch

from .config import artifact_path, load_config
from .model import build_model
from .scaling import FeatureScaler, load_feature_scaler


def load_run(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, FeatureScaler, float, dict[str, Any]]:
    """Load one trained run and validate its feature order."""
    snapshot_path = artifact_path(config, "config_snapshot_file")
    run_config = load_config(snapshot_path) if snapshot_path.exists() else config
    model_path = artifact_path(run_config, "model_file")
    scaler_path = artifact_path(run_config, "scaler_file")
    threshold_path = artifact_path(run_config, "threshold_file")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config", run_config)
    expected_features = list(checkpoint.get(
        "feature_cols",
        checkpoint_config["data"]["feature_cols"],
    ))
    if expected_features != list(checkpoint_config["data"]["feature_cols"]):
        raise ValueError("Checkpoint feature order differs from its configuration.")
    model = build_model(checkpoint_config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scaler = load_feature_scaler(scaler_path, expected_features)
    threshold = float(
        json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"]
    )
    return model, scaler, threshold, checkpoint_config
