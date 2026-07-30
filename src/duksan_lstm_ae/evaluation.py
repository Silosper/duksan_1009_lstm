"""Reconstruction scoring, configurable thresholding, and optional metrics."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import artifact_path
from .data import (
    NORMAL_LABEL,
    UNKNOWN_LABEL,
    PreparedData,
    WindowBundle,
)
from .model import window_reconstruction_errors
from .runtime import make_loader


def score_windows(
    model: torch.nn.Module,
    bundle: WindowBundle,
    config: dict[str, Any],
    device: torch.device,
) -> pd.DataFrame:
    """Return window-mean MSE and feature-mean MSE with metadata."""
    if len(bundle) == 0:
        raise ValueError("Cannot score an empty window bundle.")
    batch_size = int(config["inference"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    loader = make_loader(
        bundle.X,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    score_parts: list[np.ndarray] = []
    sensor_parts: list[np.ndarray] = []
    max_timestep_parts: list[np.ndarray] = []
    last_timestep_parts: list[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(device, non_blocking=True)
            reconstruction = model(inputs)
            if reconstruction.shape != inputs.shape:
                raise RuntimeError(
                    "Model output shape differs from reconstruction target: "
                    f"{tuple(reconstruction.shape)} != {tuple(inputs.shape)}"
                )
            scores, sensor_errors = window_reconstruction_errors(
                inputs,
                reconstruction,
            )
            timestep_errors = (reconstruction - inputs).pow(2).mean(dim=2)
            if not torch.isfinite(scores).all():
                raise FloatingPointError(
                    "Non-finite reconstruction error detected during scoring."
                )
            score_parts.append(scores.cpu().numpy())
            sensor_parts.append(sensor_errors.cpu().numpy())
            max_timestep_parts.append(
                timestep_errors.max(dim=1).values.cpu().numpy()
            )
            last_timestep_parts.append(
                timestep_errors[:, -1].cpu().numpy()
            )

    scores = np.concatenate(score_parts)
    sensor_errors = np.concatenate(sensor_parts)
    max_timestep_errors = np.concatenate(max_timestep_parts)
    last_timestep_errors = np.concatenate(last_timestep_parts)
    result = pd.DataFrame(
        {
            "start_timestamp": pd.to_datetime(bundle.start_times),
            "timestamp": pd.to_datetime(bundle.end_times),
            "segment_id": bundle.segment_ids,
            "source_start_index": bundle.row_start_indices,
            "source_end_index": bundle.row_end_indices,
            "interpolation_ratio": bundle.interpolation_ratios,
            "status_reference": bundle.status_references,
            "split": bundle.split_names,
            "label": bundle.labels,
            "max_timestep_error": max_timestep_errors,
            "last_timestep_error": last_timestep_errors,
            "anomaly_score": scores,
        }
    )
    for index, column in enumerate(bundle.feature_cols):
        result[f"error_{column}"] = sensor_errors[:, index]
    return result


def select_threshold(
    train_scores: np.ndarray,
    validation_scores: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Select a threshold without using test scores."""
    threshold_cfg = config["threshold"]
    method = threshold_cfg["method"]
    percentile = float(threshold_cfg["percentile"])
    if method == "train_percentile":
        reference = _finite_scores(train_scores, "train")
        threshold = float(np.percentile(reference, percentile))
        metadata = {
            "method": method,
            "source": "train",
            "percentile": percentile,
        }
    elif method == "validation_percentile":
        reference = _finite_scores(validation_scores, "validation")
        threshold = float(np.percentile(reference, percentile))
        metadata = {
            "method": method,
            "source": "validation",
            "percentile": percentile,
        }
    elif method == "mean_std":
        source = threshold_cfg["mean_std_source"]
        reference = _finite_scores(
            train_scores if source == "train" else validation_scores,
            source,
        )
        k = float(threshold_cfg["mean_std_k"])
        mean = float(reference.mean())
        std = float(reference.std(ddof=0))
        threshold = mean + k * std
        metadata = {
            "method": method,
            "source": source,
            "mean": mean,
            "std": std,
            "k": k,
        }
    else:
        raise ValueError(f"Unsupported threshold method: {method}")

    metadata.update(
        {
            "threshold": threshold,
            "reference_window_count": int(len(reference)),
            "score_aggregation": threshold_cfg["score_aggregation"],
        }
    )
    return threshold, metadata


def calculate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    *,
    labels_configured: bool,
) -> dict[str, Any]:
    """Calculate label metrics only when explicit STATUS mappings are usable."""
    metrics: dict[str, Any] = {
        "sample_count": int(len(scores)),
        "detected_count": int(predictions.sum()),
        "detection_rate": float(predictions.mean()) if len(predictions) else 0.0,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "label_metrics_available": False,
    }
    known = labels != UNKNOWN_LABEL
    known_labels = labels[known]
    if not labels_configured or len(known_labels) == 0:
        return metrics
    if len(np.unique(known_labels)) < 2:
        metrics["label_metrics_reason"] = (
            "Known test windows do not contain both configured classes."
        )
        return metrics

    known_scores = scores[known]
    known_predictions = predictions[known]
    matrix = confusion_matrix(known_labels, known_predictions, labels=[0, 1])
    metrics.update(
        {
            "label_metrics_available": True,
            "labeled_sample_count": int(len(known_labels)),
            "unknown_label_count": int((~known).sum()),
            "normal_count": int((known_labels == 0).sum()),
            "anomaly_count": int((known_labels == 1).sum()),
            "accuracy": float(accuracy_score(known_labels, known_predictions)),
            "precision": float(
                precision_score(
                    known_labels,
                    known_predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(known_labels, known_predictions, zero_division=0)
            ),
            "f1": float(
                f1_score(known_labels, known_predictions, zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(known_labels, known_scores)),
            "pr_auc": float(
                average_precision_score(known_labels, known_scores)
            ),
            "confusion_matrix": matrix.tolist(),
        }
    )
    return metrics


def evaluate_and_save(
    model: torch.nn.Module,
    prepared: PreparedData,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    """Set a non-test threshold and evaluate test windows when labels permit."""
    train = score_windows(model, prepared.train, config, device)
    validation = score_windows(model, prepared.validation, config, device)
    train_reference = _normal_reference_if_configured(train, config, "train")
    validation_reference = _normal_reference_if_configured(
        validation,
        config,
        "validation",
    )
    threshold, threshold_metadata = select_threshold(
        train_reference,
        validation_reference,
        config,
    )

    test = score_windows(model, prepared.test, config, device)
    test["prediction"] = (
        test["anomaly_score"].to_numpy() >= threshold
    ).astype(np.int64)
    status_cfg = config["status"]
    labels_configured = bool(
        status_cfg["normal_values"] and status_cfg["anomaly_values"]
    )
    metrics = calculate_metrics(
        test["label"].to_numpy(),
        test["anomaly_score"].to_numpy(),
        test["prediction"].to_numpy(),
        labels_configured=labels_configured,
    )
    metrics["threshold"] = threshold
    metrics["threshold_details"] = threshold_metadata

    threshold_path = artifact_path(config, "threshold_file")
    metrics_path = artifact_path(config, "metrics_file")
    evaluation_path = artifact_path(config, "evaluation_file")
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.write_text(
        json.dumps(threshold_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    test.to_csv(evaluation_path, index=False, encoding="utf-8-sig")
    print(
        f"Threshold: {threshold:.6f} | "
        f"method={threshold_metadata['method']} | "
        f"source={threshold_metadata['source']}"
    )
    if not metrics["label_metrics_available"]:
        print(
            "NOTICE: label-based metrics were not calculated because explicit "
            "and usable STATUS normal/anomaly mappings were not available."
        )
    return threshold, metrics, test


def evaluate_saved_run(
    config: dict[str, Any],
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    """Reload artifacts and rerun threshold selection and test evaluation."""
    from .artifacts import load_run
    from .data import prepare_training_data
    from .runtime import select_device

    device = select_device(config["training"]["device"])
    model, scaler, _, run_config = load_run(config, device)
    # 모델 구조와 split은 학습 당시 값을 유지하고 운영 가능한 설정만 갱신
    run_config["threshold"] = config["threshold"]
    run_config["status"] = config["status"]
    run_config["training"]["device"] = config["training"]["device"]
    run_config["artifacts"] = config["artifacts"]
    run_config["inference"] = config["inference"]
    prepared = prepare_training_data(run_config, scaler=scaler)
    return evaluate_and_save(model, prepared, run_config, device)


def _normal_reference_if_configured(
    scored: pd.DataFrame,
    config: dict[str, Any],
    split_name: str,
) -> np.ndarray:
    status_cfg = config["status"]
    if status_cfg["normal_values"]:
        normal = scored.loc[
            scored["label"].eq(NORMAL_LABEL),
            "anomaly_score",
        ].to_numpy()
        if len(normal):
            return normal
        warnings.warn(
            f"No configured-normal windows exist in {split_name}; "
            "using all windows for unsupervised threshold calibration.",
            stacklevel=2,
        )
    return scored["anomaly_score"].to_numpy()


def _finite_scores(scores: np.ndarray, source: str) -> np.ndarray:
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError(f"No finite {source} reconstruction scores.")
    return finite
