"""End-to-end LSTM Autoencoder training pipeline."""

from __future__ import annotations

import copy
import json
import math
import time
from typing import Any

import pandas as pd
import torch

from .config import artifact_path, save_config
from .data import PreparedData, prepare_training_data
from .evaluation import evaluate_and_save
from .model import build_model
from .runtime import make_loader, select_device, set_seed
from .scaling import save_feature_scaler


def _average_loss(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for batch_index, (inputs,) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            reconstruction = model(inputs)
            if reconstruction.shape != inputs.shape:
                raise RuntimeError(
                    "Validation reconstruction shape mismatch: "
                    f"{tuple(reconstruction.shape)} != {tuple(inputs.shape)}"
                )
            loss = criterion(reconstruction, inputs)
            _raise_on_nonfinite_loss(loss, inputs, "validation", batch_index)
            total_loss += loss.item() * len(inputs)
            total_samples += len(inputs)
    if total_samples == 0:
        raise ValueError("Validation DataLoader produced no samples.")
    return total_loss / total_samples


def train_model(
    config: dict[str, Any],
) -> tuple[torch.nn.Module, PreparedData, pd.DataFrame, dict[str, Any]]:
    """Train, select the best epoch, evaluate, and save all artifacts."""
    set_seed(int(config["training"]["seed"]))
    device = select_device(config["training"]["device"])
    prepared = prepare_training_data(config)
    training_cfg = config["training"]
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        prepared.train.X,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg["num_workers"]),
        pin_memory=pin_memory,
    )
    validation_loader = make_loader(
        prepared.validation.X,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg["num_workers"]),
        pin_memory=pin_memory,
    )

    model = build_model(config, device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    criterion = torch.nn.MSELoss()
    best_state = None
    best_validation_loss = float("inf")
    patience_count = 0
    history_rows: list[dict[str, float | int]] = []
    started = time.time()

    print(f"Device: {device}")
    print(f"Tensor shape: {prepared.train.X.shape}")
    print(f"Train windows: {len(prepared.train):,}")
    print(f"Validation windows: {len(prepared.validation):,}")
    print(f"Test windows: {len(prepared.test):,}")
    print(f"Model trainable parameters: {parameter_count:,}")

    # 첫 batch로 모델 입출력 shape를 학습 전에 검증
    sample_inputs = next(iter(train_loader))[0].to(device)
    with torch.no_grad():
        sample_outputs = model(sample_inputs)
    if sample_outputs.shape != sample_inputs.shape:
        raise RuntimeError(
            "LSTM-AE output shape must match input shape: "
            f"{tuple(sample_outputs.shape)} != {tuple(sample_inputs.shape)}"
        )

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for batch_index, (inputs,) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(inputs)
            loss = criterion(reconstruction, inputs)
            _raise_on_nonfinite_loss(loss, inputs, "train", batch_index)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(training_cfg["gradient_clip_norm"]),
            )
            optimizer.step()
            train_loss_sum += loss.item() * len(inputs)
            train_samples += len(inputs)

        if train_samples == 0:
            raise ValueError("Train DataLoader produced no samples.")
        train_loss = train_loss_sum / train_samples
        validation_loss = _average_loss(model, validation_loader, device)
        elapsed = time.time() - started
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.6f} | "
            f"validation={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= int(training_cfg["patience"]):
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model state.")
    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    _save_training_artifacts(
        model,
        prepared,
        history,
        best_validation_loss,
        parameter_count,
        config,
    )
    threshold, metrics, _ = evaluate_and_save(
        model,
        prepared,
        config,
        device,
    )
    print(f"Final threshold: {threshold:.6f}")
    if metrics["label_metrics_available"]:
        print(f"Test F1: {metrics['f1']:.4f}")
    return model, prepared, history, metrics


def _save_training_artifacts(
    model: torch.nn.Module,
    prepared: PreparedData,
    history: pd.DataFrame,
    best_validation_loss: float,
    parameter_count: int,
    config: dict[str, Any],
) -> None:
    model_path = artifact_path(config, "model_file")
    scaler_path = artifact_path(config, "scaler_file")
    history_path = artifact_path(config, "history_file")
    snapshot_path = artifact_path(config, "config_snapshot_file")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    clean_config = copy.deepcopy(config)
    clean_config.pop("_config_path", None)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": clean_config,
            "feature_cols": config["data"]["feature_cols"],
            "scaler_method": prepared.scaler.method,
            "split_info": prepared.split_info,
            "best_validation_loss": best_validation_loss,
            "model_parameter_count": parameter_count,
            "threshold_config": config["threshold"],
        },
        model_path,
    )
    save_feature_scaler(prepared.scaler, scaler_path)
    history.to_csv(history_path, index=False)
    save_config(clean_config, snapshot_path)

    metadata_path = model_path.parent / "training_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "best_validation_loss": best_validation_loss,
                "model_parameter_count": parameter_count,
                "feature_cols": config["data"]["feature_cols"],
                "scaler_method": prepared.scaler.method,
                "split_info": prepared.split_info,
                "status_configuration": config["status"],
                "threshold_configuration": config["threshold"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _raise_on_nonfinite_loss(
    loss: torch.Tensor,
    inputs: torch.Tensor,
    phase: str,
    batch_index: int,
) -> None:
    if math.isfinite(float(loss.detach().cpu())):
        return
    finite_ratio = float(torch.isfinite(inputs).float().mean().cpu())
    finite_inputs = inputs[torch.isfinite(inputs)]
    minimum = float(finite_inputs.min().cpu()) if finite_inputs.numel() else None
    maximum = float(finite_inputs.max().cpu()) if finite_inputs.numel() else None
    raise FloatingPointError(
        f"Non-finite {phase} loss at batch {batch_index}. "
        f"input_finite_ratio={finite_ratio:.6f}, "
        f"input_min={minimum}, input_max={maximum}. "
        "Check scaling, learning rate, and source sensor values."
    )
