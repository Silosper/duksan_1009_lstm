"""Backward-compatible entry point for the production LSTM-AE trainer.

The previous implementation duplicated preprocessing, STATUS assumptions,
splitting, and scaling. It now delegates to ``duksan_lstm_ae.training`` so the
same leakage-safe pipeline is used by every training command.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duksan_lstm_ae.config import DEFAULT_CONFIG_PATH, load_config
from duksan_lstm_ae.training import train_model


@dataclass
class TrainConfig:
    """Legacy overrides retained for callers of the previous API."""

    sequence_length: int = 60
    stride: int = 1
    max_windows: int | None = 100_000
    batch_size: int = 128
    hidden_size: int = 64
    latent_size: int = 16
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 30
    patience: int = 5
    seed: int = 42


def train_lstm_autoencoder(
    legacy_config: TrainConfig | None = None,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
):
    """Run the current trainer while honoring common legacy overrides."""
    config = load_config(config_path)
    if legacy_config is not None:
        config["window"]["sequence_length"] = legacy_config.sequence_length
        config["window"]["stride"] = legacy_config.stride
        config["window"]["max_train_windows"] = legacy_config.max_windows
        config["training"]["batch_size"] = legacy_config.batch_size
        config["training"]["learning_rate"] = legacy_config.learning_rate
        config["training"]["epochs"] = legacy_config.epochs
        config["training"]["patience"] = legacy_config.patience
        config["training"]["seed"] = legacy_config.seed
        config["model"]["hidden_size"] = legacy_config.hidden_size
        config["model"]["latent_size"] = legacy_config.latent_size
        config["model"]["num_layers"] = legacy_config.num_layers
        config["model"]["dropout"] = legacy_config.dropout
    model, _, history, metrics = train_model(config)
    return model, history.to_dict(orient="records"), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for scripts/train.py."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.max_windows is not None:
        config["window"]["max_train_windows"] = args.max_windows
    train_model(config)


if __name__ == "__main__":
    main()
