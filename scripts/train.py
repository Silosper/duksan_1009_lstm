"""CLI entry point for LSTM-AE training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duksan_lstm_ae.config import DEFAULT_CONFIG_PATH, load_config
from duksan_lstm_ae.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM-AE.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    args = parser.parse_args()
    train_model(load_config(args.config))


if __name__ == "__main__":
    main()
