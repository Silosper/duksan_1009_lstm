"""CLI-compatible wrapper for the production preprocessing pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duksan_lstm_ae.config import DEFAULT_CONFIG_PATH, load_config
from duksan_lstm_ae.preprocessing import preprocess_csv


def preprocess_data(
    data_path: str | Path | None = None,
    output_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
):
    """Preserve the old function name while using leakage-safe preprocessing.

    ``scaler_path`` remains accepted for backward API compatibility, but no
    whole-data scaler is created. The model scaler is fit on the train split.
    """
    if scaler_path is not None:
        print(
            "NOTICE: scaler_path is ignored. "
            "The scaler is fitted and saved during model training."
        )
    config = load_config(config_path)
    processed, _, _ = preprocess_csv(
        config,
        input_path=data_path,
        output_path=output_path,
    )
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and segment the raw equipment time series."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_data(
        data_path=args.input,
        output_path=args.output,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
