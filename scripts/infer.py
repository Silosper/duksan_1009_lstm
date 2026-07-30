"""CLI entry point for CSV anomaly inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duksan_lstm_ae.config import DEFAULT_CONFIG_PATH, load_config
from duksan_lstm_ae.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LSTM-AE inference.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--input-is-scaled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use --input-is-scaled only for a legacy whole-data-scaled CSV, "
        "or --no-input-is-scaled for current raw-scale sensor values.",
    )
    args = parser.parse_args()
    result = run_inference(
        load_config(args.config),
        input_path=args.input,
        output_path=args.output,
        input_is_scaled=args.input_is_scaled,
    )
    print(f"Predicted windows: {len(result):,}")
    print(f"Detected anomalies: {result['prediction'].sum():,}")


if __name__ == "__main__":
    main()
