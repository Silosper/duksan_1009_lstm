"""CLI entry point for reevaluating a trained LSTM-AE run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duksan_lstm_ae.config import DEFAULT_CONFIG_PATH, load_config
from duksan_lstm_ae.evaluation import evaluate_saved_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved LSTM-AE.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    args = parser.parse_args()
    threshold, metrics, _ = evaluate_saved_run(load_config(args.config))
    print(f"Threshold: {threshold:.6f}")
    print(f"Detection rate: {metrics['detection_rate']:.4f}")
    if metrics["label_metrics_available"]:
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
    else:
        print("Label metrics: unavailable (explicit STATUS mapping required)")


if __name__ == "__main__":
    main()
