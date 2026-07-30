"""Check configuration, data schema, and optional runtime dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duksan_lstm_ae.config import DEFAULT_CONFIG_PATH, load_config
from duksan_lstm_ae.data import load_sensor_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate project setup.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    args = parser.parse_args()
    config = load_config(args.config)
    frame = load_sensor_frame(config)

    result = {
        "config": "OK",
        "rows": len(frame),
        "start": str(frame[config["data"]["time_col"]].min()),
        "end": str(frame[config["data"]["time_col"]].max()),
        "segments": int(frame[config["data"]["segment_col"]].nunique()),
        "features": config["data"]["feature_cols"],
        "frequency": config["data"]["frequency"],
        "scaler_method": config["scaling"]["method"],
        "status_training_filter": config["status"]["use_for_training_filter"],
        "status_normal_values": config["status"]["normal_values"],
        "status_anomaly_values": config["status"]["anomaly_values"],
        "threshold_method": config["threshold"]["method"],
    }
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        result["torch"] = "NOT INSTALLED"

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
