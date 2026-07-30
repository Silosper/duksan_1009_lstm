"""Create segment-safe sliding windows and PyTorch tensors.

This module is retained for older notebook compatibility. Production LSTM-AE
training uses ``src/duksan_lstm_ae/data.py`` and does not use STATUS as input.
No STATUS meaning is assigned unless the caller explicitly supplies label_map.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_DIR
    / "data/processed/duksanind_VW_EQUIPMENT_PREPROCESSED.csv"
)
OUTPUT_DIR = PROJECT_DIR / "data/windowed"

TIME_COL = "GA_DT"
SEGMENT_COL = "SEGMENT_ID"
TARGET_COL = "STATUS"
FEATURE_COLS = [
    "CURRENT1",
    "VOLTAGE",
    "TEMP_CUR",
    "ANALOGUE",
    "GROUND",
]
DEFAULT_LABEL_MAP: dict[str, int] = {}


@dataclass
class WindowedArrays:
    """Materialized sliding-window arrays and their target metadata."""

    X: np.ndarray
    y: np.ndarray
    target_times: np.ndarray
    segment_ids: np.ndarray
    feature_cols: list[str]
    label_map: dict[str, int]

    @property
    def shape_summary(self) -> dict[str, tuple[int, ...]]:
        return {"X": self.X.shape, "y": self.y.shape}


def load_preprocessed_data(data_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the preprocessed minute-level data."""
    data_path = Path(data_path)
    df = pd.read_csv(data_path, parse_dates=[TIME_COL], low_memory=False)

    required = [TIME_COL, SEGMENT_COL, TARGET_COL, *FEATURE_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    if df[required].isna().any().any():
        null_counts = df[required].isna().sum()
        raise ValueError(
            "Input contains missing values:\n"
            + null_counts[null_counts.gt(0)].to_string()
        )

    return df.sort_values([SEGMENT_COL, TIME_COL]).reset_index(drop=True)


def count_windows(
    df: pd.DataFrame,
    sequence_length: int = 60,
    horizon: int = 1,
    stride: int = 5,
) -> int:
    """Return the number of valid windows without materializing them."""
    _validate_window_args(sequence_length, horizon, stride)
    minimum_rows = sequence_length + horizon
    lengths = df.groupby(SEGMENT_COL, sort=False).size().to_numpy()
    valid_lengths = lengths[lengths >= minimum_rows]
    if len(valid_lengths) == 0:
        return 0
    return int(np.sum((valid_lengths - minimum_rows) // stride + 1))


def create_sliding_windows(
    df: pd.DataFrame,
    sequence_length: int = 60, # 시퀀스 길이, 윈도우 사이즈
    horizon: int = 1,
    stride: int = 1,
    feature_cols: list[str] | None = None,
    label_map: Mapping[str, int] | None = None,
    max_windows: int | None = None,
) -> WindowedArrays:
    """Materialize segment-safe windows as compact NumPy arrays.

    X has shape ``(samples, sequence_length, features)`` and y has shape
    ``(samples,)``.  For each window, the target row is
    ``window_end + horizon``.
    """
    _validate_window_args(sequence_length, horizon, stride)
    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be a positive integer or None.")

    feature_cols = list(feature_cols or FEATURE_COLS)
    label_map = dict(label_map or DEFAULT_LABEL_MAP)
    required = [TIME_COL, SEGMENT_COL, TARGET_COL, *feature_cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    unknown_labels = sorted(set(df[TARGET_COL].astype(str)) - set(label_map))
    if label_map and unknown_labels:
        raise ValueError(
            f"Labels missing from label_map: {unknown_labels}. "
            "Pass a label_map containing every STATUS value."
        )

    available = count_windows(df, sequence_length, horizon, stride)
    sample_count = min(available, max_windows) if max_windows else available
    feature_count = len(feature_cols)

    X = np.empty(
        (sample_count, sequence_length, feature_count),
        dtype=np.float32,
    )
    y = np.empty(sample_count, dtype=np.int64)
    target_times = np.empty(sample_count, dtype="datetime64[ns]")
    segment_ids = np.empty(sample_count, dtype=np.int64)

    cursor = 0
    minimum_rows = sequence_length + horizon
    for segment_id, group in df.groupby(SEGMENT_COL, sort=False):
        if len(group) < minimum_rows or cursor >= sample_count:
            continue

        features = group[feature_cols].to_numpy(dtype=np.float32, copy=False)
        if label_map:
            labels = (
                group[TARGET_COL]
                .astype(str)
                .map(label_map)
                .to_numpy(dtype=np.int64, copy=False)
            )
        else:
            labels = np.full(len(group), -1, dtype=np.int64)
        times = group[TIME_COL].to_numpy(dtype="datetime64[ns]", copy=False)

        starts = np.arange(
            0,
            len(group) - minimum_rows + 1,
            stride,
            dtype=np.int64,
        )
        remaining = sample_count - cursor
        starts = starts[:remaining]
        target_indices = starts + sequence_length + horizon - 1

        batch_start = cursor
        cursor += len(starts)
        window_indices = starts[:, None] + np.arange(sequence_length)
        X[batch_start:cursor] = features[window_indices]
        y[batch_start:cursor] = labels[target_indices]
        target_times[batch_start:cursor] = times[target_indices]
        segment_ids[batch_start:cursor] = int(segment_id)

    return WindowedArrays(
        X=X[:cursor],
        y=y[:cursor],
        target_times=target_times[:cursor],
        segment_ids=segment_ids[:cursor],
        feature_cols=feature_cols,
        label_map=label_map,
    )


def to_torch_tensors(
    arrays: WindowedArrays,
    device: str = "cpu",
):
    """Convert NumPy windows to ``torch.float32``/``torch.long`` tensors."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for tensor conversion. "
            "Install it with `pip install torch`."
        ) from exc

    X_tensor = torch.from_numpy(arrays.X)
    y_tensor = torch.from_numpy(arrays.y)
    if device != "cpu":
        X_tensor = X_tensor.to(device)
        y_tensor = y_tensor.to(device)
    return X_tensor, y_tensor


def save_tensors(
    arrays: WindowedArrays,
    output_dir: str | Path = OUTPUT_DIR,
    prefix: str = "sliding_windows",
) -> tuple[Path, Path]:
    """Save tensors (.pt) and reproducibility metadata (.json)."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to save .pt files. "
            "Install it with `pip install torch`."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / f"{prefix}.pt"
    metadata_path = output_dir / f"{prefix}_metadata.json"

    X_tensor, y_tensor = to_torch_tensors(arrays)
    torch.save(
        {
            "X": X_tensor,
            "y": y_tensor,
            "target_times_ns": arrays.target_times.astype(np.int64),
            "segment_ids": torch.from_numpy(arrays.segment_ids),
            "feature_cols": arrays.feature_cols,
            "label_map": arrays.label_map,
        },
        tensor_path,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "X_shape": list(arrays.X.shape),
                "y_shape": list(arrays.y.shape),
                "X_dtype": str(arrays.X.dtype),
                "y_dtype": str(arrays.y.dtype),
                "feature_cols": arrays.feature_cols,
                "label_map": arrays.label_map,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return tensor_path, metadata_path


def _validate_window_args(
    sequence_length: int,
    horizon: int,
    stride: int,
) -> None:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create segment-safe sliding-window tensors."
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap for memory-safe experiments.",
    )
    parser.add_argument("--prefix", default="sliding_windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_preprocessed_data(args.data_path)
    available = count_windows(
        df,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        stride=args.stride,
    )
    estimated_gib = (
        available * args.sequence_length * len(FEATURE_COLS) * 4 / 1024**3
    )
    print(f"Available windows: {available:,}")
    print(f"Estimated X memory (all windows): {estimated_gib:.2f} GiB")

    arrays = create_sliding_windows(
        df,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        stride=args.stride,
        max_windows=args.max_windows,
    )
    print(f"X shape/dtype: {arrays.X.shape} / {arrays.X.dtype}")
    print(f"y shape/dtype: {arrays.y.shape} / {arrays.y.dtype}")

    tensor_path, metadata_path = save_tensors(
        arrays,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    print(f"Tensor saved to: {tensor_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
