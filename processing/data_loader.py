"""Build chronological PyTorch DataLoaders from sliding-window tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_saved_windows(tensor_path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load a file produced by create_sliding_windows.save_tensors."""
    torch = _torch()
    return torch.load(Path(tensor_path), map_location=map_location, weights_only=False)

# 데이터셋 나누기
def chronological_split(
    X,
    y,
    train_ratio: float = 0.7, # 학습 데이터 비율
    validation_ratio: float = 0.15, # 검증 데이터 비율
    purge_gap: int | None = None,
):
    """Split X/y by time order and return train, validation, test tuples.

    ``purge_gap`` discards samples at each split boundary.  With overlapping
    sliding windows, set it high enough that train/validation/test inputs do
    not share the same minute-level observations.
    """
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("Split ratios must be between 0 and 1.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be < 1.")
    if purge_gap is None:
        purge_gap = int(X.shape[1]) if getattr(X, "ndim", 0) >= 2 else 0
    if purge_gap < 0:
        raise ValueError("purge_gap must be zero or a positive integer.")

    n = len(X)
    train_end = int(n * train_ratio)
    validation_start = train_end + purge_gap
    validation_end = validation_start + int(n * validation_ratio)
    test_start = validation_end + purge_gap
    if min(train_end, validation_end - validation_start, n - test_start) == 0:
        raise ValueError("Each split must contain at least one sample.")

    return (
        (X[:train_end], y[:train_end]),
        (X[validation_start:validation_end], y[validation_start:validation_end]),
        (X[test_start:], y[test_start:]),
    )


def make_dataloaders(
    X,
    y,
    batch_size: int = 64, # 배치 크기, 윈도우 개수/배치 사이즈 = 배치의 수
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    purge_gap: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """Create train/validation/test DataLoaders from tensor or NumPy data."""
    torch = _torch()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    X_tensor = _as_tensor(X, torch.float32, torch)
    y_tensor = _as_tensor(y, torch.long, torch)
    if X_tensor.ndim != 3:
        raise ValueError("X must have shape (samples, sequence_length, features).")
    if y_tensor.ndim != 1:
        raise ValueError("y must have shape (samples,).")

    splits = chronological_split(
        X_tensor,
        y_tensor,
        train_ratio,
        validation_ratio,
        purge_gap,
    )
    loaders = {}
    for name, (X_part, y_part) in zip(
        ("train", "validation", "test"), splits
    ):
        dataset = torch.utils.data.TensorDataset(X_part, y_part)
        loaders[name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders


def _as_tensor(value, dtype, torch):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(np.asarray(value), dtype=dtype)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for DataLoader creation. "
            "Install it with `pip install torch`."
        ) from exc
    return torch
