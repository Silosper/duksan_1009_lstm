"""Train-only feature scaling with feature-order validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


@dataclass
class FeatureScaler:
    """Serializable sensor scaler that preserves training feature order."""

    method: str
    feature_cols: list[str]
    estimator: Any
    fitted_split: str = "train"

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Scale finite sensor rows and preserve all metadata columns."""
        self.validate_feature_order(list(frame[self.feature_cols].columns))
        transformed = frame.copy()
        values = transformed[self.feature_cols].to_numpy(dtype=float)
        finite_rows = np.isfinite(values).all(axis=1)
        if finite_rows.any():
            finite_values = values[finite_rows]
            if self.estimator is not None:
                finite_values = self.estimator.transform(finite_values)
            if not np.isfinite(finite_values).all():
                raise ValueError("Scaling produced NaN or infinite sensor values.")
            transformed.loc[finite_rows, self.feature_cols] = finite_values
        return transformed

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Restore sensor values to their pre-scaling units."""
        self._validate_array(values)
        if self.estimator is None:
            return np.asarray(values)
        return self.estimator.inverse_transform(values)

    def validate_feature_order(self, feature_cols: list[str]) -> None:
        if list(feature_cols) != self.feature_cols:
            raise ValueError(
                "Feature order mismatch. "
                f"trained={self.feature_cols}, received={list(feature_cols)}"
            )

    def _validate_array(self, values: np.ndarray) -> None:
        if values.ndim != 2 or values.shape[1] != len(self.feature_cols):
            raise ValueError(
                f"Expected a 2D array with {len(self.feature_cols)} features."
            )


def fit_feature_scaler(
    train_frame: pd.DataFrame,
    config: dict[str, Any],
) -> FeatureScaler:
    """Fit the configured scaler using eligible finite train rows only."""
    feature_cols = list(config["data"]["feature_cols"])
    status_col = config["data"]["status_col"]
    status_cfg = config["status"]
    candidates = train_frame
    if bool(status_cfg["use_for_training_filter"]):
        normal_values = {str(value).strip() for value in status_cfg["normal_values"]}
        candidates = candidates.loc[candidates[status_col].isin(normal_values)]

    values = candidates[feature_cols].to_numpy(dtype=float)
    finite_rows = np.isfinite(values).all(axis=1)
    values = values[finite_rows]
    if len(values) == 0:
        raise ValueError("No finite train rows are available for scaler fitting.")

    method = str(config["scaling"]["method"]).lower()
    estimators = {
        "standard": StandardScaler,
        "robust": RobustScaler,
        "minmax": MinMaxScaler,
    }
    estimator = None if method == "none" else estimators[method]()
    if estimator is not None:
        estimator.fit(values)
    print(
        f"Scaler: {method} | fit split=train | "
        f"rows={len(values):,} | features={feature_cols}"
    )
    return FeatureScaler(method, feature_cols, estimator)


def save_feature_scaler(scaler: FeatureScaler, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, destination)
    return destination


def load_feature_scaler(
    path: str | Path,
    expected_feature_cols: list[str],
) -> FeatureScaler:
    scaler = joblib.load(path)
    if not isinstance(scaler, FeatureScaler):
        raise TypeError(
            "The saved scaler predates feature-order metadata. Retrain the model "
            "or provide a FeatureScaler artifact."
        )
    scaler.validate_feature_order(expected_feature_cols)
    return scaler
