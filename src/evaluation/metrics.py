"""Forecast evaluation metrics, shared across all models for fair comparison.

Every model predicts the next-day log return; regression quality is measured with
MSE/MAE/RMSE and directional quality with directional accuracy and F1 on the sign
of the prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.asarray(y_true) - np.asarray(y_pred)
    mse = float(np.mean(err ** 2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(err)))}


def direction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    up_true = (np.asarray(y_true) > 0).astype(int)
    up_pred = (np.asarray(y_pred) > 0).astype(int)
    return {
        "directional_accuracy": float(np.mean(up_true == up_pred)),
        "f1": float(f1_score(up_true, up_pred, zero_division=0)),
    }


def evaluate(y_true, y_pred) -> dict:
    """All metrics in one dict (regression + direction)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {**regression_metrics(y_true, y_pred), **direction_metrics(y_true, y_pred),
            "n": int(len(y_true))}


def metrics_frame(rows: list[dict]) -> pd.DataFrame:
    """Tidy a list of metric dicts (each with model/asset keys) into a frame."""
    return pd.DataFrame(rows)
