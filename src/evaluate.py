from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    delta = pred - truth
    penalties = np.where(
        delta < 0,
        np.exp(-delta / 13.0) - 1.0,
        np.exp(delta / 10.0) - 1.0,
    )
    return float(np.sum(penalties))


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    abs_error = np.abs(truth - pred)
    return {
        "rmse": rmse(truth, pred),
        "mae": float(np.mean(abs_error)),
        "nasa_score": nasa_score(truth, pred),
        "bias": float(np.mean(pred - truth)),
    }
