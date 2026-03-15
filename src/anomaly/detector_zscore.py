"""Causal rolling z-score detector.

The score of point ``t`` compares it against the mean and standard deviation of
the previous ``window`` points only. That makes the detector usable online and
keeps a spike from lowering its own score through the mean it would otherwise
contribute to.
"""
from __future__ import annotations

import numpy as np

from .config import DetectorConfig
from .decomposition import fill_missing
from .detectors_base import DetectionResult, make_result, validate_series


def trailing_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the ``window`` points before each index.

    The first ``window`` positions fall back to an expanding window so every
    point receives a score; standard deviation is floored to avoid division by
    zero on flat stretches.
    """
    arr = fill_missing(values)
    if window < 2:
        raise ValueError("window must be >= 2")
    means = np.empty_like(arr)
    stds = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window)
        block = arr[start:i]
        if block.size == 0:
            means[i], stds[i] = arr[i], 0.0
            continue
        means[i] = float(np.mean(block))
        stds[i] = float(np.std(block))
    return means, stds


def rolling_zscore(values: np.ndarray, window: int, floor: float = 1e-6) -> np.ndarray:
    """Absolute, causal z-score of every point against its trailing window."""
    arr = validate_series(values)
    if floor <= 0:
        raise ValueError("floor must be positive")
    means, stds = trailing_mean_std(arr, window)
    return np.abs(arr - means) / np.maximum(stds, floor)


def detect_rolling_zscore(
    values: np.ndarray, calibration_values: np.ndarray, cfg: DetectorConfig
) -> DetectionResult:
    """Score the series and calibrate the threshold on its clean prefix."""
    arr = validate_series(values)
    calib = validate_series(calibration_values, "calibration_values")
    scores = rolling_zscore(arr, cfg.zscore_window)
    calib_scores = rolling_zscore(calib, cfg.zscore_window)
    return make_result("rolling_zscore", scores, calib_scores, cfg.fp_budget)


def score_summary(scores: np.ndarray) -> dict[str, float]:
    """Distribution summary used in the report."""
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }
