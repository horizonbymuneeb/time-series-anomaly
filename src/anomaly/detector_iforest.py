"""Isolation-forest baseline on sliding windows.

Isolation forest scores how quickly a point's window can be isolated; windows are
built from the raw series and the model is fitted on calibration windows only.
Scores are aligned to the last point of each window so they line up with the
per-timestamp labels of the other detectors.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from .config import DetectorConfig
from .decomposition import fill_missing
from .detectors_base import DetectionResult, make_result, validate_series


def sliding_windows(values: np.ndarray, window: int) -> np.ndarray:
    """Shape ``(n - window + 1, window)`` view of the filled series."""
    arr = fill_missing(values)
    if window < 2:
        raise ValueError("window must be >= 2")
    if len(arr) < window:
        raise ValueError("series shorter than the window")
    matrix = np.lib.stride_tricks.sliding_window_view(arr, window)
    return np.ascontiguousarray(matrix, dtype=np.float64)


def align_window_scores(window_scores: np.ndarray, window: int, length: int) -> np.ndarray:
    """Map one score per window onto the timestamp at the window's end."""
    scores = np.asarray(window_scores, dtype=np.float64)
    if len(scores) != length - window + 1:
        raise ValueError("window score count does not match the series length")
    out = np.empty(length, dtype=np.float64)
    out[window - 1 :] = scores
    out[: window - 1] = scores[0] if len(scores) else 0.0
    return out


def fit_iforest(matrix: np.ndarray, cfg: DetectorConfig) -> IsolationForest:
    """Fit an isolation forest on calibration windows."""
    if matrix.shape[0] < 10:
        raise ValueError("need at least ten calibration windows")
    model = IsolationForest(
        n_estimators=cfg.iforest_n_estimators,
        contamination="auto",
        random_state=cfg.iforest_random_state,
        n_jobs=1,
    )
    model.fit(matrix)
    return model


def iforest_raw_scores(model: IsolationForest, matrix: np.ndarray) -> np.ndarray:
    """Higher means more anomalous: negate the model's decision function."""
    return -model.decision_function(matrix)


def detect_isolation_forest(
    values: np.ndarray, calibration_values: np.ndarray, cfg: DetectorConfig
) -> DetectionResult:
    """Fit on the clean prefix, score the whole series, calibrate the threshold."""
    arr = validate_series(values)
    calib = validate_series(calibration_values, "calibration_values")
    window = cfg.iforest_window
    if len(calib) <= window + 10:
        raise ValueError("calibration window is too short for the isolation forest")
    model = fit_iforest(sliding_windows(calib, window), cfg)
    matrix = sliding_windows(arr, window)
    raw = iforest_raw_scores(model, matrix)
    calibration_raw = iforest_raw_scores(model, sliding_windows(calib, window))
    align_calibration = align_window_scores(calibration_raw, window, len(calib))
    scores = align_window_scores(raw, window, len(arr))
    return make_result("isolation_forest", scores, align_calibration, cfg.fp_budget)
