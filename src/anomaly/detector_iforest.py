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
