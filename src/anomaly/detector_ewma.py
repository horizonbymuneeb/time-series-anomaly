"""EWMA control-chart detector.

The exponentially weighted moving average tracks the local level; the score is
the deviation of that level from the calibration centre, divided by a robust
scale. This catches sustained level shifts that a single-point z-score misses.
"""
from __future__ import annotations

import numpy as np

from .config import DetectorConfig
from .decomposition import fill_missing, mad_scale
from .detectors_base import DetectionResult, make_result, sanitize_scores, validate_series


def ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponentially weighted moving average with ``alpha`` in (0, 1]."""
    arr = fill_missing(values)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out
