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


def ewma_center(values: np.ndarray) -> float:
    """Robust central level of a clean calibration window."""
    arr = fill_missing(values)
    return float(np.median(arr))


def control_band(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Classic EWMA control-band width around the mean of the input window.

    ``sigma_z = sigma * sqrt(alpha / (2 - alpha))`` is the asymptotic standard
    deviation of the EWMA statistic; the returned half-width uses the robust
    MAD scale instead of the plain standard deviation.
    """
    arr = fill_missing(values)
    sigma = mad_scale(arr)
    band = sigma * np.sqrt(alpha / (2.0 - alpha))
    return sigma, float(band)
