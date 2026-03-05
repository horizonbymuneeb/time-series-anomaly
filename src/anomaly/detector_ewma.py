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


def ewma_scores(values: np.ndarray, center: float, alpha: float, scale: float) -> np.ndarray:
    """Absolute deviation of the EWMA level from ``center``, in scale units."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    return np.abs(ewma(values, alpha) - center) / scale


def detect_ewma(
    values: np.ndarray, calibration_values: np.ndarray, cfg: DetectorConfig
) -> DetectionResult:
    """Score with EWMA and calibrate the threshold on the clean prefix."""
    arr = validate_series(values)
    calib = validate_series(calibration_values, "calibration_values")
    center = ewma_center(calib)
    sigma = mad_scale(calib)
    scale = max(sigma, 1e-6)
    scores = ewma_scores(arr, center, cfg.ewma_alpha, scale)
    calib_scores = sanitize_scores(ewma_scores(calib, center, cfg.ewma_alpha, scale))
    return make_result("ewma", scores, calib_scores, cfg.fp_budget)
