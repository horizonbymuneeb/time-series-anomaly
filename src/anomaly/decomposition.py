"""STL-style seasonal decomposition in numpy.

The classical STL loop is: estimate the trend with a moving average, subtract it,
average each seasonal phase, subtract the seasonal term, repeat. Two passes are
enough here because the synthetic seasonalities are additive; the ``robust``
flag adds bisquare reweighting so a spike does not drag the seasonal profile.
Missing values are linearly interpolated first and the missing mask stays
available to callers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Decomposition:
    """Result of :func:`stl_decompose`; all arrays have the input length."""

    observed: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray


def missing_mask(values: np.ndarray) -> np.ndarray:
    """True where the input is NaN or non-finite."""
    arr = np.asarray(values, dtype=np.float64)
    return ~np.isfinite(arr)


def fill_missing(values: np.ndarray) -> np.ndarray:
    """Linear interpolation of internal gaps; edges repeat the nearest value."""
    arr = np.asarray(values, dtype=np.float64).copy()
    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")
    mask = missing_mask(arr)
    if not mask.any():
        return arr
    if mask.all():
        raise ValueError("cannot interpolate a series that is entirely missing")
    idx = np.arange(len(arr))
    arr[mask] = np.interp(idx[mask], idx[~mask], arr[~mask])
    return arr


def centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with reflect padding at both edges."""
    if window < 1:
        raise ValueError("window must be >= 1")
    arr = fill_missing(values)
    if window == 1:
        return arr.copy()
    pad = window // 2
    padded = np.pad(arr, pad, mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(arr)]


def trend_component(values: np.ndarray, period: int, robust: bool = False) -> np.ndarray:
    """Trend from one or two moving-average passes, depending on the period parity."""
    if period < 2:
        raise ValueError("period must be >= 2")
    arr = fill_missing(values)
    window = period if period % 2 == 1 else period + 1
    trend = centered_moving_average(arr, window)
    if robust:
        residual = arr - trend
        weights = bisquare_weights(residual)
        trend = centered_moving_average(arr * weights, window) / np.maximum(
            centered_moving_average(weights, window), 1e-12
        )
    return trend
