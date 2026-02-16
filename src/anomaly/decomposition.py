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


def seasonal_means(detrended: np.ndarray, period: int) -> np.ndarray:
    """Average detrended value per seasonal phase, centred to sum to zero."""
    if period < 2:
        raise ValueError("period must be >= 2")
    arr = np.asarray(detrended, dtype=np.float64)
    sums = np.zeros(period, dtype=np.float64)
    counts = np.zeros(period, dtype=np.float64)
    for phase in range(period):
        block = arr[phase::period]
        sums[phase] = np.nansum(block)
        counts[phase] = np.isfinite(block).sum()
    means = sums / np.maximum(counts, 1)
    return means - means.mean()


def seasonal_component(detrended: np.ndarray, period: int) -> np.ndarray:
    """Expand phase means back to the full length."""
    return np.resize(seasonal_means(detrended, period), len(detrended))


def stl_decompose(values: np.ndarray, period: int, iterations: int = 2, robust: bool = False) -> Decomposition:
    """Iterative additive decomposition: observed = trend + seasonal + residual."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    observed = fill_missing(values)
    trend = trend_component(observed, period, robust=robust)
    seasonal = np.zeros_like(observed)
    for _ in range(iterations):
        detrended = observed - trend
        seasonal = seasonal_component(detrended, period)
        trend = trend_component(observed - seasonal, period, robust=robust)
    residual = observed - trend - seasonal
    return Decomposition(observed=observed, trend=trend, seasonal=seasonal, residual=residual)


def mad_scale(values: np.ndarray) -> float:
    """Robust scale estimate: 1.4826 * median absolute deviation."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("no finite values")
    median = np.median(finite)
    return float(1.4826 * np.median(np.abs(finite - median)))
