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
