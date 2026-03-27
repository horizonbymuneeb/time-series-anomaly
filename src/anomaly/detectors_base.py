"""Common detector contract: per-timestamp scores, a threshold and a boolean mask.

Every detector returns the same :class:`DetectionResult` so precision, recall
and detection delay can be compared without special-casing a detector.
Thresholds are always calibrated on the anomaly-free prefix of the same series,
because a global threshold is unfair when series differ in noise scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    """Scores plus the calibrated threshold that produced the mask."""

    name: str
    scores: np.ndarray
    mask: np.ndarray
    threshold: float
    fp_budget: float

    @property
    def flagged(self) -> int:
        """Number of points above the threshold."""
        return int(self.mask.sum())

    @property
    def score_mean(self) -> float:
        """Mean score over the whole series."""
        return float(np.mean(self.scores))


def validate_series(values: np.ndarray, name: str = "values") -> np.ndarray:
    """Check one-dimensional input and return it as float64."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def sanitize_scores(scores: np.ndarray) -> np.ndarray:
    """Replace non-finite scores with zero so comparisons stay defined."""
    arr = np.asarray(scores, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def calibrate_threshold(clean_scores: np.ndarray, fp_budget: float) -> float:
    """Threshold at the ``1 - fp_budget`` quantile of clean-window scores.

    If the clean window is too short to resolve the requested budget, the
    maximum clean score is used, which is the conservative choice.
    """
    if not 0.0 < fp_budget < 1.0:
        raise ValueError("fp_budget must be in (0, 1)")
    scores = sanitize_scores(clean_scores)
    if scores.size == 0:
        raise ValueError("calibration scores are empty")
    if fp_budget * scores.size < 1.0:
        return float(np.max(scores))
    return float(np.quantile(scores, 1.0 - fp_budget))
