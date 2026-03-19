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
