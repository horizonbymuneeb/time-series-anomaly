"""Detection metrics that respect the shape of the problem.

Point-level precision/recall alone reward a detector that flags half a long
level shift; a fraud-style queue owner instead cares whether each *event* was
caught and how many points late. This module reports both, plus precision@k
and recall at a fixed false-positive budget so detectors with different score
scales stay comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_labels(labels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Check that labels and a detector mask align and are boolean."""
    y = np.asarray(labels)
    predicted = np.asarray(mask)
    if y.ndim != 1 or predicted.ndim != 1:
        raise ValueError("labels and mask must be one-dimensional")
    if len(y) != len(predicted):
        raise ValueError("labels and mask must have the same length")
    return y.astype(bool), predicted.astype(bool)


def contiguous_regions(labels: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` spans of True values."""
    y = np.asarray(labels).astype(bool)
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(y):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(y)))
    return regions
