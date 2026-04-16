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


def point_precision_recall(labels: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Point-level precision, recall and F1."""
    y, predicted = validate_labels(labels, mask)
    tp = int((y & predicted).sum())
    fp = int((~y & predicted).sum())
    fn = int((y & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of the k highest-scoring points that are labelled anomalous."""
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) != len(s):
        raise ValueError("labels and scores must have the same length")
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, len(y))
    top = np.argsort(-s, kind="mergesort")[:k]
    return float(y[top].mean())
