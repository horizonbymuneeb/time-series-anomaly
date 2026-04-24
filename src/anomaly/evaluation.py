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


def threshold_at_fp_budget(labels: np.ndarray, scores: np.ndarray, fp_budget: float) -> float:
    """Lowest threshold whose flagged points stay within the FP budget.

    The budget is a fraction of all points (not of negatives), matching how alert
    volume is planned in practice.
    """
    y, _ = validate_labels(labels, np.zeros_like(np.asarray(labels)))
    s = np.asarray(scores, dtype=np.float64)
    if not 0.0 < fp_budget < 0.5:
        raise ValueError("fp_budget must be in (0, 0.5)")
    allowed = max(1, int(round(fp_budget * len(y))))
    order = np.argsort(-s, kind="mergesort")
    sorted_scores = s[order]
    sorted_labels = y[order]
    threshold = float(np.nextafter(sorted_scores.max(), np.inf))
    seen_fp = 0
    for score, label in zip(sorted_scores, sorted_labels):
        next_fp = seen_fp + (0 if label else 1)
        if next_fp > allowed:
            break
        threshold = float(score)
        seen_fp = next_fp
    return threshold


def recall_at_fp_budget(labels: np.ndarray, scores: np.ndarray, fp_budget: float) -> dict[str, float]:
    """Recall and precision achievable while respecting ``fp_budget``."""
    y, _ = validate_labels(labels, np.zeros_like(np.asarray(labels)))
    threshold = threshold_at_fp_budget(y, scores, fp_budget)
    flagged = np.asarray(scores) >= threshold
    tp = int((flagged & y).sum())
    fp = int((flagged & ~y).sum())
    positives = int(y.sum())
    return {
        "threshold": threshold,
        "recall": float(tp / positives) if positives else 0.0,
        "precision": float(tp / max(tp + fp, 1)),
        "false_positives": fp,
    }


def region_recall(labels: np.ndarray, mask: np.ndarray) -> float:
    """Share of ground-truth anomaly events with at least one flagged point."""
    y, predicted = validate_labels(labels, mask)
    regions = contiguous_regions(y)
    if not regions:
        return 0.0
    detected = sum(1 for start, end in regions if predicted[start:end].any())
    return float(detected / len(regions))


def detection_delays(labels: np.ndarray, mask: np.ndarray, max_delay: int) -> list[int]:
    """Points between each event start and its first detection (-1 if missed)."""
    y, predicted = validate_labels(labels, mask)
    if max_delay < 1:
        raise ValueError("max_delay must be >= 1")
    delays: list[int] = []
    for start, end in contiguous_regions(y):
        window_end = min(len(predicted), end + max_delay)
        hits = np.flatnonzero(predicted[start:window_end])
        delays.append(int(hits[0]) if hits.size else -1)
    return delays


def mean_delay(delays: list[int]) -> float:
    """Average delay over detected events; NaN when nothing was detected."""
    detected = [d for d in delays if d >= 0]
    return float(np.mean(detected)) if detected else float("nan")
