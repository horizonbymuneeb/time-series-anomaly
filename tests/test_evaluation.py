"""Metric definitions are pinned with hand-built masks."""

import numpy as np
import pandas as pd
import pytest

from src.anomaly.evaluation import (
    aggregate_metrics,
    compare_table,
    contiguous_regions,
    detection_delays,
    evaluate_detector,
    mean_delay,
    point_precision_recall,
    precision_at_k,
    recall_at_fp_budget,
    region_recall,
    threshold_at_fp_budget,
)

LABELS = np.array([0, 1, 1, 0, 0, 1, 0, 0])
MASK = np.array([0, 0, 1, 0, 0, 1, 0, 0])


def test_regions_and_point_metrics():
    assert contiguous_regions(LABELS) == [(1, 3), (5, 6)]
    assert contiguous_regions(np.zeros(4, dtype=bool)) == []
    point = point_precision_recall(LABELS, MASK)
    assert point["precision"] == 1.0
    assert point["recall"] == pytest.approx(2 / 3)
    assert point["tp"] == 2.0 and point["fn"] == 1.0
    with pytest.raises(ValueError):
        point_precision_recall(LABELS, MASK[:3])


def test_precision_at_k():
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.3, 0.7, 0.05, 0.0])
    assert precision_at_k(LABELS, scores, 3) == pytest.approx(1.0)
    assert precision_at_k(LABELS, scores, 5) == pytest.approx(3 / 5)
    assert precision_at_k(LABELS, scores, 8) == pytest.approx(3 / 8)
    with pytest.raises(ValueError):
        precision_at_k(LABELS, scores, 0)
    with pytest.raises(ValueError):
        precision_at_k(LABELS, scores[:2], 1)


def test_threshold_at_fp_budget_and_recall():
    labels = np.array([0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    threshold = threshold_at_fp_budget(labels, scores, 0.25)
    assert threshold == 0.8
    report = recall_at_fp_budget(labels, scores, 0.25)
    assert report["recall"] == 1.0
    assert report["false_positives"] == 1
    with pytest.raises(ValueError):
        threshold_at_fp_budget(labels, scores, 0.9)
