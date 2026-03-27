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


def test_region_recall_and_detection_delays():
    assert region_recall(LABELS, MASK) == 1.0
    missed = np.zeros_like(MASK)
    missed[5] = 1
    assert region_recall(LABELS, missed) == 0.5
    assert region_recall(np.zeros(4, dtype=bool), np.zeros(4, dtype=bool)) == 0.0
    delays = detection_delays(LABELS, MASK, max_delay=3)
    assert delays == [1, 0]
    late = np.zeros_like(MASK)
    late[6] = 1
    assert detection_delays(LABELS, late, max_delay=3) == [-1, 1]
    assert mean_delay([1, 0, 2]) == 1.0
    assert np.isnan(mean_delay([-1]))
    with pytest.raises(ValueError):
        detection_delays(LABELS, MASK, max_delay=0)


def test_evaluate_detector_and_aggregate():
    scores = np.linspace(1.0, 0.0, len(LABELS))
    row = evaluate_detector(LABELS, scores, MASK, precision_k=2, max_delay=3)
    assert row["precision"] == 1.0
    assert row["recall"] == pytest.approx(2 / 3)
    assert row["detected_events"] == 2.0
    assert row["events"] == 2.0
    assert row["mean_delay"] == pytest.approx(0.5)
    assert "precision_at_k" in row

    aggregate = aggregate_metrics([row, row])
    assert aggregate["series"] == 2.0
    assert aggregate["recall"] == pytest.approx(row["recall"])
    with pytest.raises(ValueError):
        aggregate_metrics([])


def test_compare_table_sorts_by_f1():
    good = {
        "precision": 1.0, "recall": 1.0, "f1": 1.0, "precision_at_k": 1.0,
        "region_recall": 1.0, "mean_delay": 0.0, "events": 2.0, "series": 1.0,
    }
    weak = {**good, "precision": 0.2, "recall": 0.2, "f1": 0.2, "mean_delay": 5.0}
    table = compare_table({"good": [good], "weak": [weak]})
    assert isinstance(table, pd.DataFrame)
    assert list(table.index) == ["good", "weak"]
    assert {"precision_at_k", "f1", "region_recall", "mean_delay"} <= set(table.columns)
