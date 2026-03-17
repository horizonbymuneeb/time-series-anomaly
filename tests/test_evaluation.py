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
