"""Run every detector over every series and collect comparable metrics."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .decomposition import fill_missing
from .detector_ewma import detect_ewma
from .detector_iforest import detect_isolation_forest
from .detector_zscore import detect_rolling_zscore
from .detectors_base import DetectionResult
from .evaluation import compare_table, evaluate_detector
from .generator import AnomalySeries, dataset_summary, generate_dataset

DETECTORS = {
    "rolling_zscore": detect_rolling_zscore,
    "ewma": detect_ewma,
    "isolation_forest": detect_isolation_forest,
}


def available_detectors() -> list[str]:
    """Detector names in a stable order, used by the CLI and the report."""
    return list(DETECTORS)
