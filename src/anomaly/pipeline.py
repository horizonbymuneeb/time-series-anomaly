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


def prepare_series(series: AnomalySeries) -> tuple[np.ndarray, np.ndarray]:
    """Return the interpolated series and its anomaly-free calibration prefix."""
    values = fill_missing(series.values)
    calibration = fill_missing(series.calibration_values)
    return values, calibration


def run_detector(name: str, series: AnomalySeries, cfg: Config) -> DetectionResult:
    """Run one named detector on one series."""
    if name not in DETECTORS:
        raise ValueError(f"unknown detector {name!r}; choose from {available_detectors()}")
    values, calibration = prepare_series(series)
    try:
        return DETECTORS[name](values, calibration, cfg.detector)
    except ValueError as error:
        raise ValueError(f"detector {name} failed on series seed {series.seed}: {error}") from error


def run_series(series: AnomalySeries, cfg: Config) -> dict[str, DetectionResult]:
    """Run all detectors on a single series."""
    return {name: run_detector(name, series, cfg) for name in available_detectors()}
