"""The experiment runner produces one comparable row per detector per series."""

import json
from pathlib import Path

import numpy as np

from src.anomaly.pipeline import (
    available_detectors,
    format_report,
    load_results,
    metrics_row,
    run_detector,
    run_experiment,
    run_series,
    save_results,
)
from src.anomaly.run import main as run_main
