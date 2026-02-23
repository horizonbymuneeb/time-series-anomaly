"""Each detector must produce aligned scores and a calibrated mask."""

import numpy as np
import pytest

from src.anomaly.detector_ewma import (
    control_band,
    detect_ewma,
    ewma,
    ewma_center,
    ewma_scores,
)
from src.anomaly.detector_iforest import (
    align_window_scores,
    detect_isolation_forest,
    fit_iforest,
    iforest_raw_scores,
    sliding_windows,
)
from src.anomaly.detector_zscore import detect_rolling_zscore, rolling_zscore, trailing_mean_std
from src.anomaly.detectors_base import (
    DetectionResult,
    calibrate_threshold,
    make_result,
    sanitize_scores,
    scores_to_mask,
    summarize_result,
    validate_series,
)
from src.anomaly.decomposition import fill_missing
from src.anomaly.pipeline import run_series
