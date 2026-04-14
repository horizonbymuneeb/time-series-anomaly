"""The synthetic data is the ground truth for every metric, so pin its behaviour."""

import numpy as np
import pytest

from src.anomaly.generator import (
    AnomalyEvent,
    ar1_noise,
    base_series,
    calibration_length,
    generate_dataset,
    generate_series,
    inject_level_shift,
    inject_missing_block,
    inject_spike,
    inject_variance_change,
    label_events,
    seasonal_component,
    trend_component,
)
