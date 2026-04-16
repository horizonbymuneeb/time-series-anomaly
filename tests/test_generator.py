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


def test_series_are_deterministic_per_seed(tiny_config):
    a = generate_series(tiny_config, 11)
    b = generate_series(tiny_config, 11)
    assert np.allclose(a.values, b.values, equal_nan=True)
    assert (a.labels == b.labels).all()
    assert a.events == b.events


def test_different_seeds_change_the_series(tiny_config):
    a = generate_series(tiny_config, 1)
    b = generate_series(tiny_config, 2)
    assert not np.allclose(a.values, b.values, equal_nan=True)


def test_calibration_prefix_is_clean(tiny_config, tiny_dataset):
    for series in tiny_dataset:
        prefix = series.calibration_end
        assert prefix == calibration_length(tiny_config.generator.length, tiny_config.generator.calibration_fraction)
        assert not series.labels[:prefix].any()
        assert all(event.start >= prefix for event in series.events)
