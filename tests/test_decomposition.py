"""STL-style decomposition: exact on synthetic components, robust to gaps."""

import numpy as np
import pytest

from src.anomaly.decomposition import (
    bisquare_weights,
    centered_moving_average,
    fill_missing,
    mad_scale,
    missing_mask,
    residual_scores,
    seasonal_component,
    seasonal_means,
    seasonal_strength,
    stl_decompose,
    trend_component,
    trend_strength,
)


def test_fill_missing_interpolates_and_keeps_edges():
    values = np.array([1.0, np.nan, np.nan, 4.0])
    filled = fill_missing(values)
    assert np.allclose(filled, [1.0, 2.0, 3.0, 4.0])
    assert missing_mask(values).tolist() == [False, True, True, False]
    with pytest.raises(ValueError):
        fill_missing(np.array([np.nan, np.nan]))


def test_centered_moving_average_preserves_constants_and_ramps():
    constant = np.full(50, 7.0)
    assert np.allclose(centered_moving_average(constant, 5), 7.0)
    ramp = np.arange(50, dtype=float)
    smoothed = centered_moving_average(ramp, 5)
    assert np.allclose(smoothed, ramp, atol=6.0)
    with pytest.raises(ValueError):
        centered_moving_average(ramp, 0)


def test_trend_of_a_linear_ramp():
    ramp = 0.5 * np.arange(240, dtype=float)
    trend = trend_component(ramp, 24)
    interior = slice(30, -30)
    assert np.allclose(trend[interior], ramp[interior], atol=0.5)
    with pytest.raises(ValueError):
        trend_component(ramp, 1)
