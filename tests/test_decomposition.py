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


def test_seasonal_means_sum_to_zero_and_expand():
    period = 24
    t = np.arange(period * 6, dtype=float)
    wave = 3.0 * np.sin(2 * np.pi * t / period)
    means = seasonal_means(wave, period)
    assert abs(means.sum()) < 1e-9
    assert np.isclose(means.max(), 3.0, atol=0.05)
    expanded = seasonal_component(wave, period)
    assert expanded.shape == wave.shape
    assert np.isclose(expanded[0], means[0])


def test_decomposition_reconstructs_the_observed_series():
    rng = np.random.default_rng(4)
    t = np.arange(720, dtype=float)
    observed = (
        0.002 * t
        + 2.0 * np.sin(2 * np.pi * t / 24)
        + 1.0 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 0.3, size=len(t))
    )
    decomposition = stl_decompose(observed, period=24, iterations=2)
    reconstructed = decomposition.trend + decomposition.seasonal + decomposition.residual
    assert np.allclose(reconstructed, observed, atol=1e-9)
    assert decomposition.observed.shape == observed.shape
    assert seasonal_strength(decomposition) > 0.5
    assert 0.0 <= trend_strength(decomposition) <= 1.0
    with pytest.raises(ValueError):
        stl_decompose(observed, 24, iterations=0)


def test_robust_decomposition_ignores_a_spike():
    t = np.arange(480, dtype=float)
    observed = 2.0 * np.sin(2 * np.pi * t / 24)
    spiked = observed.copy()
    spiked[100:102] += 40.0
    plain = stl_decompose(spiked, 24, robust=False)
    robust = stl_decompose(spiked, 24, robust=True)
    assert abs(plain.seasonal[100] - robust.seasonal[100]) >= 0.0
    assert robust.residual[100] > 20.0


def test_mad_scale_and_bisquare_weights():
    assert mad_scale(np.array([1.0, 2.0, 3.0, 4.0, 100.0])) == pytest.approx(1.4826 * 1.0)
    with pytest.raises(ValueError):
        mad_scale(np.array([np.nan]))
    weights = bisquare_weights(np.array([0.0, 1.0, 100.0]))
    assert np.isclose(weights[0], 1.0)
    assert weights[-1] < 0.01
    assert ((weights >= 0) & (weights <= 1)).all()


def test_residual_scores_are_positive_and_scale_free():
    scores = residual_scores(np.arange(240, dtype=float) % 24, period=24)
    assert scores.shape == (240,)
    assert (scores >= 0).all()
    assert np.isfinite(scores).all()
