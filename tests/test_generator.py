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


def test_labelled_points_have_kinds_and_events_do_not_overlap(tiny_dataset):
    kinds = set()
    for series in tiny_dataset:
        assert series.labels.sum() > 0
        kinds.update(series.events and [event.kind for event in series.events] or [])
        covered = np.zeros(series.length, dtype=bool)
        for event in series.events:
            assert event.start < event.end <= series.length
            assert not covered[event.start : event.end].any()
            covered[event.start : event.end] = True
            assert series.labels[event.start : event.end].all()
            assert set(series.kinds[event.start : event.end]) == {event.kind}
    assert kinds == {"spike", "level_shift", "variance_change", "missing_block"}


def test_missing_block_is_nan_and_labelled():
    values = np.arange(100, dtype=float)
    out = inject_missing_block(values, 40, 5)
    assert np.isnan(out[40:45]).all()
    assert np.allclose(out[:40], values[:40])
    labels, kinds = label_events(100, [AnomalyEvent("missing_block", 40, 45)], ["missing_block"])
    assert labels[40:45].all() and labels.sum() == 5
    assert set(kinds[40:45]) == {"missing_block"}


def test_injections_are_bounded_to_their_region():
    values = np.zeros(200)
    spike = inject_spike(values, 50, 2, 6.0, np.random.default_rng(0))
    assert np.abs(spike[50:52]).max() > 3.0
    assert np.allclose(spike[:50], 0.0) and np.allclose(spike[52:], 0.0)

    shifted = inject_level_shift(values, 60, 20, 3.0, np.random.default_rng(0))
    assert shifted[60:80].min() > 2.0
    assert np.allclose(shifted[:60], 0.0)

    varied = inject_variance_change(np.ones(200), 70, 10, 3.0, np.random.default_rng(0))
    assert varied[70:80].std() > varied[:60].std()


def test_trend_and_seasonality_components():
    trend = trend_component(10, 0.5)
    assert trend.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    wave = seasonal_component(48, [(24, 2.0)], None)
    assert abs(wave[:24].sum()) < 1e-9
    assert np.isclose(wave[6], 2.0)  # sin(pi/2)
    with pytest.raises(ValueError):
        seasonal_component(10, [])


def test_ar1_noise_has_requested_variance_and_errors():
    rng = np.random.default_rng(3)
    noise = ar1_noise(4000, 1.0, 0.5, rng)
    assert 0.5 < noise.std() < 2.2
    assert np.corrcoef(noise[:-1], noise[1:])[0, 1] > 0.2
    with pytest.raises(ValueError):
        ar1_noise(10, -1.0, 0.2, rng)
    with pytest.raises(ValueError):
        ar1_noise(10, 1.0, 1.5, rng)


def test_dataset_summary_counts_events(tiny_config, tiny_dataset):
    from src.anomaly.generator import dataset_summary

    summary = dataset_summary(tiny_dataset)
    assert summary["series"] == tiny_config.generator.n_series
    assert summary["events"] == sum(len(s.events) for s in tiny_dataset)
    assert 0 < summary["point_rate"] < 0.5
    assert set(summary["events_by_kind"]) <= {"spike", "level_shift", "variance_change", "missing_block"}


def test_base_series_is_smooth_and_calibration_length_guarded(tiny_config):
    rng = np.random.default_rng(0)
    clean = base_series(tiny_config, rng)
    assert clean.shape == (tiny_config.generator.length,)
    assert abs(float(np.mean(np.diff(clean)))) < 5.0
    assert calibration_length(1000, 0.3) == 300
    with pytest.raises(ValueError):
        calibration_length(100, 0.9)
