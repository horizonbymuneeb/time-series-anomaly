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


def test_detection_result_contract():
    scores = np.array([0.0, 5.0, 1.0, 9.0])
    result = make_result("toy", scores, np.array([0.0, 1.0, 0.5]), 0.2)
    assert isinstance(result, DetectionResult)
    assert result.mask.shape == scores.shape
    assert result.flagged == int(result.mask.sum())
    assert scores_to_mask(scores, 4.0).tolist() == [False, True, False, True]
    summary = summarize_result(result)
    assert summary["flagged"] == result.flagged


def test_calibration_threshold_uses_quantile_and_guards_short_windows():
    clean = np.arange(100, dtype=float)
    threshold = calibrate_threshold(clean, 0.1)
    assert threshold == pytest.approx(np.quantile(clean, 0.9))
    assert calibrate_threshold(np.array([3.0]), 0.01) == 3.0
    with pytest.raises(ValueError):
        calibrate_threshold(clean, 1.0)
    with pytest.raises(ValueError):
        calibrate_threshold(np.array([]), 0.1)
    assert np.allclose(sanitize_scores(np.array([np.nan, np.inf, -np.inf, 2.0])), [0, 0, 0, 2])
    with pytest.raises(ValueError):
        validate_series(np.zeros((2, 2)))


def test_rolling_zscore_is_causal_and_flags_a_spike(tiny_dataset):
    z = rolling_zscore(np.array([1.0, 1.0, 1.0, 10.0, 1.0]), 3)
    assert z.shape == (5,)
    assert z[3] > z[0]
    assert z[3] > 5.0
    means, stds = trailing_mean_std(np.array([0.0, 2.0, 2.0]), 2)
    assert means[2] == pytest.approx(1.0)
    assert stds[2] == pytest.approx(1.0)

    series = tiny_dataset[0]
    result = detect_rolling_zscore(
        fill_missing(series.values), fill_missing(series.calibration_values),
        __import__("src.anomaly.config", fromlist=["DetectorConfig"]).DetectorConfig(fp_budget=0.02),
    )
    assert result.scores.shape == series.values.shape
    assert result.mask.dtype == bool
    assert result.threshold == pytest.approx(
        calibrate_threshold(rolling_zscore(fill_missing(series.calibration_values), 48), 0.02)
    )


def test_ewma_smooths_and_control_band_formula():
    smooth = ewma(np.array([0.0, 10.0, 10.0]), alpha=0.5)
    assert smooth[1] == pytest.approx(5.0)
    assert smooth[2] == pytest.approx(7.5)
    with pytest.raises(ValueError):
        ewma(np.arange(5.0), 1.5)

    clean = np.linspace(0.0, 1.0, 200)
    sigma, band = control_band(clean, 0.2)
    assert sigma > 0 and band == pytest.approx(sigma * np.sqrt(0.2 / 1.8))
    assert ewma_center(np.array([1.0, 2.0, 100.0])) == 2.0
    scores = ewma_scores(np.arange(10.0), center=0.0, alpha=0.3, scale=max(sigma, 1e-6))
    assert (scores >= 0).all()
    with pytest.raises(ValueError):
        ewma_scores(np.arange(10.0), 0.0, 0.3, 0.0)


def test_ewma_detector_flags_a_level_shift(tiny_config, tiny_dataset):
    from src.anomaly.generator import inject_level_shift

    series = tiny_dataset[0]
    clean = fill_missing(series.clean)
    shifted_values = inject_level_shift(clean, 300, 40, 6.0, np.random.default_rng(0))
    calibration = fill_missing(series.calibration_values)
    result = detect_ewma(shifted_values, calibration, tiny_config.detector)
    assert len(result.mask) == series.length
    assert result.threshold > 0
    assert result.mask[300:340].any()
    assert result.flagged > 0


def test_isolation_forest_windows_and_alignment():
    matrix = sliding_windows(np.arange(20, dtype=float), 5)
    assert matrix.shape == (16, 5)
    assert np.allclose(matrix[0], np.arange(5))
    assert np.allclose(matrix[-1], np.arange(15, 20))
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    aligned = align_window_scores(scores, 5, 8)
    assert aligned.shape == (8,)
    assert np.allclose(aligned[4:], scores)
    assert aligned[0] == scores[0]
    with pytest.raises(ValueError):
        sliding_windows(np.arange(3.0), 5)
    with pytest.raises(ValueError):
        align_window_scores(scores, 5, 20)


def test_isolation_forest_detector(tiny_config, tiny_dataset):
    from src.anomaly.config import DetectorConfig

    cfg = DetectorConfig(iforest_window=12, iforest_n_estimators=40, iforest_random_state=3, fp_budget=0.02)
    series = tiny_dataset[0]
    values = fill_missing(series.values)
    calibration = fill_missing(series.calibration_values)
    model = fit_iforest(sliding_windows(calibration, cfg.iforest_window), cfg)
    raw = iforest_raw_scores(model, sliding_windows(values, cfg.iforest_window))
    assert raw.shape[0] == len(values) - cfg.iforest_window + 1
    result = detect_isolation_forest(values, calibration, cfg)
    assert result.scores.shape == values.shape
    assert 0 < result.flagged < len(values)
    with pytest.raises(ValueError):
        fit_iforest(np.zeros((5, 12)), cfg)


def test_every_detector_runs_through_the_registry(tiny_config, tiny_dataset):
    results = run_series(tiny_dataset[1], tiny_config)
    assert set(results) == {"rolling_zscore", "ewma", "isolation_forest"}
    for result in results.values():
        assert result.scores.shape == tiny_dataset[1].values.shape
        assert result.mask.shape == tiny_dataset[1].values.shape
        assert np.isnan(result.scores).sum() == 0
