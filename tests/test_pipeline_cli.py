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


def test_available_detectors_and_unknown_name(tiny_config, tiny_dataset):
    assert available_detectors() == ["rolling_zscore", "ewma", "isolation_forest"]
    try:
        run_detector("nope", tiny_dataset[0], tiny_config)
    except ValueError as error:
        assert "unknown detector" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_run_experiment_shapes(tiny_config):
    experiment = run_experiment(tiny_config)
    comparison = experiment["comparison"]
    assert len(comparison) == 3
    assert set(comparison.index) == set(available_detectors())
    assert list(comparison.index) == list(comparison.sort_values("f1", ascending=False).index)
    rows = experiment["per_series"]
    assert len(rows) == 3 * tiny_config.generator.n_series
    for row in rows:
        assert {"series", "detector", "precision", "recall", "precision_at_k"} <= set(row)
    summary = experiment["dataset_summary"]
    assert summary["series"] == tiny_config.generator.n_series


def test_metrics_row_matches_detector_result(tiny_config, tiny_dataset):
    series = tiny_dataset[0]
    result = run_detector("rolling_zscore", series, tiny_config)
    row = metrics_row(series, "rolling_zscore", result, tiny_config)
    assert row["detector"] == "rolling_zscore"
    assert 0.0 <= row["precision"] <= 1.0
    assert 0.0 <= row["recall"] <= 1.0
    assert row["flagged"] == result.flagged


def test_save_load_and_report(tiny_config, tmp_path):
    experiment = run_experiment(tiny_config)
    path = save_results(tmp_path / "nested" / "results.json", experiment)
    payload = load_results(path)
    assert payload["detectors"] == available_detectors()
    assert len(payload["comparison"]) == 3
    assert payload["dataset_summary"]["series"] == tiny_config.generator.n_series
    report = format_report(experiment)
    assert "rolling_zscore" in report and "isolation_forest" in report
    assert "events" in report
    import pytest

    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "missing.json")


def test_cli_writes_results(tiny_config, tmp_path):
    config_path = tmp_path / "config.yaml"
    import yaml

    from src.anomaly.config import DEFAULT_CONFIG_PATH, read_yaml

    raw = read_yaml(DEFAULT_CONFIG_PATH)
    raw.update({"seed": 3, "generator": {**raw["generator"], "n_series": 2, "length": 400, "anomaly_rate": 0.04},
                "detector": {**raw["detector"], "iforest_n_estimators": 30, "iforest_window": 10}})
    config_path.write_text(yaml.safe_dump(raw))
    out_dir = tmp_path / "outputs"
    assert run_main(["--config", str(config_path), "--output-dir", str(out_dir)]) == 0
    payload = json.loads((out_dir / "results.json").read_text())
    assert len(payload["comparison"]) == 3
    assert np.isfinite([row["f1"] for row in payload["comparison"]]).all()
