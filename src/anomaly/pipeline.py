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


def metrics_row(series: AnomalySeries, name: str, result: DetectionResult, cfg: Config) -> dict[str, float]:
    """Evaluate one detector result against the ground-truth labels."""
    row = evaluate_detector(
        series.labels,
        result.scores,
        result.mask,
        cfg.evaluation.precision_k,
        cfg.detector.delay_window,
    )
    row["series"] = float(series.seed)
    row["detector"] = name  # type: ignore[assignment]
    row["flagged"] = float(result.flagged)
    return row


def run_experiment(cfg: Config) -> dict[str, object]:
    """Generate the dataset, run every detector and aggregate the metrics."""
    dataset = generate_dataset(cfg)
    per_series_metrics: dict[str, list[dict[str, float]]] = {name: [] for name in available_detectors()}
    rows: list[dict[str, object]] = []
    for series in dataset:
        results = run_series(series, cfg)
        for name, result in results.items():
            row = metrics_row(series, name, result, cfg)
            per_series_metrics[name].append(row)  # type: ignore[arg-type]
            rows.append({"series": series.seed, "detector": name, **_strip(row)})
    comparison = compare_table(per_series_metrics)
    return {
        "config": cfg,
        "dataset_summary": dataset_summary(dataset),
        "per_detector": {name: rows_for for name, rows_for in per_series_metrics.items()},
        "per_series": rows,
        "comparison": comparison,
    }


def _strip(row: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in row.items() if key not in {"series", "detector"}}


def save_results(path: str | Path, experiment: dict[str, object]) -> Path:
    """Persist summary, per-series metrics and the comparison table as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison: pd.DataFrame = experiment["comparison"]  # type: ignore[assignment]
    payload = {
        "dataset_summary": experiment["dataset_summary"],
        "comparison": comparison.reset_index().to_dict(orient="records"),
        "per_series": experiment["per_series"],
        "detectors": available_detectors(),
    }
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return path


def load_results(path: str | Path) -> dict[str, object]:
    """Read a JSON report written by :func:`save_results`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"results not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def format_report(experiment: dict[str, object]) -> str:
    """Console table: one row per detector with the headline metrics."""
    comparison: pd.DataFrame = experiment["comparison"]  # type: ignore[assignment]
    summary = experiment["dataset_summary"]
    lines = [
        f"dataset: {summary['series']} series, {summary['points']} points, "
        f"{summary['events']} events ({summary['point_rate']:.2%} anomalous points)",
        "",
        comparison.round(3).to_string(),
    ]
    return "\n".join(lines)
