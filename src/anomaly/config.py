"""Typed access to ``config.yaml``; every value is validated at load time."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
ANOMALY_TYPES: tuple[str, ...] = ("spike", "level_shift", "variance_change", "missing_block")
SERIES_MIN_LENGTH = 100


@dataclass(frozen=True)
class GeneratorConfig:
    """Shape of the synthetic series and the anomalies injected into them."""

    n_series: int = 12
    length: int = 1500
    period: int = 24
    weekly_period: int = 168
    trend_slope: float = 0.0006
    noise_std: float = 1.0
    ar_coefficient: float = 0.35
    calibration_fraction: float = 0.30
    anomaly_rate: float = 0.05
    anomaly_types: tuple[str, ...] = ANOMALY_TYPES
    spike_magnitude: float = 6.0
    level_shift_offset: float = 3.0
    variance_factor: float = 3.5
    missing_value: float = 0.0


@dataclass(frozen=True)
class DetectorConfig:
    """Detector hyperparameters and threshold calibration settings."""

    zscore_window: int = 48
    zscore_threshold_sigma: float = 4.0
    ewma_alpha: float = 0.15
    ewma_sigma: float = 2.5
    iforest_window: int = 24
    iforest_n_estimators: int = 120
    iforest_random_state: int = 7
    fp_budget: float = 0.01
    delay_window: int = 12


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation knobs that are not detector-specific."""

    precision_k: int = 50


@dataclass(frozen=True)
class Config:
    """Full experiment configuration."""

    seed: int
    generator: GeneratorConfig
    detector: DetectorConfig
    evaluation: EvaluationConfig


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping, raising ``FileNotFoundError`` when absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    return raw


def require_section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one config section, failing with the section name."""
    if name not in raw:
        raise ValueError(f"config is missing required section: {name!r}")
    section = raw[name]
    if not isinstance(section, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return section
