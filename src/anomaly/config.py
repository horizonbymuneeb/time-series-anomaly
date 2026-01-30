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


def _int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != int(value):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    return int(value)


def _float(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number, got {value!r}")
    return float(value)


def build_config(raw: Mapping[str, Any]) -> Config:
    """Parse and validate a raw mapping into a :class:`Config`."""
    gen = require_section(raw, "generator")
    det = require_section(raw, "detector")
    ev = require_section(raw, "evaluation")
    anomaly_types = tuple(str(t) for t in gen.get("anomaly_types", ANOMALY_TYPES))
    cfg = Config(
        seed=_int(raw.get("seed", 0), "seed"),
        generator=GeneratorConfig(
            n_series=_int(gen.get("n_series", 12), "generator.n_series"),
            length=_int(gen.get("length", 1500), "generator.length"),
            period=_int(gen.get("period", 24), "generator.period"),
            weekly_period=_int(gen.get("weekly_period", 168), "generator.weekly_period"),
            trend_slope=_float(gen.get("trend_slope", 0.0006), "generator.trend_slope"),
            noise_std=_float(gen.get("noise_std", 1.0), "generator.noise_std"),
            ar_coefficient=_float(gen.get("ar_coefficient", 0.35), "generator.ar_coefficient"),
            calibration_fraction=_float(
                gen.get("calibration_fraction", 0.3), "generator.calibration_fraction"
            ),
            anomaly_rate=_float(gen.get("anomaly_rate", 0.05), "generator.anomaly_rate"),
            anomaly_types=anomaly_types,
            spike_magnitude=_float(gen.get("spike_magnitude", 6.0), "generator.spike_magnitude"),
            level_shift_offset=_float(
                gen.get("level_shift_offset", 3.0), "generator.level_shift_offset"
            ),
            variance_factor=_float(gen.get("variance_factor", 3.5), "generator.variance_factor"),
            missing_value=_float(gen.get("missing_value", 0.0), "generator.missing_value"),
        ),
        detector=DetectorConfig(
            zscore_window=_int(det.get("zscore_window", 48), "detector.zscore_window"),
            zscore_threshold_sigma=_float(
                det.get("zscore_threshold_sigma", 4.0), "detector.zscore_threshold_sigma"
            ),
            ewma_alpha=_float(det.get("ewma_alpha", 0.15), "detector.ewma_alpha"),
            ewma_sigma=_float(det.get("ewma_sigma", 2.5), "detector.ewma_sigma"),
            iforest_window=_int(det.get("iforest_window", 24), "detector.iforest_window"),
            iforest_n_estimators=_int(
                det.get("iforest_n_estimators", 120), "detector.iforest_n_estimators"
            ),
            iforest_random_state=_int(
                det.get("iforest_random_state", 7), "detector.iforest_random_state"
            ),
            fp_budget=_float(det.get("fp_budget", 0.01), "detector.fp_budget"),
            delay_window=_int(det.get("delay_window", 12), "detector.delay_window"),
        ),
        evaluation=EvaluationConfig(
            precision_k=_int(ev.get("precision_k", 50), "evaluation.precision_k")
        ),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    """Collect every out-of-range value and raise once."""
    problems: list[str] = []
    g, d = cfg.generator, cfg.detector
    if g.n_series < 1:
        problems.append("generator.n_series must be >= 1")
    if g.length < SERIES_MIN_LENGTH:
        problems.append(f"generator.length must be >= {SERIES_MIN_LENGTH}")
    if g.period < 2 or g.weekly_period < g.period:
        problems.append("generator.period/weekly_period are inconsistent")
    if not 0.0 < g.calibration_fraction < 0.8:
        problems.append("generator.calibration_fraction must be in (0, 0.8)")
    if not 0.0 < g.anomaly_rate < 0.5:
        problems.append("generator.anomaly_rate must be in (0, 0.5)")
    if not set(g.anomaly_types) <= set(ANOMALY_TYPES):
        problems.append(f"generator.anomaly_types must be within {ANOMALY_TYPES}")
    if g.noise_std <= 0:
        problems.append("generator.noise_std must be positive")
    if not 0.0 <= g.ar_coefficient < 0.95:
        problems.append("generator.ar_coefficient must be in [0, 0.95)")
    if g.spike_magnitude <= 0 or g.level_shift_offset == 0:
        problems.append("anomaly magnitudes must be non-zero")
    if g.variance_factor <= 1.0:
        problems.append("generator.variance_factor must be > 1")
    if d.zscore_window < 4:
        problems.append("detector.zscore_window must be >= 4")
    if d.zscore_threshold_sigma <= 0 or d.ewma_sigma <= 0:
        problems.append("threshold sigmas must be positive")
    if not 0.0 < d.ewma_alpha <= 1.0:
        problems.append("detector.ewma_alpha must be in (0, 1]")
    if d.iforest_window < 2 or d.iforest_n_estimators < 10:
        problems.append("iforest window/estimator settings are too small")
    if not 0.0 < d.fp_budget < 0.5:
        problems.append("detector.fp_budget must be in (0, 0.5)")
    if d.delay_window < 1:
        problems.append("detector.delay_window must be >= 1")
    if cfg.evaluation.precision_k < 1:
        problems.append("evaluation.precision_k must be >= 1")
    if problems:
        raise ValueError("invalid config: " + "; ".join(problems))
