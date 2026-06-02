"""Labelled synthetic series: trend + two seasonalities + AR(1) noise + anomalies.

Each series keeps its first ``calibration_fraction`` of points anomaly-free so a
detector can calibrate its threshold on genuinely clean data. Anomalies are then
injected at random, non-overlapping positions and every one is labelled with its
type and its exact start/end, which is what makes detection delay measurable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import ANOMALY_TYPES, Config


@dataclass(frozen=True)
class AnomalyEvent:
    """One injected anomaly with its ground-truth extent."""

    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class AnomalySeries:
    """A generated series plus labels needed for evaluation."""

    values: np.ndarray
    clean: np.ndarray
    labels: np.ndarray
    kinds: np.ndarray
    events: tuple[AnomalyEvent, ...]
    calibration_end: int
    seed: int

    @property
    def length(self) -> int:
        """Number of observations."""
        return int(self.values.shape[0])

    @property
    def calibration_values(self) -> np.ndarray:
        """Anomaly-free prefix used for threshold selection."""
        return self.values[: self.calibration_end]

    @property
    def evaluation_values(self) -> np.ndarray:
        """Everything from the calibration boundary onwards."""
        return self.values[self.calibration_end :]


def trend_component(length: int, slope: float) -> np.ndarray:
    """Linear drift; small relative to the seasonal amplitude."""
    if length < 1:
        raise ValueError("length must be positive")
    return slope * np.arange(length, dtype=np.float64)


def seasonal_component(
    length: int,
    periods: Sequence[tuple[int, float]],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sum of sine waves; optional per-series phase jitter keeps series distinct."""
    if not periods:
        raise ValueError("at least one (period, amplitude) pair is required")
    t = np.arange(length, dtype=np.float64)
    component = np.zeros(length, dtype=np.float64)
    for period, amplitude in periods:
        if period < 2 or amplitude < 0:
            raise ValueError("period must be >= 2 and amplitude >= 0")
        phase = 0.0 if rng is None else float(rng.uniform(0, 2 * np.pi))
        component += amplitude * np.sin(2 * np.pi * t / period + phase)
    return component


def ar1_noise(length: int, std: float, coefficient: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) noise so residual variance is not i.i.d. white noise."""
    if length < 1:
        raise ValueError("length must be positive")
    if std <= 0:
        raise ValueError("std must be positive")
    if not 0.0 <= coefficient < 1.0:
        raise ValueError("coefficient must be in [0, 1)")
    innovations = rng.normal(0.0, std, size=length)
    noise = np.empty(length, dtype=np.float64)
    noise[0] = innovations[0]
    for i in range(1, length):
        noise[i] = coefficient * noise[i - 1] + innovations[i]
    return noise


def base_series(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Clean series: trend + daily and weekly cycles + AR(1) noise."""
    g = cfg.generator
    periods = ((g.period, 2.5), (g.weekly_period, 1.4), (max(g.period // 2, 2), 0.6))
    return (
        trend_component(g.length, g.trend_slope)
        + seasonal_component(g.length, periods, rng)
        + ar1_noise(g.length, g.noise_std, g.ar_coefficient, rng)
    )


def inject_spike(values: np.ndarray, start: int, width: int, magnitude: float, rng: np.random.Generator) -> np.ndarray:
    """Short additive excursion, positive or negative."""
    out = values.copy()
    sign = 1.0 if rng.random() < 0.6 else -1.0
    out[start : start + width] += sign * magnitude * float(rng.uniform(0.8, 1.3))
    return out


def inject_level_shift(
    values: np.ndarray, start: int, duration: int, offset: float, rng: np.random.Generator
) -> np.ndarray:
    """Step change that persists for ``duration`` points."""
    out = values.copy()
    out[start : start + duration] += offset * float(rng.uniform(0.8, 1.2))
    return out


def inject_variance_change(
    values: np.ndarray, start: int, duration: int, factor: float, rng: np.random.Generator
) -> np.ndarray:
    """Stretch or compress the local deviation from its pre-window mean."""
    out = values.copy()
    local = float(np.mean(values[max(0, start - 10) : start])) if start > 0 else float(values[0])
    region = slice(start, start + duration)
    out[region] = local + (values[region] - local) * factor + rng.normal(0.0, 0.4, size=duration)
    return out


def inject_missing_block(values: np.ndarray, start: int, duration: int, fill: float = np.nan) -> np.ndarray:
    """Replace a segment with a sentinel; the label still marks it anomalous."""
    out = values.copy()
    out[start : start + duration] = fill
    return out


def injection_lengths(cfg: Config) -> dict[str, tuple[int, int]]:
    """Minimum and maximum duration (in points) for each anomaly type."""
    g = cfg.generator
    return {
        "spike": (1, 3),
        "level_shift": (g.period // 2 + 2, g.period * 2),
        "variance_change": (g.period // 2 + 2, g.period * 2),
        "missing_block": (2, max(3, g.period // 4)),
    }


def choose_events(
    length: int, calibration_end: int, rate: float, kinds: Sequence[str], rng: np.random.Generator
) -> list[AnomalyEvent]:
    """Pick non-overlapping labelled regions after the calibration prefix."""
    usable = length - calibration_end
    if usable <= 0:
        raise ValueError("calibration window leaves no evaluation points")
    budget = max(1, int(round(usable * rate)))
    lens = injection_lengths_for(kinds)
    events: list[AnomalyEvent] = []
    occupied = np.zeros(length, dtype=bool)
    attempts = 0
    while len(events) < budget and attempts < budget * 40:
        attempts += 1
        kind = str(rng.choice(list(kinds)))
        low, high = lens[kind]
        width = int(rng.integers(low, high + 1))
        start = int(rng.integers(calibration_end, max(calibration_end + 1, length - width)))
        end = min(length, start + width)
        margin = 2
        if occupied[max(0, start - margin) : min(length, end + margin)].any():
            continue
        occupied[start:end] = True
        events.append(AnomalyEvent(kind=kind, start=start, end=end))
    return sorted(events, key=lambda event: event.start)


def injection_lengths_for(kinds: Sequence[str]) -> dict[str, tuple[int, int]]:
    """Duration ranges for a subset of anomaly kinds, independent of config."""
    defaults = {
        "spike": (1, 3),
        "level_shift": (14, 48),
        "variance_change": (14, 48),
        "missing_block": (2, 6),
    }
    unknown = set(kinds) - set(defaults)
    if unknown:
        raise ValueError(f"unknown anomaly kinds: {sorted(unknown)}")
    return {kind: defaults[kind] for kind in kinds}


def apply_events(
    values: np.ndarray, events: Sequence[AnomalyEvent], cfg: Config, rng: np.random.Generator
) -> np.ndarray:
    """Apply each event to a copy of the clean series."""
    g = cfg.generator
    out = values.copy()
    for event in events:
        duration = event.end - event.start
        if event.kind == "spike":
            out = inject_spike(out, event.start, duration, g.spike_magnitude, rng)
        elif event.kind == "level_shift":
            out = inject_level_shift(out, event.start, duration, g.level_shift_offset, rng)
        elif event.kind == "variance_change":
            out = inject_variance_change(out, event.start, duration, g.variance_factor, rng)
        elif event.kind == "missing_block":
            out = inject_missing_block(out, event.start, duration, np.nan)
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"unsupported anomaly kind: {event.kind}")
    return out


def label_events(length: int, events: Sequence[AnomalyEvent], kinds: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Boolean labels and per-point kind labels from the event list."""
    labels = np.zeros(length, dtype=bool)
    kind_labels = np.full(length, "", dtype=object)
    for event in events:
        labels[event.start : event.end] = True
        kind_labels[event.start : event.end] = event.kind
    return labels, kind_labels


def calibration_length(length: int, fraction: float) -> int:
    """Number of leading points reserved for the clean calibration window."""
    if not 0.0 < fraction < 0.8:
        raise ValueError("fraction must be in (0, 0.8)")
    return max(10, int(round(length * fraction)))


def generate_series(cfg: Config, seed: int) -> AnomalySeries:
    """Generate one labelled series from a seed."""
    rng = np.random.default_rng(seed)
    clean = base_series(cfg, rng)
    cal_end = calibration_length(cfg.generator.length, cfg.generator.calibration_fraction)
    events = choose_events(
        cfg.generator.length, cal_end, cfg.generator.anomaly_rate,
        cfg.generator.anomaly_types, rng,
    )
    values = apply_events(clean, events, cfg, rng)
    labels, kinds = label_events(cfg.generator.length, events, cfg.generator.anomaly_types)
    return AnomalySeries(
        values=values,
        clean=clean,
        labels=labels,
        kinds=kinds,
        events=tuple(events),
        calibration_end=cal_end,
        seed=seed,
    )


def generate_dataset(cfg: Config) -> list[AnomalySeries]:
    """One series per configured series count, seeded from ``cfg.seed``."""
    return [generate_series(cfg, cfg.seed + i) for i in range(cfg.generator.n_series)]


def dataset_summary(series: Sequence[AnomalySeries]) -> dict[str, object]:
    """Counts used by the CLI and the README."""
    if not series:
        raise ValueError("empty dataset")
    total_points = sum(s.length for s in series)
    anomalous = sum(int(s.labels.sum()) for s in series)
    by_kind: dict[str, int] = {}
    for s in series:
        for event in s.events:
            by_kind[event.kind] = by_kind.get(event.kind, 0) + 1
    return {
        "series": len(series),
        "points": total_points,
        "anomalous_points": anomalous,
        "point_rate": anomalous / total_points,
        "events": sum(len(s.events) for s in series),
        "events_by_kind": by_kind,
    }
