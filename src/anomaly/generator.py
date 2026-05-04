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
