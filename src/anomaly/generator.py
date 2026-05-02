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
