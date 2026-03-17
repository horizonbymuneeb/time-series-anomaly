"""Common detector contract: per-timestamp scores, a threshold and a boolean mask.

Every detector returns the same :class:`DetectionResult` so precision, recall
and detection delay can be compared without special-casing a detector.
Thresholds are always calibrated on the anomaly-free prefix of the same series,
because a global threshold is unfair when series differ in noise scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
