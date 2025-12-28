"""Make ``src`` importable and share a small, realistic dataset."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly.config import load_config  # noqa: E402
from src.anomaly.generator import generate_dataset  # noqa: E402

REAL_CONFIG = ROOT / "config.yaml"
TINY_OVERRIDES = {
    "seed": 5,
    "generator": {"n_series": 3, "length": 600, "anomaly_rate": 0.04},
    "detector": {
        "zscore_window": 24,
        "iforest_window": 12,
        "iforest_n_estimators": 40,
        "delay_window": 8,
        "fp_budget": 0.02,
    },
    "evaluation": {"precision_k": 30},
}


@pytest.fixture(scope="session")
def tiny_config():
    """Same code path as the real config, small enough for a fast suite."""
    return load_config(REAL_CONFIG, TINY_OVERRIDES)


@pytest.fixture(scope="session")
def tiny_dataset(tiny_config):
    """Three labelled series with every anomaly type."""
    return generate_dataset(tiny_config)
