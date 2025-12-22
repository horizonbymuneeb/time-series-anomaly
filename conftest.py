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
