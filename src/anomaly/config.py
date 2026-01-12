"""Typed access to ``config.yaml``; every value is validated at load time."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
ANOMALY_TYPES: tuple[str, ...] = ("spike", "level_shift", "variance_change", "missing_block")
