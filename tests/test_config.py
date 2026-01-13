"""Config values are validated before any detector runs."""

import pytest

from src.anomaly.config import (
    ANOMALY_TYPES,
    DEFAULT_CONFIG_PATH,
    build_config,
    config_to_dict,
    describe_config,
    load_config,
    read_yaml,
    require_section,
    validate_config,
)


def test_default_config_loads_with_expected_defaults():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.generator.length >= 100
    assert cfg.generator.period == 24
    assert cfg.generator.weekly_period == 168
    assert set(cfg.generator.anomaly_types) == set(ANOMALY_TYPES)
    assert 0 < cfg.detector.fp_budget < 1


def test_overrides_apply_per_section():
    cfg = load_config(DEFAULT_CONFIG_PATH, {"generator": {"length": 400}, "detector": {"ewma_alpha": 0.4}})
    assert cfg.generator.length == 400
    assert cfg.detector.ewma_alpha == 0.4


def test_missing_section_is_named():
    with pytest.raises(ValueError, match="detector"):
        build_config({"seed": 1, "generator": {}, "evaluation": {}})
