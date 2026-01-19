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


def test_invalid_values_are_rejected():
    raw = read_yaml(DEFAULT_CONFIG_PATH)
    raw["generator"]["calibration_fraction"] = 0.95
    with pytest.raises(ValueError, match="calibration_fraction"):
        build_config(raw)

    raw = read_yaml(DEFAULT_CONFIG_PATH)
    raw["detector"]["ewma_alpha"] = 0.0
    with pytest.raises(ValueError, match="ewma_alpha"):
        build_config(raw)

    raw = read_yaml(DEFAULT_CONFIG_PATH)
    raw["generator"]["anomaly_types"] = ["spike", "teleport"]
    with pytest.raises(ValueError, match="anomaly_types"):
        build_config(raw)


def test_short_series_and_zero_noise_rejected():
    raw = read_yaml(DEFAULT_CONFIG_PATH)
    raw["generator"]["length"] = 20
    raw["generator"]["noise_std"] = 0.0
    with pytest.raises(ValueError) as error:
        build_config(raw)
    message = str(error.value)
    assert "length" in message and "noise_std" in message


def test_require_section_and_missing_file(tmp_path):
    with pytest.raises(ValueError, match="mapping"):
        require_section({"evaluation": 3}, "evaluation")
    with pytest.raises(FileNotFoundError):
        read_yaml(tmp_path / "absent.yaml")
