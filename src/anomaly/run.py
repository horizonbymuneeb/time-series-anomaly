"""CLI: generate labelled series, run the detectors, print and save the report.

    python -m src.anomaly.run --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import describe_config, load_config
from .pipeline import format_report, run_experiment, save_results


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI surface: config path and output directory."""
    parser = argparse.ArgumentParser(description="Run the time-series anomaly benchmark")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--output-dir", default="outputs", help="where results.json is written")
    return parser
