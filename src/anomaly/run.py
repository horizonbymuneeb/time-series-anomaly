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


def main(argv: list[str] | None = None) -> int:
    """Load the config, run all detectors on all series, print and save results."""
    args = build_argument_parser().parse_args(argv)
    cfg = load_config(args.config)
    print(f"config: {describe_config(cfg)}")
    experiment = run_experiment(cfg)
    print(format_report(experiment))
    path = save_results(Path(args.output_dir) / "results.json", experiment)
    print(f"\nresults: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
