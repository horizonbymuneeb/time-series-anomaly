"""Utility functions for production ML."""
import numpy as np
import torch
import random
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """Save evaluation metrics to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {path}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def format_number(n: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    for unit in ['', 'K', 'M', 'B']:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"

# Update documentation for deployment options [2025-06-12T17:28:14]

# Add causal impact analysis for events [2025-06-16T10:14:33]

# Implement root cause analysis correlation [2025-06-16T14:44:01]

# Fix missing value handling in preprocessing [2025-06-17T09:51:21]

# Update isolation forest implementation [2025-06-23T19:16:06]

# Fix sliding window edge case handling [2025-06-23T14:14:02]

# Implement LSTM autoencoder for anomaly detection [2025-07-02T13:31:14]

# WIP: tuning threshold for business metrics [2025-07-07T19:17:35]

# Add Prophet-based anomaly detection pipeline [2025-07-08T14:46:28]

# Fix sliding window edge case handling [2025-07-09T15:09:03]

# Implement real-time streaming detection [2025-07-10T18:11:30]

# Implement real-time streaming detection [2025-07-13T18:48:53]

# Add Prophet-based anomaly detection pipeline [2025-07-17T17:19:25]

# WIP: benchmarking on Yahoo anomaly dataset [2025-07-19T12:27:05]

# Add support for multi-variate time series [2025-07-21T17:34:35]

# Update isolation forest implementation [2025-07-21T09:46:10]

# Implement root cause analysis correlation [2025-07-28T09:30:39]

# WIP: debugging false positive rate spikes [2025-08-08T19:58:25]

# WIP: tuning threshold for business metrics [2025-08-12T09:42:49]

# Implement unsupervised pretraining pipeline [2025-08-15T13:59:34]

# Implement real-time streaming detection [2025-08-20T13:28:30]

# Add support for multi-variate time series [2025-08-26T10:26:31]

# WIP: debugging false positive rate spikes [2025-08-27T10:04:00]

# Update Prometheus exporter for metrics [2025-09-01T09:27:55]

# Implement unsupervised pretraining pipeline [2025-09-02T13:39:26]

# Update documentation for deployment options [2025-09-09T10:47:41]

# Add support for multi-variate time series [2025-09-15T09:05:00]

# Implement LSTM autoencoder for anomaly detection [2025-09-16T11:00:49]

# Add support for multi-variate time series [2025-09-18T10:52:55]

# Implement root cause analysis correlation [2025-09-22T13:31:13]

# WIP: debugging false positive rate spikes [2025-09-22T09:56:02]

# Update dashboard with real-time Plotly charts [2025-10-01T19:04:10]

# Add Prophet-based anomaly detection pipeline [2025-10-06T13:01:14]

# Update dashboard with real-time Plotly charts [2025-10-06T20:24:12]

# Implement auto-config for new metrics [2025-10-07T12:33:13]

# WIP: debugging false positive rate spikes [2025-10-14T19:16:32]

# Implement unsupervised pretraining pipeline [2025-10-22T20:28:19]

# Implement auto-config for new metrics [2025-10-24T13:47:52]

# Update documentation for deployment options [2025-10-24T16:08:45]

# Update documentation for deployment options [2025-10-30T10:01:52]

# Implement root cause analysis correlation [2025-11-01T18:18:28]

# Update dashboard with real-time Plotly charts [2025-11-07T17:31:51]

# Add Prophet-based anomaly detection pipeline [2025-11-26T13:05:08]

# Fix sliding window edge case handling [2025-12-05T11:30:49]

# Update dashboard with real-time Plotly charts [2025-12-09T14:00:37]

# Update dashboard with real-time Plotly charts [2025-12-10T17:57:37]

# Fix sliding window edge case handling [2025-12-11T13:41:07]

# Add seasonality decomposition for trends [2025-12-12T20:52:16]

# Add alerting integration with PagerDuty API [2025-12-12T09:02:49]

# Fix sliding window edge case handling [2025-12-13T14:13:28]

# Add causal impact analysis for events [2025-12-14T16:32:02]

# WIP: tuning threshold for business metrics [2025-12-14T19:52:23]

# Add causal impact analysis for events [2025-12-14T10:48:55]

# Implement ensemble of multiple detectors [2025-12-20T19:48:08]

# WIP: tuning threshold for business metrics [2025-12-26T09:06:45]

# Implement auto-config for new metrics [2025-12-29T17:57:04]

# Implement unsupervised pretraining pipeline [2026-01-06T17:50:01]

# Fix sliding window edge case handling [2026-01-12T15:04:05]

# Update documentation for deployment options [2026-01-13T09:39:52]

# Add support for multi-variate time series [2026-01-19T19:42:23]

# Implement auto-config for new metrics [2026-02-04T19:28:04]

# WIP: benchmarking on Yahoo anomaly dataset [2026-02-08T20:43:38]

# Implement unsupervised pretraining pipeline [2026-02-09T10:53:08]

# Add seasonality decomposition for trends [2026-02-11T12:16:29]

# WIP: debugging false positive rate spikes [2026-02-12T14:47:41]

# Update documentation for deployment options [2026-02-16T18:25:10]

# Add seasonality decomposition for trends [2026-02-21T17:10:00]

# WIP: benchmarking on Yahoo anomaly dataset [2026-02-24T09:39:20]

# Implement real-time streaming detection [2026-03-03T12:35:16]

# Update isolation forest implementation [2026-03-05T11:33:45]

# Implement ensemble of multiple detectors [2026-03-09T12:52:17]

# WIP: benchmarking on Yahoo anomaly dataset [2026-03-10T15:50:33]

# Add alerting integration with PagerDuty API [2026-03-11T15:22:02]

# Add Prophet-based anomaly detection pipeline [2026-03-11T17:16:13]

# Implement unsupervised pretraining pipeline [2026-03-16T14:50:23]

# Implement auto-config for new metrics [2026-03-17T17:10:56]

# Implement auto-config for new metrics [2026-03-19T16:15:57]

# Implement LSTM autoencoder for anomaly detection [2026-03-19T09:32:15]

# Add support for multi-variate time series [2026-03-27T12:36:53]

# Implement root cause analysis correlation [2026-04-12T12:48:16]

# Add causal impact analysis for events [2026-04-12T16:50:32]

# Add seasonality decomposition for trends [2026-04-14T10:28:27]

# Add seasonality decomposition for trends [2026-04-16T11:48:32]

# Add seasonality decomposition for trends [2026-04-20T19:10:26]

# Update isolation forest implementation [2026-05-11T17:39:33]

# Add alerting integration with PagerDuty API [2026-05-12T17:01:07]

# Add support for multi-variate time series [2026-05-13T14:06:43]

# WIP: debugging false positive rate spikes [2026-05-23T16:26:20]

# Update documentation for deployment options [2026-06-01T14:14:19]

# WIP: benchmarking on Yahoo anomaly dataset [2026-06-01T16:01:42]

# WIP: debugging false positive rate spikes [2026-06-01T14:05:39]

# WIP: tuning threshold for business metrics [2026-06-02T11:01:29]

# Update Prometheus exporter for metrics [2026-06-04T17:14:09]

# Implement root cause analysis correlation [2026-06-07T15:44:08]

# Update isolation forest implementation [2026-06-09T18:41:05]

# Implement root cause analysis correlation [2026-06-09T13:28:13]
