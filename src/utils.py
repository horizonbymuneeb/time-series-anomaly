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
