"""STL-style decomposition: exact on synthetic components, robust to gaps."""

import numpy as np
import pytest

from src.anomaly.decomposition import (
    bisquare_weights,
    centered_moving_average,
    fill_missing,
    mad_scale,
    missing_mask,
    residual_scores,
    seasonal_component,
    seasonal_means,
    seasonal_strength,
    stl_decompose,
    trend_component,
    trend_strength,
)
