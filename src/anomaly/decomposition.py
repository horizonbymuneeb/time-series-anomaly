"""STL-style seasonal decomposition in numpy.

The classical STL loop is: estimate the trend with a moving average, subtract it,
average each seasonal phase, subtract the seasonal term, repeat. Two passes are
enough here because the synthetic seasonalities are additive; the ``robust``
flag adds bisquare reweighting so a spike does not drag the seasonal profile.
Missing values are linearly interpolated first and the missing mask stays
available to callers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
