"""Detection metrics that respect the shape of the problem.

Point-level precision/recall alone reward a detector that flags half a long
level shift; a fraud-style queue owner instead cares whether each *event* was
caught and how many points late. This module reports both, plus precision@k
and recall at a fixed false-positive budget so detectors with different score
scales stay comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
