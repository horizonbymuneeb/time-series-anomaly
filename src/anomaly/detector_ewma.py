"""EWMA control-chart detector.

The exponentially weighted moving average tracks the local level; the score is
the deviation of that level from the calibration centre, divided by a robust
scale. This catches sustained level shifts that a single-point z-score misses.
"""
