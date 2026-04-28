"""Labelled synthetic series: trend + two seasonalities + AR(1) noise + anomalies.

Each series keeps its first ``calibration_fraction`` of points anomaly-free so a
detector can calibrate its threshold on genuinely clean data. Anomalies are then
injected at random, non-overlapping positions and every one is labelled with its
type and its exact start/end, which is what makes detection delay measurable.
"""
