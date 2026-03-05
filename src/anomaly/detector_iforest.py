"""Isolation-forest baseline on sliding windows.

Isolation forest scores how quickly a point's window can be isolated; windows are
built from the raw series and the model is fitted on calibration windows only.
Scores are aligned to the last point of each window so they line up with the
per-timestamp labels of the other detectors.
"""
