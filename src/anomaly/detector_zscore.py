"""Causal rolling z-score detector.

The score of point ``t`` compares it against the mean and standard deviation of
the previous ``window`` points only. That makes the detector usable online and
keeps a spike from lowering its own score through the mean it would otherwise
contribute to.
"""
