"""Geometric radio horizon with an explicit earth factor.

The historical M0 helper uses 3570 * (sqrt(h1) + sqrt(h2)). That coefficient
matches the k=1 geometric horizon and remains unchanged in the M0 module.
"""
from __future__ import annotations

import math

EARTH_RADIUS_M = 6371000.0
PRIMARY_K = 4.0 / 3.0
REFERENCE_K = 1.0


def geometric_horizon_m(height_a_m: float, height_b_m: float, k: float) -> float:
    """d = sqrt(2*k*R*h1) + sqrt(2*k*R*h2), with every term in metres."""
    if float(k) <= 0.0 or float(height_a_m) < 0.0 or float(height_b_m) < 0.0:
        raise ValueError("horizon arguments")
    radius = EARTH_RADIUS_M
    factor = 2.0 * float(k) * radius
    return math.sqrt(factor * float(height_a_m)) + math.sqrt(factor * float(height_b_m))
