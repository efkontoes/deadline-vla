"""Method step 1: the latency deadline implied by speed and clearance.

During one inference the drone flies blind for v*L metres, then needs v^2/(2a) to stop. Safe iff

    v L + v^2 / (2a) <= d - m      =>      L_max(v, d) = (d - m - v^2/(2a)) / v
"""

from __future__ import annotations

import math

SPEED_EPS = 1e-6


def max_latency(v: float, d: float, a: float, m: float) -> float:
    """L_max in seconds.

    Returns ``inf`` when ``v`` is (near) zero: a hovering drone has no deadline. Returns a negative
    number when even instant inference is unsafe; the caller treats that as "must slow down now".
    """
    if a <= 0:
        raise ValueError("braking deceleration a must be positive")
    v = abs(v)
    if v < SPEED_EPS:
        return math.inf
    return (d - m - v * v / (2.0 * a)) / v


def violates(v: float, L: float, d: float, a: float, m: float) -> bool:
    """True when the safety inequality v L + v^2/(2a) <= d - m does not hold."""
    v = abs(v)
    return v * L + v * v / (2.0 * a) > d - m
