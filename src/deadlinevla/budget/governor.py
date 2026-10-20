"""Method step 2: the fastest safe speed for a latency, and the speed governor.

The deadline inequality read the other way, multiplied by 2a, is a quadratic in v:

    v^2 + 2aL v - 2a(d - m) = 0      =>      v*(L, d) = -aL + sqrt(a^2 L^2 + 2a(d - m))
"""

from __future__ import annotations

import math

import numpy as np

from deadlinevla.config import Config, a_brake


def safe_speed(L: float, d: float, a: float, m: float) -> float:
    """v* in m/s; zero if the clearance is at or below the margin."""
    room = d - m
    if room <= 0:
        return 0.0
    if not math.isfinite(L):
        return 0.0
    L = max(L, 0.0)
    aL = a * L
    return -aL + math.sqrt(aL * aL + 2.0 * a * room)


def govern(v_cmd: np.ndarray, L_alloc: float, d: float, cfg: Config) -> np.ndarray:
    """Rescale the commanded linear velocity so its norm is at most min(v*, v_max).

    Direction and yaw rate (last component) are kept. ``d`` should be the clearance in the direction
    of the command, so a drone inside the margin can still be commanded away from the obstacle.
    """
    v_star = safe_speed(L_alloc, d, a_brake(cfg), cfg.drone.margin)
    return cap_speed(v_cmd, min(v_star, cfg.drone.v_max))


def cap_speed(v_cmd: np.ndarray, cap: float) -> np.ndarray:
    out = np.array(v_cmd, dtype=np.float64, copy=True)
    n = float(np.linalg.norm(out[:3]))
    if n > cap:
        out[:3] *= cap / n if n > 0 else 0.0
    return out
