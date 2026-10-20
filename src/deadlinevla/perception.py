"""One perception model for everything that "sees": the policy and the governor.

An obstacle exists for a consumer only once some camera, at its current tier, can detect it. Detection
range follows from resolution (pinhole camera, obstacle of size s):

    f_px(r) = W(r) / (2 tan(FOV/2))          focal length in pixels at tier r
    p       = f_px(r) s / d                   pixels the obstacle spans at distance d
    d_det(r) = f_px(r) s / p_min              furthest distance at which it is detected

Example (FOV 90 deg, s = 0.5 m, p_min = 8 px): low tier W = 128 gives 4 m, high tier W = 512 gives 16 m.

``perception.tier_width`` is the sensor width per tier used for detection. It is deliberately separate
from ``cameras.tiers[].res``: the light config renders tiny images for CPU speed, but detection follows
the real tier widths the VLA would see.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from deadlinevla.config import Config


def focal_px(width: float, fov_deg: float) -> float:
    return width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))


def detection_range(tiers: np.ndarray, size: np.ndarray, cfg: Config) -> np.ndarray:
    """d_det for each (tier, size) pair; 0 for tier 0 (camera off)."""
    widths = np.asarray(cfg.perception.tier_width, dtype=np.float64)[np.asarray(tiers, dtype=np.int64)]
    return focal_px(1.0, cfg.cameras.fov) * widths * np.asarray(size, dtype=np.float64) / cfg.perception.p_min


def visible(p_body: np.ndarray, size: np.ndarray, surface_d: np.ndarray, alloc: Sequence[int],
            cfg: Config) -> np.ndarray:
    """Which obstacles some active camera detects at its tier.

    ``p_body`` (N, 2): body-frame offsets to obstacle centres; ``size`` (N,): obstacle size s (diameter);
    ``surface_d`` (N,): distance to the obstacle surface.
    """
    p = np.asarray(p_body, dtype=np.float64).reshape(-1, 2)
    n = len(p)
    seen = np.zeros(n, dtype=bool)
    if n == 0:
        return seen
    dist = np.linalg.norm(p, axis=1)
    bearing = np.arctan2(p[:, 1], p[:, 0])
    ang_r = np.arcsin(np.clip((np.asarray(size) / 2) / np.maximum(dist, 1e-9), 0.0, 1.0))
    half_fov = math.radians(cfg.cameras.fov) / 2
    for yaw, tier in zip(np.radians(np.asarray(cfg.cameras.cam_yaws, dtype=np.float64)), alloc):
        if tier <= 0:
            continue
        off = (bearing - yaw + math.pi) % (2 * math.pi) - math.pi
        in_fov = np.abs(off) <= half_fov + ang_r
        in_range = np.asarray(surface_d) <= detection_range(np.full(n, tier), size, cfg)
        seen |= in_fov & in_range
    return seen
