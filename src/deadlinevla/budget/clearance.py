"""Method step 7: clearance d to the nearest obstacle inside a cone around v_hat.

``oracle`` (v1) reads the simulator's ground truth; ``depth`` (v2) takes the minimum range in the cone
from the depth images. Same interface, so the ablation swaps one config field.
"""

from __future__ import annotations

import functools
from typing import Any, Optional, Sequence

import numpy as np

from deadlinevla.config import Config


@functools.lru_cache(maxsize=16)
def pixel_rays(res: int, fov_deg: float, cam_yaws: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Unit body-frame ray per pixel, (V, H, W, 3), and the depth-to-range factor, (H, W).

    Pinhole camera, square image, horizontal fov; image x to the right, y down; body x forward, y
    left, z up.
    """
    f = (res / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    c = (np.arange(res) + 0.5 - res / 2.0) / f
    xn = c[None, :].repeat(res, 0)            # right
    yn = -c[:, None].repeat(res, 1)           # up
    norm = np.sqrt(1.0 + xn ** 2 + yn ** 2)
    rays = []
    for yaw in np.radians(cam_yaws):
        cy, sy = np.cos(yaw), np.sin(yaw)
        # camera forward = (cy, sy, 0); camera right = (sy, -cy, 0); up = z
        d = np.stack([cy + xn * sy, sy - xn * cy, yn], axis=-1) / norm[..., None]
        rays.append(d)
    out = np.stack(rays)
    out.setflags(write=False)
    norm.setflags(write=False)
    return out, norm


def depth_clearance(depth: np.ndarray, v_hat: Optional[np.ndarray], z: float, fov_deg: float,
                    cam_yaws: Sequence[float], cone_deg: float, radius: float, max_range: float,
                    floor_ignore_z: float = 0.2) -> float:
    """Minimum range to a non-floor point inside the cone, minus the airframe radius.

    ``depth`` is (V, H, W) distance to image plane. ``v_hat`` is a body-frame unit vector; ``None``
    means hovering, in which case every direction counts.
    """
    V, H, W = depth.shape
    rays, norm = pixel_rays(H, float(fov_deg), tuple(float(y) for y in cam_yaws))
    rng = depth * norm[None]
    valid = (depth < max_range * 0.999) & (z + rng * rays[..., 2] > floor_ignore_z)
    if v_hat is not None:
        cosang = rays @ np.asarray(v_hat, dtype=np.float64)
        valid &= cosang >= np.cos(np.radians(cone_deg))
    if not valid.any():
        return float(max_range)
    return float(max(rng[valid].min() - radius, 0.0))


def estimate_clearance(obs: dict[str, Any], v_hat: Optional[np.ndarray], cfg: Config) -> float:
    """Clearance d in metres for the configured ``budget.clearance_source``.

    ``v_hat`` is a body-frame unit direction (``None`` = hovering, every direction counts). The
    oracle evaluates the simulator's ground-truth geometry in that direction when the observation
    carries it, and otherwise falls back to ``info.clearance`` (computed along the current velocity).
    """
    src = cfg.budget.clearance_source
    if src == "oracle":
        priv = obs.get("privileged")
        if priv is None:
            return float(obs["info"]["clearance"])
        from deadlinevla.sim.scenes import cone_clearance

        v_w = None
        if v_hat is not None:
            c, s = np.cos(priv["yaw"]), np.sin(priv["yaw"])
            v_w = np.array([c * v_hat[0] - s * v_hat[1], s * v_hat[0] + c * v_hat[1], 0.0])
        return cone_clearance(priv["scene"], priv["pos"], priv["time"], v_w, cfg.budget.cone_deg,
                              cfg.drone.radius, cfg.sim.max_range)
    if src == "depth":
        return depth_clearance(np.asarray(obs["depth"]), v_hat, float(obs["state"][6]), cfg.cameras.fov,
                               cfg.cameras.cam_yaws, cfg.budget.cone_deg, cfg.drone.radius,
                               cfg.sim.max_range, cfg.budget.floor_ignore_z)
    raise ValueError(f"unknown clearance_source {src!r}")
