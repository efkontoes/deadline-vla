"""The simulator contract every backend implements, plus braking measurement.

Observation dict (one environment)
    images      (V, R, R, 3) uint8   six RGB views at the highest tier resolution R
    depth       (V, R, R) float32    distance to image plane, m (used by the depth clearance ablation)
    state       (7,) float64         STATE_KEYS below
    vel_body    (3,) float64         body-frame linear velocity
    instruction str
    info        dict                 collision, goal_reached, clearance (oracle d), done, ...
    privileged  dict                 ground truth for the expert and the surrogate policy only

Actions are body-frame velocity commands ``float[4] = (vx, vy, vz, yaw_rate)``; a standard
controller inside the env tracks them.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import numpy as np

from deadlinevla.config import Config
from deadlinevla.sim.scenes import Scene, empty_scene

STATE_KEYS = ("vx_b", "vy_b", "vz_b", "yaw_rate", "roll", "pitch", "z")
ACTION_DIM = 4


class DroneEnv(Protocol):
    cfg: Config

    def reset(self, scene: Scene) -> dict[str, Any]: ...

    def step(self, v_cmd: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def clearance(self, v_hat: Optional[np.ndarray], cone_deg: float) -> float: ...


def make_env(cfg: Config) -> DroneEnv:
    """Build the configured backend: ``light`` (numpy) or ``isaac`` (Isaac Lab 3.0)."""
    if cfg.sim.backend == "light":
        from deadlinevla.sim.light_env import LightDroneEnv

        return LightDroneEnv(cfg)
    if cfg.sim.backend == "isaac":
        from deadlinevla.sim.isaac_env import IsaacDroneEnv

        return IsaacDroneEnv(cfg)
    raise ValueError(f"unknown sim.backend {cfg.sim.backend!r}")


def measure_braking_details(env: DroneEnv, speeds=(2.0, 4.0, 6.0, 8.0), settle_s: float = 3.0) -> dict:
    """Fly straight at each speed, command zero velocity, record the stopping distance.

    Returns the conservative effective deceleration ``a = min_v v^2 / (2 s(v))`` (so the braking
    model v^2/(2a) upper-bounds every measured stop, controller lag included) and the raw data.
    """
    ctrl_dt = 1.0 / env.cfg.drone.ctrl_hz
    rows = []
    for v in speeds:
        obs = env.reset(empty_scene(altitude=env.cfg.sim.altitude))
        for _ in range(int(settle_s / ctrl_dt)):
            obs, _ = env.step(np.array([v, 0.0, 0.0, 0.0]))
        v0 = float(np.linalg.norm(obs["vel_body"]))
        p0 = np.array(obs["privileged"]["pos"])
        for _ in range(int(10.0 / ctrl_dt)):
            obs, _ = env.step(np.zeros(ACTION_DIM))
            if np.linalg.norm(obs["vel_body"]) < 0.02:
                break
        s = float(np.linalg.norm(np.array(obs["privileged"]["pos"])[:2] - p0[:2]))
        rows.append({"v": v0, "stop_dist": s, "a_eff": v0 * v0 / (2 * max(s, 1e-6))})
    v = np.array([r["v"] for r in rows])
    s = np.array([r["stop_dist"] for r in rows])
    # Also report the two-parameter fit s = v^2/(2a) + v*t_lag for the record.
    X = np.column_stack([v ** 2 / 2, v])
    (inv_a, t_lag), *_ = np.linalg.lstsq(X, s, rcond=None)
    return {"a_brake": float(min(r["a_eff"] for r in rows)), "fit_a": float(1 / inv_a) if inv_a > 0 else None,
            "fit_lag_s": float(t_lag), "runs": rows}


def measure_braking(env: DroneEnv) -> float:
    """Measured braking deceleration a (m/s^2). Nothing downstream uses a hand-typed value."""
    return measure_braking_details(env)["a_brake"]
