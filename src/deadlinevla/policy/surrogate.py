"""Perception-limited surrogate policy for studying the deadline module before a real VLA exists.

It reads ground truth, but only what the allocation would let a camera see: an obstacle is detected
if it lies inside an active camera's horizontal field of view and within that camera's tier range
(``model.surrogate_ranges``). It then steers reactively toward the goal (the language goal is
assumed grounded) around detected obstacles at the cruise speed. Undetected obstacles are invisible,
so allocation decides what can be avoided, and latency decides how stale the decision is.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from deadlinevla.config import Config
from deadlinevla.policy.base import PolicyAdapter, PolicySpec

N_HEADINGS = 72


class SurrogatePolicy(PolicyAdapter):
    name = "surrogate"
    trainable = False
    checkpoint_name = "policy.txt"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.H = cfg.model.chunk_size or 5
        self.K = cfg.model.num_denoise_steps or 10
        self.ranges = np.asarray(cfg.model.surrogate_ranges, dtype=np.float64)
        self.cruise = cfg.drone.v_max
        self.cam_yaws = np.radians(np.asarray(cfg.cameras.cam_yaws, dtype=np.float64))
        self.half_fov = math.radians(cfg.cameras.fov) / 2
        self._importance = np.ones(len(self.cam_yaws))

    def spec(self) -> PolicySpec:
        return PolicySpec(self.H, self.K, self.cfg.tier_res, self.cfg.tier_tokens)

    def set_speed_cap(self, v_max: float) -> None:
        self.cruise = v_max

    def forward_loss(self, batch, alloc):
        raise NotImplementedError("the surrogate policy is not trained")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("surrogate policy: no weights\n", encoding="utf-8")
        return path

    def load(self, path, device="cpu"):
        return None

    def view_importance(self):
        return self._importance

    # ---------------------------------------------------------------------------------------------
    def detect(self, priv: dict[str, Any], alloc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Obstacles (xy, r, vel) visible under ``alloc``."""
        scene = priv["scene"]
        pos, yaw, t = priv["pos"], priv["yaw"], priv["time"]
        xy, r, _ = scene.circles(t)
        vel = np.zeros_like(xy)
        if scene.mover is not None:
            vel[-1] = scene.mover.velocity(t)
        if len(xy) == 0:
            return xy, r, vel
        rel = xy - pos[:2]
        dist = np.linalg.norm(rel, axis=1)
        bearing = np.arctan2(rel[:, 1], rel[:, 0]) - yaw
        seen = np.zeros(len(xy), dtype=bool)
        imp = np.zeros(len(self.cam_yaws))
        for j, (cy, tier) in enumerate(zip(self.cam_yaws, alloc)):
            if tier == 0:
                continue
            off = (bearing - cy + math.pi) % (2 * math.pi) - math.pi
            ang_r = np.arcsin(np.clip(r / np.maximum(dist, 1e-6), 0, 1))
            vis = (np.abs(off) <= self.half_fov + ang_r) & (dist - r <= self.ranges[tier])
            seen |= vis
            # proximity importance (B3's stand-in for attention): near obstacles in view matter most
            imp[j] = float((1.0 / np.maximum(dist[vis] - r[vis], 0.5)).sum())
        self._importance = imp + 0.05
        return xy[seen], r[seen], vel[seen]

    def predict_chunk(self, obs, alloc):
        priv = obs["privileged"]
        cfg = self.cfg
        pos, yaw = priv["pos"], priv["yaw"]
        goal = priv["scene"].goal_xy
        xy, r, vel = self.detect(priv, np.asarray(alloc))
        to_goal = goal - pos[:2]
        dist_goal = float(np.linalg.norm(to_goal))
        goal_dir = math.atan2(to_goal[1], to_goal[0])

        # predict movers half a second ahead, inflate everything by airframe + margin
        xy = xy + vel * 0.5
        # pass obstacles outside the margin the governor enforces, or every pass looks like a collision course
        infl = r + cfg.drone.radius + cfg.drone.margin + 0.15
        heads = np.linspace(-math.pi, math.pi, N_HEADINGS, endpoint=False)
        d = np.stack([np.cos(heads), np.sin(heads)], 1)
        free = np.full(N_HEADINGS, 50.0)       # free ray length along each heading
        cone_d = np.full(N_HEADINGS, 50.0)     # clearance in the governor's cone around each heading
        if len(xy):
            rel = xy - pos[:2]
            b = d @ rel.T                                   # (Hd, N)
            c = (rel ** 2).sum(1) - infl ** 2
            disc = b ** 2 - c
            s = b - np.sqrt(np.maximum(disc, 0))
            # already inside an inflated obstacle: only headings that point into it are blocked
            hit = (disc >= 0) & np.where(c < 0, b > 0, s > 0)
            s = np.where(c < 0, 0.0, s)
            free = np.where(hit, s, 50.0).min(axis=1)
            dist = np.linalg.norm(rel, axis=1)
            ang = np.arccos(np.clip(b / np.maximum(dist, 1e-9), -1, 1))
            half = np.arcsin(np.clip(r / np.maximum(dist, 1e-9), 0, 1))
            in_cone = ang - half <= math.radians(cfg.budget.cone_deg)
            cone_d = np.where(in_cone, dist - r - cfg.drone.radius, 50.0).min(axis=1)
        need = min(dist_goal, 3.0)
        progress = np.cos(heads - goal_dir)
        # headings the governor would stop (inside the margin) are as bad as blocked ones
        blocked = (free < need) | (cone_d < cfg.drone.margin + 0.15)
        score = progress + 0.15 * np.minimum(free, 6.0) / 6.0 - 3.0 * blocked
        k = int(np.argmax(score))
        speed = min(self.cruise, max(free[k] - 0.3, 0.0) * 2.0 + 0.5,
                    math.sqrt(2 * 4.0 * max(dist_goal - 0.3, 0.0)) + 0.3)
        v_w = d[k] * speed
        vz = 1.5 * (cfg.sim.altitude - pos[2])
        # Hold yaw: a body-frame command stays valid however stale it is, and the drone flies
        # sideways or backwards when it must, which is exactly where 360 degree input pays off.
        yaw_rate = 0.0
        cy, sy = math.cos(yaw), math.sin(yaw)
        a = np.array([cy * v_w[0] + sy * v_w[1], -sy * v_w[0] + cy * v_w[1], vz, yaw_rate])
        return np.repeat(a[None], self.H, axis=0)
