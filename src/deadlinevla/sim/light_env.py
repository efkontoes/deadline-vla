"""Light backend: a numpy point-mass multirotor with six ray-cast pinhole cameras.

Not a replacement for Isaac Lab; it exists so the whole pipeline runs and is tested on a CPU. The
dynamics are a first-order velocity tracker with an acceleration limit, so braking deceleration is a
property of the model that ``measure_braking`` has to measure, not a typed constant.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from deadlinevla.config import Config
from deadlinevla.sim.scenes import PALETTE, Scene, collides, cone_clearance

SKY = np.array([172, 200, 232], dtype=np.float64)
FLOOR_A = np.array([112, 128, 98], dtype=np.float64)
FLOOR_B = np.array([96, 110, 86], dtype=np.float64)


def render_cameras(pos: np.ndarray, yaw: float, scene: Scene, t: float, res: int, fov_deg: float,
                   cam_yaws_deg, max_range: float) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast all cameras. Returns RGB (V, R, R, 3) uint8 and depth-to-image-plane (V, R, R)."""
    f = (res / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    c = (np.arange(res) + 0.5 - res / 2.0) / f
    xn, yn = c, -c                                   # column -> right, row -> up
    cam = yaw + np.radians(np.asarray(cam_yaws_deg, dtype=np.float64))
    ang = cam[:, None] - np.arctan(xn)[None, :]      # (V, W) horizontal ray heading
    dx, dy = np.cos(ang), np.sin(ang)
    hscale = np.sqrt(1.0 + xn ** 2)                  # horizontal distance per unit depth

    xy, r, h = scene.circles(t)
    goal_r = 0.35
    xy = np.vstack([xy, scene.goal_xy[None]]) if len(xy) else scene.goal_xy[None]
    r = np.append(r, goal_r)
    h = np.append(h, 0.9)
    rgb_obj = np.vstack([scene.obs_rgb.astype(np.float64).reshape(-1, 3),
                         np.full((1 if scene.mover is not None else 0, 3), (60, 60, 70)),
                         np.array(PALETTE[scene.goal_color], dtype=np.float64)[None]])

    ox, oy = xy[:, 0] - pos[0], xy[:, 1] - pos[1]
    b = dx[..., None] * ox + dy[..., None] * oy                  # (V, W, N)
    cc = ox ** 2 + oy ** 2 - r ** 2
    disc = b ** 2 - cc
    sq = np.sqrt(np.maximum(disc, 0.0))
    s = np.where(cc < 0, 0.01, b - sq)                           # inside an obstacle: hit at once
    ok = (disc >= 0) & (s > 0)
    depth_obj = np.where(ok, s / hscale[None, :, None], np.inf)  # (V, W, N)
    zhit = pos[2] + yn[None, :, None, None] * depth_obj[:, None]  # (V, H, W, N)
    hit = (zhit >= 0.0) & (zhit <= h)
    d_all = np.where(hit, depth_obj[:, None], np.inf)
    idx = np.argmin(d_all, axis=-1)                              # (V, H, W)
    d_obj = np.take_along_axis(d_all, idx[..., None], axis=-1)[..., 0]

    with np.errstate(divide="ignore"):
        d_floor = np.where(yn < 0, pos[2] / np.maximum(-yn, 1e-12), np.inf)  # (H,)
    d_floor = np.broadcast_to(d_floor[None, :, None], d_obj.shape)
    depth = np.minimum(np.minimum(d_obj, d_floor), max_range)

    obj_mask = (d_obj <= d_floor) & np.isfinite(d_obj)
    floor_mask = ~obj_mask & np.isfinite(d_floor)
    # floor checker from the world-frame hit point
    fx = pos[0] + dx[:, None, :] * hscale[None, None, :] * d_floor
    fy = pos[1] + dy[:, None, :] * hscale[None, None, :] * d_floor
    with np.errstate(invalid="ignore"):
        checker = ((np.floor(fx / 2.0) + np.floor(fy / 2.0)) % 2 == 0)[..., None]
    floor_rgb = np.where(checker, FLOOR_A, FLOOR_B)
    sky_rgb = SKY * (0.9 + 0.1 * (np.arange(res)[None, :, None, None] / res))
    img = np.where(obj_mask[..., None], rgb_obj[idx], np.where(floor_mask[..., None], floor_rgb, sky_rgb))
    shade = np.clip(1.0 - depth / (1.6 * max_range), 0.35, 1.0)[..., None]
    img = np.where((obj_mask | floor_mask)[..., None], img * shade, img)
    return np.clip(img, 0, 255).astype(np.uint8), depth.astype(np.float32)


class LightDroneEnv:
    """One numpy drone. ``num_envs`` > 1 is handled by callers holding several instances."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.res = cfg.tier_res[3]
        self.n_sub = max(1, int(round(cfg.sim.physics_hz / cfg.drone.ctrl_hz)))
        self.dt_phys = 1.0 / (cfg.drone.ctrl_hz * self.n_sub)
        self.scene: Optional[Scene] = None

    # ---------------------------------------------------------------------------------------------
    def reset(self, scene: Scene) -> dict[str, Any]:
        self.scene = scene
        self.pos = scene.start[:3].astype(np.float64).copy()
        self.yaw = float(scene.start[3])
        self.vel = np.zeros(3)
        self.yaw_rate = 0.0
        self.t = 0.0
        self.distance = 0.0
        self.done = False
        self.max_t = self.cfg.sim.max_episode_s
        info = self._info(collision=False, goal=False, oob=False)
        return self._obs(info)

    def step(self, v_cmd: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.scene is None:
            raise RuntimeError("call reset(scene) first")
        d = self.cfg.drone
        v_cmd = np.asarray(v_cmd, dtype=np.float64)
        collision = goal = oob = False
        for _ in range(self.n_sub):
            cy, sy = math.cos(self.yaw), math.sin(self.yaw)
            target = np.array([cy * v_cmd[0] - sy * v_cmd[1], sy * v_cmd[0] + cy * v_cmd[1], v_cmd[2]])
            acc = (target - self.vel) / d.tau_vel
            na = float(np.linalg.norm(acc))
            if na > d.a_max:
                acc *= d.a_max / na
            self.vel += acc * self.dt_phys
            step = self.vel * self.dt_phys
            self.pos += step
            self.pos[2] = min(self.pos[2], 4.0)
            self.distance += float(np.linalg.norm(step[:2]))
            self.yaw_rate = float(np.clip(v_cmd[3], -d.yaw_rate_max, d.yaw_rate_max))
            self.yaw = (self.yaw + self.yaw_rate * self.dt_phys + math.pi) % (2 * math.pi) - math.pi
            self.t += self.dt_phys
            if collides(self.scene, self.pos, self.t, d.radius):
                collision = True
                break
            if np.linalg.norm(self.pos[:2] - self.scene.goal_xy) < self.cfg.sim.goal_radius:
                goal = True
                break
        if (np.abs(self.pos[:2]) > self.scene.arena / 2 + 2.0).any():
            oob = True
        info = self._info(collision=collision, goal=goal, oob=oob)
        return self._obs(info), info

    def clearance(self, v_hat: Optional[np.ndarray], cone_deg: float) -> float:
        return cone_clearance(self.scene, self.pos, self.t, v_hat, cone_deg, self.cfg.drone.radius,
                              self.cfg.sim.max_range)

    # ---------------------------------------------------------------------------------------------
    def _info(self, collision: bool, goal: bool, oob: bool) -> dict[str, Any]:
        speed = float(np.linalg.norm(self.vel))
        v_hat = self.vel / speed if speed >= self.cfg.budget.hover_speed else None
        timeout = self.t >= self.max_t - 1e-9
        self.done = collision or goal or oob or timeout
        return {"collision": collision, "goal_reached": goal, "out_of_bounds": oob, "timeout": timeout,
                "done": self.done, "clearance": self.clearance(v_hat, self.cfg.budget.cone_deg),
                "time": self.t, "distance": self.distance}

    def _obs(self, info: dict[str, Any]) -> dict[str, Any]:
        cfg = self.cfg
        if cfg.sim.render:
            images, depth = render_cameras(self.pos, self.yaw, self.scene, self.t, self.res, cfg.cameras.fov,
                                           cfg.cameras.cam_yaws, cfg.sim.max_range)
        else:  # policies that need no pixels (the surrogate with oracle clearance) skip the cameras
            images = depth = None
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        vb = np.array([cy * self.vel[0] + sy * self.vel[1], -sy * self.vel[0] + cy * self.vel[1], self.vel[2]])
        state = np.array([vb[0], vb[1], vb[2], self.yaw_rate, 0.0, 0.0, self.pos[2]])
        return {
            "images": images, "depth": depth, "state": state, "vel_body": vb,
            "instruction": self.scene.instruction, "info": info,
            "privileged": {"scene": self.scene, "pos": self.pos.copy(), "vel": self.vel.copy(),
                           "yaw": self.yaw, "time": self.t},
        }
