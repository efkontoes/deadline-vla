"""Relative deadline (v3): reason about every obstacle's closing speed, not only the drone's own motion.

A *contact* is one obstacle seen from the drone (body frame): centre offset ``p``, combined radius ``R``
(obstacle + airframe), and the obstacle's velocity ``w``. Its surface clearance is d = |p| - R, its
direction r_hat = p / |p|, and its closing speed under drone velocity v is c = (v - w) . r_hat.

Threats. Shrinking clearance alone is not a threat: an obstacle you fly past closes at v cos(theta)
until you pass it, but never needs braking. A contact is a threat under v if the straight-line relative
motion brings its surface within the margin m (miss distance < m), if it is already inside the margin,
or if it lies in the cone ahead of v (``cone_deg``; a grazed obstacle is one heading change away from a
collision course). Only threats enter the deadline and the governor.

Deadline. To survive contact i the drone must cancel c_i before d_i runs out, with deceleration a. Same
algebra as braking:

    L_max,i = (d_i - m - c_i^2/(2a)) / c_i     (on course, c_i > 0),     L_max = min_i L_max,i

A static obstacle dead ahead while flying at v has c = v and gives the original formula, so this is a
strict generalisation. A threat closing from behind now tightens the deadline before it is close.

Weights. Closing speed replaces own-velocity alignment, per camera j (all closing contacts count here:
seeing what you pass matters even when it is not on course):

    c_j = max(0, v . u_j, max_{i in camera j} c_i),     w_j = eps + (c_j / max_k c_k)^kappa

Flying forward through a static world this reproduces the original weights; something closing from
behind raises the rear camera's weight before it is near.

Governor. Every contact on a collision course under the command u constrains it:
(u - w_i) . r_hat_i <= v*(L, d_i), the closing speed must be one the drone can still cancel. Contacts
ahead of the command are met by slowing down along the command (the original governor); anything else,
such as a threat from the side or behind, or one closing faster than braking can fix, is met by removing the
offending component, which accelerates the drone away along -r_hat. Inside the margin (v* = 0) only
motion away or tangent is allowed, so the drone slides off an obstacle instead of deadlocking.

Contacts come from ground truth in simulation (one per obstacle); the depth ablation builds one pseudo-
contact per camera from the depth image, with closing speed from frame-to-frame differences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from deadlinevla.budget.governor import safe_speed


@dataclass
class Contacts:
    p: np.ndarray          # (N, 2) body-frame offset drone -> obstacle centre, m
    R: np.ndarray          # (N,) obstacle radius + airframe radius, m
    w: np.ndarray          # (N, 2) body-frame obstacle velocity, m/s

    def __post_init__(self):
        self.p = np.asarray(self.p, dtype=np.float64).reshape(-1, 2)
        self.R = np.asarray(self.R, dtype=np.float64).reshape(-1)
        self.w = np.asarray(self.w, dtype=np.float64).reshape(-1, 2)
        self.dist = np.linalg.norm(self.p, axis=1)
        self.r_hat = self.p / np.maximum(self.dist, 1e-9)[:, None]
        self.d = np.maximum(self.dist - self.R, 0.0)

    def __len__(self) -> int:
        return len(self.R)

    def closing(self, v: np.ndarray) -> np.ndarray:
        """How fast each clearance shrinks under drone velocity v (negative: opening)."""
        v2 = np.asarray(v, dtype=np.float64)[:2]
        return ((v2[None] - self.w) * self.r_hat).sum(1)

    def miss(self, v: np.ndarray) -> np.ndarray:
        """Surface distance at closest approach under straight-line relative motion (current d if receding)."""
        v2 = np.asarray(v, dtype=np.float64)[:2]
        rel = self.w - v2[None]                        # obstacle velocity relative to the drone
        rr = (rel ** 2).sum(1)
        t = np.where(rr > 1e-12, -(self.p * rel).sum(1) / np.maximum(rr, 1e-12), 0.0)
        closest = np.linalg.norm(self.p + rel * np.maximum(t, 0.0)[:, None], axis=1) - self.R
        return np.where(t > 0, closest, self.d)

    def on_course(self, v: np.ndarray, m: float) -> np.ndarray:
        return (self.miss(v) < m) | (self.d < m)

    def in_cone(self, v: np.ndarray, cone_deg: float) -> np.ndarray:
        """Any part of the contact within ``cone_deg`` of the direction of v (none if v is ~0)."""
        v2 = np.asarray(v, dtype=np.float64)[:2]
        n = float(np.linalg.norm(v2))
        if n < 1e-6:
            return np.zeros(len(self), dtype=bool)
        ang = np.arccos(np.clip(self.r_hat @ (v2 / n), -1.0, 1.0))
        half = np.arcsin(np.clip(self.R / np.maximum(self.dist, 1e-9), 0.0, 1.0))
        return ang - half <= math.radians(cone_deg)

    def threat(self, v: np.ndarray, m: float, cone_deg: float) -> np.ndarray:
        """On a collision course, or inside the cone ahead of v.

        The straight-line course test alone is too permissive: an obstacle grazed just outside the margin
        becomes a collision course after one small heading change, at short range and full closing speed.
        Keeping everything in the cone ahead (the v2 rule) makes the relative governor at least as
        conservative as v2 ahead while adding threats from the side and behind.
        """
        return self.on_course(v, m) | self.in_cone(v, cone_deg)


def contacts_from_kinematics(kin: dict[str, np.ndarray], yaw: float) -> Contacts:
    """World-frame ground truth (``scenes.obstacle_kinematics``) to body-frame contacts."""
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, s], [-s, c]])                       # world -> body
    return Contacts(p=kin["p"] @ rot.T, R=kin["R"], w=kin["v_obs"] @ rot.T)


def contacts_from_cameras(d_cam: np.ndarray, c_cam: np.ndarray, cam_axes: np.ndarray,
                          v_body: np.ndarray, max_range: float) -> Contacts:
    """Depth ablation: one point-like pseudo-contact per camera on its optical axis.

    ``c_cam`` is the measured closing speed (from the change of ``d_cam`` between ticks); the obstacle's
    velocity along the axis follows as (v . u_j - c_j); its lateral velocity is unobservable here.
    """
    u = cam_axes[:, :2]
    keep = d_cam < max_range * 0.999
    along = u @ np.asarray(v_body, dtype=np.float64)[:2] - c_cam
    return Contacts(p=u[keep] * d_cam[keep, None], R=np.zeros(int(keep.sum())), w=u[keep] * along[keep, None])


def _closing_threats(contacts: Contacts, v: np.ndarray, m: float, cone_deg: float) -> tuple[np.ndarray, np.ndarray]:
    c = contacts.closing(v)
    sel = (c > 1e-6) & contacts.threat(v, m, cone_deg)
    return c[sel], contacts.d[sel]


def relative_deadline(contacts: Contacts, v_body: np.ndarray, a: float, m: float, cone_deg: float = 30.0) -> float:
    """L_max = min over closing threats of (d - m - c^2/2a) / c; inf when none."""
    if not len(contacts):
        return math.inf
    c, d = _closing_threats(contacts, v_body, m, cone_deg)
    if not len(c):
        return math.inf
    return float(((d - m - c ** 2 / (2 * a)) / c).min())


def min_slack(contacts: Contacts, v_body: np.ndarray, L: float, a: float, m: float, cone_deg: float = 30.0) -> float:
    """min over closing threats of d - m - c L - c^2/2a (negative: relative deadline violated)."""
    if not len(contacts):
        return math.inf
    c, d = _closing_threats(contacts, v_body, m, cone_deg)
    if not len(c):
        return math.inf
    return float((d - m - c * L - c ** 2 / (2 * a)).min())


def camera_closing(contacts: Contacts, v_body: np.ndarray, cam_axes: np.ndarray) -> np.ndarray:
    """c_j per camera: own motion along u_j, or the fastest-closing contact the camera sees."""
    u = cam_axes[:, :2]
    v2 = np.asarray(v_body, dtype=np.float64)[:2]
    c_cam = np.maximum(0.0, u @ v2)
    if len(contacts):
        owner = np.argmax(contacts.r_hat @ u.T, axis=1)      # cameras tile 360 degrees
        c = contacts.closing(v_body)
        for j in range(len(u)):
            mine = c[owner == j]
            if len(mine):
                c_cam[j] = max(c_cam[j], float(mine.max()))
    return c_cam


def closing_weights(c_cam: np.ndarray, kappa: float, eps: float, still: float = 0.3) -> np.ndarray:
    """w_j = eps + (c_j / max c)^kappa; uniform when nothing closes faster than ``still`` m/s."""
    top = float(c_cam.max()) if len(c_cam) else 0.0
    if top < still:
        return np.ones(len(c_cam))
    return eps + (np.maximum(c_cam, 0.0) / top) ** kappa


def govern_relative(v_cmd: np.ndarray, L: float, contacts: Contacts, a: float, m: float, v_max: float,
                    v_now: Optional[np.ndarray] = None, cone_deg: float = 30.0, passes: int = 8) -> np.ndarray:
    """Make the commanded closing speed toward every on-course contact one the drone can still cancel.

    ``v_cmd`` is body-frame (vx, vy, vz, yaw_rate); vz and yaw rate are kept. A contact counts if it is on
    course under the command *or* under the current velocity ``v_now``: in a sharp turn the drone's
    momentum still carries it along the old heading.
    """
    out = np.array(v_cmd, dtype=np.float64, copy=True)
    u = out[:2].copy()
    if len(contacts):
        r = contacts.r_hat
        o = (contacts.w * r).sum(1)                           # obstacle speed along r_hat
        b = np.array([safe_speed(L, d, a, m) for d in contacts.d]) + o
        momentum = contacts.threat(v_now, m, cone_deg) if v_now is not None else np.zeros(len(contacts), bool)
        active = contacts.threat(u, m, cone_deg) | momentum
        if float(np.linalg.norm(u)) > 1e-9 and active.any():
            # 1) on-course contacts ahead of the command that braking can satisfy: slow down along it.
            #    Contacts with nothing left to brake for (b <= 0, e.g. inside the margin) are projected in
            #    step 2 instead, so the drone keeps its tangential motion and slides off.
            along = r @ u
            ahead = active & (along > 1e-9) & (b > 1e-9) & (along > b)
            if ahead.any():
                u *= float(np.min(b[ahead] / along[ahead]))
        # 2) anything on course and still violated (side, behind, or closing faster than braking can fix):
        #    remove the offending component, i.e. push away along -r_hat
        for _ in range(passes):
            active = contacts.threat(u, m, cone_deg) | momentum
            excess = np.where(active, r @ u - b, -np.inf)
            worst = int(np.argmax(excess))
            if excess[worst] <= 1e-9:
                break
            u -= excess[worst] * r[worst]
    n = float(np.linalg.norm(u))
    if n > v_max:
        u *= v_max / n
    out[:2] = u
    return out


def depth_camera_clearances(depth: np.ndarray, z: float, fov_deg: float, cam_yaws, cone_deg: float,
                            radius: float, max_range: float, floor_ignore_z: float = 0.2,
                            prev: Optional[np.ndarray] = None, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Per-camera clearance from depth, and closing speed from the change since the previous tick."""
    from deadlinevla.budget.clearance import depth_clearance

    axes = np.radians(np.asarray(cam_yaws, dtype=np.float64))
    d_cam = np.array([depth_clearance(depth, np.array([math.cos(y), math.sin(y), 0.0]), z, fov_deg, cam_yaws,
                                      cone_deg, radius, max_range, floor_ignore_z) for y in axes])
    if prev is None:
        return d_cam, np.zeros_like(d_cam)
    c_cam = np.where((d_cam < max_range * 0.999) & (prev < max_range * 0.999), (prev - d_cam) / dt, 0.0)
    return d_cam, c_cam
