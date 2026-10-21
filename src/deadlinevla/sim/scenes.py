"""Procedural task scenes shared by every simulator backend.

Obstacles are vertical cylinders taller than the flight envelope, a goal object with a colour and a
class, and (for ``rear_threat``) one moving obstacle approaching from a side or rear camera.

Families
    goal_nav     goals in any direction, including behind the start
    sparse_nav   goal_nav at a quarter of the clutter: the deadline binds only some of the time
    highspeed    long runs (use a large arena and goal_dist) through sparse clutter, with small obstacles
                 placed close to the straight line from start to goal: detection range at speed decides
    rear_threat  goal ahead, a moving obstacle crossing in from the rear/side
    slalom       rows of obstacles across the arena with one gap each, forcing fast flight through gaps
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from deadlinevla.config import Config

PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (215, 40, 40), "blue": (40, 80, 220), "yellow": (230, 205, 40),
    "green": (40, 170, 70), "orange": (240, 135, 30), "purple": (150, 60, 190),
}
CLASSES = ("backpack", "cone", "box", "barrel", "chair")
FAMILIES = ("goal_nav", "sparse_nav", "rear_threat", "slalom", "highspeed")
_FAMILY_ID = {"goal_nav": 0, "rear_threat": 1, "slalom": 2, "sparse_nav": 3, "highspeed": 4}  # stable streams
OBSTACLE_HEIGHT = 6.0


@dataclass
class Mover:
    """A moving obstacle. ``linear``: straight line from p0 with ``vel``. ``intercept``: hunts the drone.

    The intercepting threat re-aims every control tick at the true intercept point computed from the drone's
    real position and velocity (``steer``). It makes one pass: once it has come within ``lock_range`` and the
    range starts opening, it locks its heading and flies on. A hunter that turned round forever would make
    survival impossible for any drone capped below the threat speed, which tests the cap, not perception.
    """

    p0: np.ndarray
    vel: np.ndarray
    r: float
    h: float = OBSTACLE_HEIGHT
    t_start: float = 0.5
    mode: str = "linear"
    lock_range: float = 3.0

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        self._anchor_t = self.t_start
        self._anchor_p = np.array(self.p0, dtype=np.float64)
        self._vel = np.array(self.vel, dtype=np.float64)
        self.locked = False
        self.came_close = False
        self.catchable_ticks = 0
        self.uncatchable_ticks = 0

    def pos(self, t: float) -> np.ndarray:
        if t <= self.t_start:
            return np.array(self.p0, dtype=np.float64)
        if t <= self._anchor_t:
            return self._anchor_p.copy()
        return self._anchor_p + self._vel * (t - self._anchor_t)

    def velocity(self, t: float) -> np.ndarray:
        return self._vel.copy() if t >= self.t_start else np.zeros(2)

    def steer(self, t: float, drone_pos: np.ndarray, drone_vel: np.ndarray) -> None:
        """Re-aim at the intercept (intercept mode only). Called by the env once per control tick."""
        if self.mode != "intercept" or t < self.t_start or self.locked:
            return
        p_o = self.pos(t)
        delta = np.asarray(drone_pos, dtype=np.float64)[:2] - p_o
        rng = float(np.linalg.norm(delta))
        closing = float(delta @ (self._vel - np.asarray(drone_vel, dtype=np.float64)[:2])) > 0
        if rng < self.lock_range:
            self.came_close = True
        if self.came_close and not closing:
            self.locked = True            # one pass: the range is opening after a near miss
            return
        t_hit = intercept_time(delta, np.asarray(drone_vel, dtype=np.float64)[:2], float(np.linalg.norm(self.vel)))
        speed = float(np.linalg.norm(self.vel))
        if t_hit is None:
            self.uncatchable_ticks += 1   # the drone outruns it: keep the current heading
            return
        self.catchable_ticks += 1
        aim = np.asarray(drone_pos, dtype=np.float64)[:2] + np.asarray(drone_vel, dtype=np.float64)[:2] * t_hit
        direction = aim - p_o
        n = float(np.linalg.norm(direction))
        if n > 1e-9:
            self._anchor_p, self._anchor_t = p_o, t
            self._vel = direction / n * speed


def intercept_time(delta: np.ndarray, v_d: np.ndarray, s_o: float) -> float | None:
    """Earliest t > 0 with |delta + v_d t| = s_o t, where delta = p_drone - p_obstacle; None if uncatchable.

    (|v_d|^2 - s_o^2) t^2 + 2 (delta . v_d) t + |delta|^2 = 0
    """
    A = float(v_d @ v_d) - s_o * s_o
    B = 2.0 * float(delta @ v_d)
    C = float(delta @ delta)
    if abs(A) < 1e-9:
        return -C / B if B < 0 else None
    disc = B * B - 4 * A * C
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    roots = [t for t in ((-B - sq) / (2 * A), (-B + sq) / (2 * A)) if t > 1e-9]
    return min(roots) if roots else None


@dataclass
class Scene:
    family: str
    seed: int
    arena: float
    obs_xy: np.ndarray                 # (N, 2)
    obs_r: np.ndarray                  # (N,)
    obs_h: np.ndarray                  # (N,)
    obs_rgb: np.ndarray                # (N, 3) uint8
    start: np.ndarray                  # (4,) x, y, z, yaw
    goal_xy: np.ndarray                # (2,)
    goal_color: str
    goal_cls: str
    mover: Optional[Mover] = None
    instruction: str = ""
    meta: dict = field(default_factory=dict)

    def circles(self, t: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Static obstacles plus the mover at time t: (xy, r, h)."""
        if self.mover is None:
            return self.obs_xy, self.obs_r, self.obs_h
        m = self.mover
        return (np.vstack([self.obs_xy, m.pos(t)[None]]), np.append(self.obs_r, m.r),
                np.append(self.obs_h, m.h))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _grey(rng: np.random.Generator) -> np.ndarray:
    base = rng.uniform(70, 150)
    tint = rng.uniform(-12, 12, size=3)
    return np.clip(base + tint + np.array([8, 0, -10]), 0, 255)


def _scatter(rng, n, arena, rad_range, keep_clear: list[tuple[np.ndarray, float]]):
    xy, r = [], []
    half = arena / 2 - 0.5
    tries = 0
    while len(xy) < n and tries < n * 50:
        tries += 1
        p = rng.uniform(-half, half, size=2)
        rad = rng.uniform(*rad_range)
        if any(np.linalg.norm(p - c) < rad + clear for c, clear in keep_clear):
            continue
        xy.append(p)
        r.append(rad)
    return xy, r


def build_scene(family: str, seed: int, cfg: Optional[Config] = None) -> Scene:
    """Procedurally place obstacles at clutter_density, a start pose and a goal object."""
    if family not in FAMILIES:
        raise ValueError(f"unknown scene family {family!r}; known: {FAMILIES}")
    cfg = cfg or Config()
    sc, sim = cfg.scenes, cfg.sim
    rng = np.random.default_rng([seed, _FAMILY_ID[family]])
    A = sim.arena
    half = A / 2
    yaw0 = rng.uniform(-math.pi, math.pi)
    lo, hi = sc.goal_dist[0], sc.goal_dist[-1]

    for _ in range(100):
        start_xy = rng.uniform(-half / 3, half / 3, size=2)
        if family in ("goal_nav", "sparse_nav"):
            bearing = rng.uniform(-math.pi, math.pi)
        elif family == "rear_threat":
            bearing = rng.uniform(-math.pi / 4, math.pi / 4)
        else:
            bearing = rng.uniform(-math.pi / 8, math.pi / 8)
        dist = rng.uniform(lo, hi) if family != "slalom" else hi
        goal_dir = np.array([math.cos(yaw0 + bearing), math.sin(yaw0 + bearing)])
        goal = start_xy + dist * goal_dir
        if (np.abs(goal) < half - 1.5).all():
            break

    keep = [(start_xy, 1.8), (goal, 1.4)]
    clutter = {"goal_nav": 1.0, "sparse_nav": 0.25, "rear_threat": 0.6, "slalom": 0.0, "highspeed": 0.15}
    density = sc.clutter_density * clutter[family]
    n = int(round(density * A * A))
    xy, r = _scatter(rng, n, A, sc.obstacle_radius, keep)
    meta: dict = {"bearing": bearing}

    if family == "slalom":
        # Rows perpendicular to the start->goal line, spanning the arena, one 2.8 m gap per row.
        perp = np.array([-goal_dir[1], goal_dir[0]])
        rr, spacing, gap = 0.4, 1.2, 2.8
        rows = []
        s = 3.5
        side = 1.0 if rng.random() < 0.5 else -1.0
        while s < dist - 2.5:
            offset = side * rng.uniform(0.8, 3.0)
            side = -side
            centre = start_xy + s * goal_dir
            lateral = np.arange(-A, A, spacing)
            for lat in lateral:
                if abs(lat - offset) < gap / 2 + rr:
                    continue
                p = centre + lat * perp
                if (np.abs(p) < half).all():
                    xy.append(p)
                    r.append(rr)
            rows.append(float(s))
            s += 4.5
        meta["rows"] = rows

    if family == "highspeed":
        # Small obstacles (0.5-0.8 m) near the direct line, 9-14 m apart: at speed, the drone must detect
        # them far enough ahead to brake or swerve.
        perp = np.array([-goal_dir[1], goal_dir[0]])
        s = rng.uniform(10.0, 14.0)
        placed = []
        while s < dist - 6.0:
            rr = rng.uniform(0.25, 0.4)
            p = start_xy + s * goal_dir + rng.uniform(-1.0, 1.0) * perp
            if (np.abs(p) < half - 0.5).all():
                xy.append(p)
                r.append(rr)
                placed.append(float(s))
            s += rng.uniform(9.0, 14.0)
        meta["path_obstacles"] = placed

    mover = None
    if family == "rear_threat":
        side = 1.0 if rng.random() < 0.5 else -1.0
        rel_bearing = side * rng.uniform(math.radians(110), math.radians(160))
        d0 = rng.uniform(8.0, 11.0)
        p0 = start_xy + d0 * np.array([math.cos(yaw0 + rel_bearing), math.sin(yaw0 + rel_bearing)])
        s_aim = rng.uniform(5.0, 8.0)
        aim = start_xy + s_aim * goal_dir
        t_arrive = s_aim / 4.0 + 0.5
        if sc.threat_mode == "intercept":
            speed = sc.threat_speed          # the hunter flies at exactly the configured speed
        else:
            speed = float(np.clip(np.linalg.norm(aim - p0) / t_arrive, 2.0, 2.0 * sc.threat_speed))
        vel = (aim - p0) / np.linalg.norm(aim - p0) * speed
        mover = Mover(p0=p0, vel=vel, r=sc.threat_radius, t_start=0.5, mode=sc.threat_mode)
        meta["mover_speed"] = speed

    goal_color = str(rng.choice(list(PALETTE)))
    goal_cls = str(rng.choice(CLASSES))
    rgb = np.array([_grey(rng) for _ in xy], dtype=np.uint8).reshape(-1, 3)
    scene = Scene(
        family=family, seed=seed, arena=A,
        obs_xy=np.array(xy, dtype=np.float64).reshape(-1, 2), obs_r=np.array(r, dtype=np.float64),
        obs_h=np.full(len(xy), OBSTACLE_HEIGHT), obs_rgb=rgb,
        start=np.array([start_xy[0], start_xy[1], sim.altitude, yaw0]),
        goal_xy=goal, goal_color=goal_color, goal_cls=goal_cls, mover=mover, meta=meta,
    )
    scene.instruction = make_instruction(scene)
    return scene


def empty_scene(arena: float = 200.0, altitude: float = 1.5) -> Scene:
    """No obstacles; used by measure_braking."""
    z = np.zeros((0, 2))
    return Scene(family="empty", seed=0, arena=arena, obs_xy=z, obs_r=np.zeros(0), obs_h=np.zeros(0),
                 obs_rgb=np.zeros((0, 3), np.uint8), start=np.array([0.0, 0.0, altitude, 0.0]),
                 goal_xy=np.array([arena, 0.0]), goal_color="red", goal_cls="box")


def relative_direction(scene: Scene) -> str:
    rel = math.atan2(*(scene.goal_xy - scene.start[:2])[::-1]) - scene.start[3]
    b = math.degrees(_wrap(rel))
    if abs(b) <= 45:
        return "in front of you"
    if abs(b) >= 135:
        return "behind you"
    return "to your left" if b > 0 else "to your right"


def make_instruction(scene: Scene) -> str:
    """Templated language goal; sometimes names the goal's direction relative to the start."""
    rng = np.random.default_rng([scene.seed, 7])
    c, k = scene.goal_color, scene.goal_cls
    direction = relative_direction(scene)
    templates = [
        f"fly to the {c} {k}",
        f"go to the {c} {k} {direction}",
        f"navigate to the {k} that is {c}",
        f"the {c} {k} is {direction}, fly there",
    ]
    return templates[int(rng.integers(len(templates)))]


def cone_clearance(scene: Scene, pos: np.ndarray, t: float, v_hat: Optional[np.ndarray],
                   cone_deg: float, radius: float, max_range: float) -> float:
    """Ground-truth distance from the airframe to the nearest obstacle surface inside the cone.

    ``v_hat`` is a world-frame direction (only its horizontal part is used); ``None`` means hover,
    where every direction counts. An obstacle counts if any part of it falls inside the cone.
    """
    xy, r, h = scene.circles(t)
    if len(xy) == 0:
        return float(max_range)
    rel = xy - pos[:2]
    dist = np.linalg.norm(rel, axis=1)
    surface = dist - r - radius
    sel = h + radius >= pos[2]
    if v_hat is not None:
        v2 = np.asarray(v_hat, dtype=np.float64)[:2]
        n = np.linalg.norm(v2)
        if n > 1e-6:
            v2 = v2 / n
            cosc = np.clip((rel @ v2) / np.maximum(dist, 1e-9), -1.0, 1.0)
            half = np.arcsin(np.clip(r / np.maximum(dist, 1e-9), 0.0, 1.0))
            sel &= (np.arccos(cosc) - half <= math.radians(cone_deg)) | (dist <= r)
    if not sel.any():
        return float(max_range)
    return float(np.clip(surface[sel].min(), 0.0, max_range))


def obstacle_kinematics(scene: Scene, pos: np.ndarray, vel: np.ndarray, t: float, radius: float,
                        max_range: float) -> dict[str, np.ndarray]:
    """Ground truth per obstacle whose surface is within ``max_range``: what the relative deadline needs.

    World frame: ``p`` (N, 2) offset from the drone to each obstacle centre, ``R`` (N,) obstacle radius
    plus the airframe radius, ``v_obs`` (N, 2) obstacle velocity. ``vel`` is accepted for symmetry with
    the depth path; closing speeds are computed from it by ``budget.relative.Contacts``.
    """
    xy, r, h = scene.circles(t)
    if len(xy) == 0:
        return {"p": np.zeros((0, 2)), "R": np.zeros(0), "v_obs": np.zeros((0, 2)), "size": np.zeros(0)}
    v_obs = np.zeros_like(xy)
    if scene.mover is not None:
        v_obs[-1] = scene.mover.velocity(t)
    rel = xy - pos[:2]
    d = np.linalg.norm(rel, axis=1) - r - radius
    keep = (d <= max_range) & (h + radius >= pos[2])
    return {"p": rel[keep], "R": r[keep] + radius, "v_obs": v_obs[keep], "size": 2 * r[keep]}


def collides(scene: Scene, pos: np.ndarray, t: float, radius: float) -> bool:
    xy, r, h = scene.circles(t)
    if pos[2] <= 0.05:
        return True
    if len(xy) == 0:
        return False
    d = np.linalg.norm(xy - pos[:2], axis=1)
    return bool(((d < r + radius) & (pos[2] < h + radius)).any())


def occupancy_grid(scene: Scene, res: float, inflate: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean grid (nx, ny) of static obstacles inflated by ``inflate``; returns (grid, origin)."""
    n = int(math.ceil(scene.arena / res))
    origin = np.array([-scene.arena / 2, -scene.arena / 2])
    cs = origin[0] + (np.arange(n) + 0.5) * res
    gx, gy = np.meshgrid(cs, cs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    occ = np.zeros(len(pts), dtype=bool)
    for start in range(0, len(scene.obs_xy), 64):
        c = scene.obs_xy[start:start + 64]
        rr = scene.obs_r[start:start + 64]
        d = np.linalg.norm(pts[:, None, :] - c[None], axis=2)
        occ |= (d < rr[None] + inflate).any(axis=1)
    return occ.reshape(n, n), origin
