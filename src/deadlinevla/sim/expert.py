"""Privileged scripted expert: A* on the occupancy grid, smoothed, tracked by pure pursuit.

Its speed is capped by the same braking model the policy is judged by (``safe_speed`` with the
expert's assumed latency), so demonstrations never violate the deadline.
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Optional

import numpy as np

from deadlinevla.budget.governor import safe_speed
from deadlinevla.config import Config, a_brake
from deadlinevla.sim.scenes import Scene, cone_clearance, occupancy_grid

Path = np.ndarray  # (K, 2) world-frame waypoints, start first, goal last

_NEIGH = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]


def _astar(grid: np.ndarray, s: tuple[int, int], g: tuple[int, int]) -> Optional[list[tuple[int, int]]]:
    nx, ny = grid.shape

    def h(c):
        dx, dy = abs(c[0] - g[0]), abs(c[1] - g[1])
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    open_heap = [(h(s), 0.0, s)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {s: 0.0}
    closed = set()
    while open_heap:
        _, gc, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        closed.add(cur)
        for dx, dy, w in _NEIGH:
            n = (cur[0] + dx, cur[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or grid[n] or n in closed:
                continue
            if dx and dy and (grid[cur[0] + dx, cur[1]] or grid[cur[0], cur[1] + dy]):
                continue  # no corner cutting
            nc = gc + w
            if nc < cost.get(n, math.inf):
                cost[n] = nc
                came[n] = cur
                heapq.heappush(open_heap, (nc + h(n), nc, n))
    return None


def _line_free(grid: np.ndarray, origin: np.ndarray, res: float, a: np.ndarray, b: np.ndarray) -> bool:
    n = max(2, int(np.linalg.norm(b - a) / (res * 0.5)) + 1)
    pts = a[None] + np.linspace(0, 1, n)[:, None] * (b - a)[None]
    ij = np.floor((pts - origin) / res).astype(int)
    if (ij < 0).any() or (ij >= grid.shape[0]).any():
        return False
    return not grid[ij[:, 0], ij[:, 1]].any()


def plan_path(scene: Scene, start: np.ndarray, goal: np.ndarray, res: float = 0.25,
              inflate: float = 0.95) -> Optional[Path]:
    """A* on the scene's occupancy grid, then greedy line-of-sight shortcutting."""
    grid, origin = occupancy_grid(scene, res, inflate)

    def to_cell(p):
        return tuple(np.clip(np.floor((np.asarray(p) - origin) / res).astype(int), 0, grid.shape[0] - 1))

    s, g = to_cell(start[:2]), to_cell(goal[:2])
    grid = grid.copy()
    grid[s] = grid[g] = False
    cells = _astar(grid, s, g)
    if cells is None:
        return None
    pts = [np.asarray(start[:2], dtype=np.float64)]
    pts += [origin + (np.array(c) + 0.5) * res for c in cells[1:-1]]
    pts.append(np.asarray(goal[:2], dtype=np.float64))
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _line_free(grid, origin, res, pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return np.array(out)


def _project(path: Path, p: np.ndarray) -> tuple[float, float]:
    """Arc length of the closest point on the polyline, and the distance to it."""
    best = (0.0, math.inf)
    s0 = 0.0
    for a, b in zip(path[:-1], path[1:]):
        seg = b - a
        L = float(np.linalg.norm(seg))
        u = 0.0 if L < 1e-9 else float(np.clip((p - a) @ seg / (L * L), 0.0, 1.0))
        d = float(np.linalg.norm(a + u * seg - p))
        if d < best[1]:
            best = (s0 + u * L, d)
        s0 += L
    return best


def _point_at(path: Path, s: float) -> np.ndarray:
    for a, b in zip(path[:-1], path[1:]):
        L = float(np.linalg.norm(b - a))
        if s <= L:
            return a + (b - a) * (s / max(L, 1e-9))
        s -= L
    return path[-1]


def _corner_speed(path: Path, s: float, a: float) -> float:
    """Speed from which the drone can still slow to a safe turning speed at every upcoming vertex."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    best = math.inf
    for k in range(1, len(path) - 1):
        ahead = cum[k] - s
        if ahead < -0.2:
            continue
        if ahead > 12.0:
            break
        u, w = path[k] - path[k - 1], path[k + 1] - path[k]
        cosang = float(u @ w / max(np.linalg.norm(u) * np.linalg.norm(w), 1e-9))
        turn = math.acos(float(np.clip(cosang, -1.0, 1.0)))
        v_turn = max(1.0, 6.0 * (1.0 - turn / math.pi) ** 2)
        best = min(best, math.sqrt(v_turn ** 2 + 2 * a * 0.5 * max(ahead, 0.0)))
    return best


def expert_action(state: dict[str, Any], path: Path, cfg: Config) -> np.ndarray:
    """Pure-pursuit body-frame velocity along ``path`` with the braking-model speed cap.

    ``state`` is the env's privileged dict: scene, pos, vel, yaw, time.
    """
    scene: Scene = state["scene"]
    pos, vel, yaw, t = state["pos"], state["vel"], state["yaw"], state["time"]
    ex, dr = cfg.expert, cfg.drone
    a, m = a_brake(cfg), dr.margin
    cruise = ex.cruise_speed or dr.v_max

    s, _ = _project(path, pos[:2])
    cur_speed = float(np.linalg.norm(vel[:2]))
    look = float(np.clip(0.4 + 0.15 * cur_speed, 0.5, ex.lookahead))
    target = _point_at(path, s + look)
    to_t = target - pos[:2]
    dist_goal = float(np.linalg.norm(scene.goal_xy - pos[:2]))
    direction = to_t / max(float(np.linalg.norm(to_t)), 1e-9)

    speed = min(cruise, math.sqrt(2 * a * 0.5 * max(dist_goal - 0.3, 0.0)) + 0.3)
    speed = min(speed, _corner_speed(path, s, a))
    if ex.speed_profile:
        d = cone_clearance(scene, pos, t, np.append(direction, 0.0), cfg.budget.cone_deg, dr.radius,
                           cfg.sim.max_range)
        speed = min(speed, safe_speed(ex.assumed_latency, d, a, m))
    v_des = direction * speed

    mv = scene.mover
    if mv is not None and t >= mv.t_start - 1.0:
        rel = pos[:2] - mv.pos(t)
        vrel = vel[:2] - mv.vel
        vv = float(vrel @ vrel)
        tca = -float(rel @ vrel) / vv if vv > 1e-9 else 0.0
        if 0.0 < tca < 2.5:
            closest = rel + vrel * tca
            if np.linalg.norm(closest) < mv.r + dr.radius + 1.5:
                perp = np.array([-mv.vel[1], mv.vel[0]]) / max(float(np.linalg.norm(mv.vel)), 1e-9)
                side = 1.0 if closest @ perp >= 0 else -1.0
                v_des = v_des + side * perp * 4.0
                n = float(np.linalg.norm(v_des))
                if n > cruise:
                    v_des *= cruise / n

    vz = 1.5 * (cfg.sim.altitude - pos[2])
    if np.linalg.norm(v_des) > 0.5:
        heading = math.atan2(v_des[1], v_des[0])
        err = (heading - yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_rate = float(np.clip(2.0 * err, -dr.yaw_rate_max, dr.yaw_rate_max))
    else:
        yaw_rate = 0.0
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([cy * v_des[0] + sy * v_des[1], -sy * v_des[0] + cy * v_des[1], vz, yaw_rate])


class Expert:
    """Stateful wrapper: plans on reset, replans when knocked more than 1.5 m off the path."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.inflate = cfg.drone.radius + cfg.drone.margin + 0.2
        self.path: Optional[Path] = None

    def reset(self, scene: Scene) -> bool:
        self.scene = scene
        self.path = plan_path(scene, scene.start[:2], scene.goal_xy, self.cfg.expert.grid_res, self.inflate)
        return self.path is not None

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        st = obs["privileged"]
        if self.path is None:
            raise RuntimeError("no path: reset() failed or was not called")
        _, off = _project(self.path, st["pos"][:2])
        if off > 1.5:
            new = plan_path(self.scene, st["pos"][:2], self.scene.goal_xy, self.cfg.expert.grid_res, self.inflate)
            if new is not None:
                self.path = new
        return expert_action(st, self.path, self.cfg)
