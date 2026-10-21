"""Replay evaluation episodes and animate them side by side.

``replay_cell`` re-flies one evaluation cell exactly as ``eval/sweep.py::run_cell`` did: same Pilot, same
latency RNG (seeded per speed cap and consumed across the cell's scenes in order), same scenes. It records
the drone, the threat and the allocation every tick. Replays are checked against the logged outcomes, so a
GIF never shows a different episode from the one in the tables.

``animate_side_by_side`` draws several methods on the same scene and clock: top-down obstacles, the drone,
each camera's view wedge coloured by its tier with radius equal to its detection range, the threat and a
speed strip. Pillow writes the GIF (no ffmpeg needed).
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from deadlinevla.config import Config
from deadlinevla.sim.scenes import PALETTE, Scene


@dataclass
class Trace:
    method: str
    family: str
    seed: int
    success: bool
    collision: bool
    time: float
    pos: np.ndarray            # (T, 3)
    yaw: np.ndarray            # (T,)
    mover: Optional[np.ndarray]  # (T, 2) or None
    alloc: np.ndarray          # (T, V)
    tokens: np.ndarray         # (T,)
    latency: np.ndarray        # (T,) predicted latency of the allocation in force, s
    speed: np.ndarray          # (T,)
    outcome: str = ""           # goal reached | COLLISION | out of bounds | timeout
    scene: Scene = field(repr=False, default=None)


def replay_cell(cfg: Config, scenes: Sequence[Scene], method: str, cap: float, table_path: str | Path,
                governor: bool = True, keep: Optional[set[tuple[str, int]]] = None) -> list[Trace]:
    """Re-fly a whole cell (all scenes, in order) and return a Trace per scene (trajectories only for ``keep``)."""
    from deadlinevla.budget.latency_model import LatencyModel, LatencyTable, make_latency_source
    from deadlinevla.policy.base import make_policy
    from deadlinevla.serve.pilot import Pilot
    from deadlinevla.serve.runtime import SimulatedRuntime
    from deadlinevla.sim.drone_env import make_env

    table = LatencyTable.load(table_path)
    latency = LatencyModel(table, cfg.budget.latency_stat, cfg.budget.max_residual, cfg.tier_tokens)
    source = make_latency_source(cfg.serve.board, cfg.tier_tokens, table)
    adapter = make_policy(cfg)
    env = make_env(cfg)
    rng = np.random.default_rng([cfg.run.seed, int(round(cap * 100))])      # as eval/sweep.py::run_cell
    pilot = Pilot(cfg, SimulatedRuntime(adapter, source, rng), latency, method, governor, adapter)
    pilot.set_speed_cap(cap)
    max_ticks = int(cfg.sim.max_episode_s * cfg.drone.ctrl_hz) + 1
    traces = []
    for scene in scenes:
        want = keep is None or (scene.family, scene.seed) in keep
        obs = env.reset(scene)
        pilot.reset()
        rec: dict[str, list] = {"pos": [], "yaw": [], "mover": [], "alloc": [], "lat": []}
        info = obs["info"]
        for _ in range(max_ticks):
            action = pilot.tick(obs)
            if want:
                rec["pos"].append(env.pos.copy())
                rec["yaw"].append(env.yaw)
                rec["mover"].append(scene.mover.pos(env.t) if scene.mover is not None else None)
                rec["alloc"].append(pilot.alloc.copy())
                rec["lat"].append(pilot.L_pred)
            obs, info = env.step(action)
            if info["done"]:
                break
        lg = pilot.log
        outcome = ("goal reached" if info["goal_reached"] else "COLLISION" if info["collision"]
                   else "out of bounds" if info.get("out_of_bounds") else "timeout")
        tr = Trace(method=method, family=scene.family, seed=scene.seed, success=bool(info["goal_reached"]),
                   collision=bool(info["collision"]), time=float(info["time"]), pos=np.zeros((0, 3)),
                   yaw=np.zeros(0), mover=None, alloc=np.zeros((0, len(cfg.cameras.cam_yaws))),
                   tokens=np.zeros(0), latency=np.zeros(0), speed=np.zeros(0), outcome=outcome, scene=scene)
        if want:
            tr.pos = np.array(rec["pos"])
            tr.yaw = np.array(rec["yaw"])
            tr.mover = np.array(rec["mover"]) if scene.mover is not None else None
            tr.alloc = np.array(rec["alloc"])
            tr.tokens = np.asarray(lg["tokens"], dtype=float)
            tr.latency = np.array(rec["lat"])
            tr.speed = np.asarray(lg["speed"], dtype=float)
        traces.append(tr)
    return traces


def check_against_log(traces: Sequence[Trace], episodes: Sequence[dict[str, Any]], cap: float) -> None:
    """Every replayed outcome must equal the logged one (method, scene, cap); raise otherwise."""
    logged = {(e["method"], e["family"], e["seed"]): e for e in episodes
              if e["speed_cap"] == cap and e["governor"]}
    for tr in traces:
        e = logged.get((tr.method, tr.family, tr.seed))
        if e is None:
            raise KeyError(f"no logged episode for {tr.method} {tr.family} seed {tr.seed} at cap {cap}")
        if (e["success"], e["collision"]) != (tr.success, tr.collision) or abs(e["time"] - tr.time) > 1e-6:
            raise AssertionError(f"replay differs from the log for {tr.method} {tr.family} seed {tr.seed}: "
                                 f"logged {e['success']}/{e['collision']}/{e['time']:.2f}, "
                                 f"replayed {tr.success}/{tr.collision}/{tr.time:.2f}")


# --------------------------------------------------------------------------------------------------
# Animation

TIER_FILL = {1: "#c6dbef", 2: "#6baed6", 3: "#2171b5"}        # one hue, light -> dark (sequential)
OBST, OBST_SEEN, THREAT = "#a8a7a1", "#52514e", "#e34948"


def _detected(trace: Trace, k: int, cfg: Config) -> np.ndarray:
    from deadlinevla.perception import visible

    xy, r, _ = trace.scene.circles(k * cfg.dt)
    if len(xy) == 0:
        return np.zeros(0, dtype=bool)
    rel = xy - trace.pos[k, :2]
    c, s = math.cos(trace.yaw[k]), math.sin(trace.yaw[k])
    p_body = rel @ np.array([[c, s], [-s, c]]).T
    return visible(p_body, 2 * r, np.linalg.norm(rel, axis=1) - r, trace.alloc[k], cfg)


def animate_side_by_side(traces: Sequence[Trace], titles: Sequence[str], cfg: Config, out: str | Path,
                         headline: str, footer: str, fps: int = 10, max_frames: int = 140,
                         wedge_size: float = 0.5, dpi: int = 72) -> Path:
    """One panel per trace (same scene), common clock, written as a GIF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle, Polygon, Wedge

    from deadlinevla.eval.report import INK, INK_2, SURFACE, _style_axes
    from deadlinevla.perception import detection_range

    n = len(traces)
    scene = traces[0].scene
    T = max(len(t.pos) for t in traces)
    frames = np.unique(np.linspace(0, T - 1, min(T, max_frames)).round().astype(int))
    det = detection_range(np.arange(4), np.full(4, wedge_size), cfg)
    yaws = np.radians(np.asarray(cfg.cameras.cam_yaws, dtype=float))
    half = math.degrees(math.radians(cfg.cameras.fov) / 2)

    pts = np.vstack([t.pos[:, :2] for t in traces] + [scene.goal_xy[None]]
                    + [t.mover for t in traces if t.mover is not None])
    lo, hi = pts.min(0) - 5.0, pts.max(0) + 5.0
    span = max(hi - lo)
    centre = (lo + hi) / 2
    lo, hi = centre - span / 2, centre + span / 2

    fig = plt.figure(figsize=(4.2 * n, 5.6), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, n, height_ratios=[4.2, 1.0], hspace=0.28, wspace=0.12,
                          left=0.03, right=0.99, top=0.84, bottom=0.16)
    fig.text(0.01, 0.965, headline, color=INK, fontsize=12, ha="left", va="top", weight="bold")
    fig.text(0.01, 0.915, "simulation · synthetic latency · surrogate policy · wedges: camera view, colour = "
             f"tier, length = detection range for a {wedge_size:g} m object", color=INK_2, fontsize=8,
             ha="left", va="top")
    fig.text(0.01, 0.01, "\n".join(textwrap.wrap(footer, 60 * n)), color=INK_2, fontsize=8, ha="left",
             va="bottom")

    panels = []
    vmax = max(float(t.speed.max()) for t in traces if len(t.speed)) * 1.1 + 0.1
    for i, (tr, title) in enumerate(zip(traces, titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(SURFACE)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#e4e3df")
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        xy, r, _ = scene.circles(0.0)
        n_static = len(scene.obs_xy)
        discs = [ax.add_patch(Circle(xy[j], r[j], color=OBST, lw=0, zorder=2)) for j in range(n_static)]
        ax.plot(*scene.goal_xy, marker="*", markersize=16, color=tuple(c / 255 for c in PALETTE[scene.goal_color]),
                markeredgecolor=INK, markeredgewidth=0.6, zorder=5)
        wedges = [ax.add_patch(Wedge((0, 0), 1, 0, 1, color=TIER_FILL[1], alpha=0.0, lw=0, zorder=1))
                  for _ in yaws]
        trail, = ax.plot([], [], color=INK, lw=1.2, alpha=0.6, zorder=3)
        drone = ax.add_patch(Polygon(np.zeros((3, 2)), closed=True, color=INK, zorder=6))
        threat = ax.add_patch(Circle((0, 0), max(scene.mover.r, span / 70) if scene.mover else 0.1, color=THREAT, lw=0,
                                     alpha=1.0 if scene.mover else 0.0, zorder=4))
        threat_trail, = ax.plot([], [], color=THREAT, lw=1.0, alpha=0.5, ls="--", zorder=3)
        status = ax.text(0.03, 0.03, "", transform=ax.transAxes, fontsize=10, color=INK, va="bottom",
                         weight="bold")
        info = ax.text(0.97, 0.97, "", transform=ax.transAxes, fontsize=8, color=INK, va="top", ha="right",
                       family="monospace")
        sx = fig.add_subplot(gs[1, i])
        _style_axes(sx)
        sx.set_xlim(0, T * cfg.dt)
        sx.set_ylim(0, vmax)
        sx.set_ylabel("speed m/s", color=INK_2, fontsize=7)
        sx.tick_params(labelsize=7)
        sline, = sx.plot([], [], color="#2a78d6", lw=1.5)
        panels.append(dict(tr=tr, discs=discs, wedges=wedges, trail=trail, drone=drone, threat=threat,
                           threat_trail=threat_trail, status=status, info=info, sline=sline))

    def draw(frame_idx):
        artists = []
        for p in panels:
            tr: Trace = p["tr"]
            k = min(frame_idx, len(tr.pos) - 1)
            x, y = tr.pos[k, :2]
            h = tr.yaw[k]
            for wdg, cam, tier in zip(p["wedges"], yaws, tr.alloc[k]):
                if tier == 0:
                    wdg.set_alpha(0.0)
                    continue
                c = math.degrees(h + cam)
                wdg.set_center((x, y))
                wdg.set_radius(float(det[int(tier)]))
                wdg.set_theta1(c - half)
                wdg.set_theta2(c + half)
                wdg.set_color(TIER_FILL[int(tier)])
                wdg.set_alpha(0.35)
            seen = _detected(tr, k, cfg)
            for j, disc in enumerate(p["discs"]):
                disc.set_color(OBST_SEEN if j < len(seen) and seen[j] else OBST)
            p["trail"].set_data(tr.pos[: k + 1, 0], tr.pos[: k + 1, 1])
            size = span / 45
            tri = np.array([[1.4, 0.0], [-0.8, 0.8], [-0.8, -0.8]]) * size
            rot = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
            p["drone"].set_xy(tri @ rot.T + [x, y])
            if tr.mover is not None:
                p["threat"].set_center(tuple(tr.mover[k]))
                p["threat_trail"].set_data(tr.mover[: k + 1, 0], tr.mover[: k + 1, 1])
            p["info"].set_text(f"{tr.tokens[k]:4.0f} tok\n{tr.latency[k] * 1000:4.0f} ms\n{tr.speed[k]:4.1f} m/s")
            t_axis = np.arange(k + 1) * cfg.dt
            p["sline"].set_data(t_axis, tr.speed[: k + 1])
            if k == len(tr.pos) - 1:
                p["status"].set_text(tr.outcome)
                p["status"].set_color("#008300" if tr.success else "#e34948")
            else:
                p["status"].set_text(f"t = {k * cfg.dt:4.1f} s")
                p["status"].set_color(INK)
            artists += [*p["wedges"], p["trail"], p["drone"], p["threat"], p["threat_trail"], p["status"],
                        p["info"], p["sline"], *p["discs"]]
        return artists

    hold = [frames[-1]] * fps                     # hold the last frame for a second
    anim = FuncAnimation(fig, draw, frames=list(frames) + hold, blit=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out
