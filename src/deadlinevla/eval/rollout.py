"""Closed-loop rollouts. Fixed seeds per scene, so every method faces identical scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from deadlinevla.config import Config
from deadlinevla.serve.pilot import Pilot
from deadlinevla.sim.drone_env import DroneEnv
from deadlinevla.sim.expert import plan_path
from deadlinevla.sim.scenes import Scene, build_scene


@dataclass
class EpisodeLog:
    method: str
    governor: bool
    family: str
    seed: int
    speed_cap: float
    success: bool
    collision: bool
    timeout: bool
    time: float
    distance: float
    a_brake: float
    margin: float
    stalls: int
    dt: float = 0.1
    ticks: dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "ticks"}
        d["n_ticks"] = int(len(self.ticks.get("speed", [])))
        return d


def build_eval_scenes(cfg: Config, families: Optional[Sequence[str]] = None,
                      per_family: Optional[int] = None) -> list[Scene]:
    """Deterministic evaluation scenes (seeds from ``eval.seed_offset``), all solvable by the expert."""
    scenes = []
    inflate = cfg.drone.radius + cfg.drone.margin + 0.2
    for fam in families or cfg.scenes.families:
        seed, got = cfg.eval.seed_offset, 0
        while got < (per_family or cfg.eval.family_episodes.get(fam, cfg.eval.episodes_per_cell)):
            sc = build_scene(fam, seed, cfg)
            seed += 1
            if plan_path(sc, sc.start[:2], sc.goal_xy, cfg.expert.grid_res, inflate) is not None:
                scenes.append(sc)
                got += 1
    return scenes


def run_episodes(pilot: Pilot, env: DroneEnv, scenes: Sequence[Scene], speed_cap: float) -> list[EpisodeLog]:
    pilot.set_speed_cap(speed_cap)
    cfg = env.cfg
    max_ticks = int(cfg.sim.max_episode_s * cfg.drone.ctrl_hz) + 1
    logs = []
    for scene in scenes:
        obs = env.reset(scene)
        pilot.reset()
        info = obs["info"]
        for _ in range(max_ticks):
            obs, info = env.step(pilot.tick(obs))
            if info["done"]:
                break
        ticks = {k: np.asarray(v) for k, v in pilot.log.items()}
        logs.append(EpisodeLog(
            method=pilot.mode, governor=pilot.use_governor, family=scene.family, seed=scene.seed,
            speed_cap=float(speed_cap), success=bool(info["goal_reached"]), collision=bool(info["collision"]),
            timeout=bool(info["timeout"]), time=float(info["time"]), distance=float(info["distance"]),
            a_brake=pilot.a, margin=cfg.drone.margin, stalls=pilot.executor.stalls, dt=cfg.dt, ticks=ticks))
    return logs
