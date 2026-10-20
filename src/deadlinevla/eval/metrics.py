"""Episode metrics, including the deadline-violation rate.

A tick violates the deadline when v L_actual + v^2/(2a) > d - m, with L_actual the latency of the
inference whose result the drone is flying on. Only moving ticks count (v >= hover speed): when
hovering the deadline is undefined (L_max = inf), and sitting inside the margin is a clearance event,
not a latency one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from deadlinevla.eval.rollout import EpisodeLog

HOVER_SPEED = 0.3


@dataclass
class Metrics:
    episodes: int
    success_rate: float
    collision_rate: float
    collisions_per_km: float
    time_to_goal: float          # mean over successful episodes, s (nan if none)
    mean_speed: float            # m/s over all ticks
    tokens_per_tick: float
    violation_rate: float
    mean_latency: float          # s, actual
    stall_rate: float
    km: float

    def to_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in asdict(self).items()}


def violation_flags(log: EpisodeLog) -> np.ndarray:
    t = log.ticks
    v, L, d = t["speed"], t["L_act"], t["clearance"]
    return (v >= HOVER_SPEED) & (v * L + v ** 2 / (2 * log.a_brake) > d - log.margin)


def summarize(logs: Sequence[EpisodeLog]) -> Metrics:
    if not logs:
        raise ValueError("no episodes to summarise")
    km = sum(lg.distance for lg in logs) / 1000.0
    collisions = sum(lg.collision for lg in logs)
    succ = [lg.time for lg in logs if lg.success]
    speeds = np.concatenate([lg.ticks["speed"] for lg in logs])
    tokens = np.concatenate([lg.ticks["tokens"] for lg in logs])
    viol = np.concatenate([violation_flags(lg) for lg in logs])
    lat = np.concatenate([lg.ticks["L_act"][lg.ticks["requested"].astype(bool)] for lg in logs])
    n_ticks = len(speeds)
    return Metrics(
        episodes=len(logs),
        success_rate=float(np.mean([lg.success for lg in logs])),
        collision_rate=collisions / len(logs),
        collisions_per_km=collisions / km if km > 0 else float("nan"),
        time_to_goal=float(np.mean(succ)) if succ else float("nan"),
        mean_speed=float(speeds.mean()) if n_ticks else 0.0,
        tokens_per_tick=float(tokens.mean()) if n_ticks else 0.0,
        violation_rate=float(viol.mean()) if n_ticks else 0.0,
        mean_latency=float(lat.mean()) if len(lat) else 0.0,
        stall_rate=sum(lg.stalls for lg in logs) / max(n_ticks, 1),
        km=km,
    )
