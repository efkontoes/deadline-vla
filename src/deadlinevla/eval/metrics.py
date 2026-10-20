"""Episode metrics, including the deadline-violation rate and what causes each violation.

A tick violates the deadline when v L + v^2/(2a) > d - m, where L is the latency of the inference
whose chunk the drone is flying (the doc's L_actual) and d is clearance along the current velocity.
Only moving ticks count (v >= hover speed): when hovering the deadline is undefined (L_max = inf),
and sitting inside the margin is a clearance event, not a latency one.

Every violating tick is assigned the first cause that applies, in this order:

    ungoverned          governor off and the drone flies faster than v*
    clearance_jump      d fell faster than the drone's own motion explains: an obstacle entered the cone
    direction_mismatch  the governor judged the commanded direction, which had much more room than the
                        velocity direction the metric uses (the drone is turning)
    dynamics_lag        the command is capped but the drone is still shedding speed from earlier ticks
    latency_tail        the inference took longer than its predicted (p95) latency
    latency_switch      the flown chunk came from a slower allocation than the one the governor is now
                        assuming (the allocation changed while the old chunk is still being flown)
    staleness           the flown action is older than the latency the governor assumed (chunk play-out
                        or a held action after the chunk ran out)
    other               none of the above: a governor or metric bug would show up here
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from deadlinevla.eval.rollout import EpisodeLog

HOVER_SPEED = 0.3
CAUSES = ("ungoverned", "clearance_jump", "direction_mismatch", "dynamics_lag", "latency_tail",
          "latency_switch", "staleness", "other")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class Metrics:
    episodes: int
    success_rate: float
    success_ci: tuple[float, float]
    collision_rate: float
    collision_ci: tuple[float, float]
    collisions_per_km: float
    time_to_goal: float          # mean over successful episodes, s (nan if none)
    mean_speed: float            # m/s over all ticks
    tokens_per_tick: float
    violation_rate: float        # doc definition, over moving ticks
    violation_rate_cmd: float    # same inequality with clearance along the command (what the governor sees)
    violation_rate_age: float    # same inequality with the real information age instead of the latency
    violation_rate_rel: float    # v3 relative deadline over all ticks: some contact's closing speed too high
    violation_causes: dict = field(default_factory=dict)   # cause -> fraction of violating ticks
    mean_latency: float = 0.0    # s, actual, per request
    mean_age: float = 0.0        # s, information age of flown actions
    stall_rate: float = 0.0
    km: float = 0.0

    def to_dict(self) -> dict:
        out = {}
        for k, v in asdict(self).items():
            if isinstance(v, float) and np.isnan(v):
                v = None
            out[k] = list(v) if isinstance(v, tuple) else v
        return out


def _moving(log: EpisodeLog) -> np.ndarray:
    return log.ticks["speed"] >= HOVER_SPEED


def _violates(v, L, d, a, m) -> np.ndarray:
    return v * L + v ** 2 / (2 * a) > d - m


def violation_flags(log: EpisodeLog) -> np.ndarray:
    t = log.ticks
    return _moving(log) & _violates(t["speed"], t["L_src"], t["clearance"], log.a_brake, log.margin)


def violation_causes(log: EpisodeLog) -> np.ndarray:
    """Cause label per tick ('' where the tick does not violate)."""
    t = log.ticks
    viol = violation_flags(log)
    v, d = t["speed"], t["clearance"]
    d_prev = np.concatenate([[d[0]], d[:-1]])
    v_prev = np.concatenate([[v[0]], v[:-1]])
    tests = [
        ("ungoverned", (not log.governor) & (v > t["v_star"] + 0.1)),
        ("clearance_jump", d_prev - d > v_prev * log.dt + 0.25),
        ("direction_mismatch", t["d_cmd"] > d + 0.5),
        ("dynamics_lag", v > t["v_star"] + 0.1),
        ("latency_tail", t["L_src"] > t["L_pred_src"] + 1e-9),
        ("latency_switch", t["L_src"] > t["L_pred"] + log.dt + 1e-9),
        ("staleness", t["age"] > t["L_pred_src"] + log.dt + 1e-9),
    ]
    out = np.full(len(v), "", dtype=object)
    todo = viol.copy()
    for name, cond in tests:
        hit = todo & np.asarray(cond, dtype=bool)
        out[hit] = name
        todo &= ~hit
    out[todo] = "other"
    return out


def summarize(logs: Sequence[EpisodeLog]) -> Metrics:
    if not logs:
        raise ValueError("no episodes to summarise")
    n = len(logs)
    km = sum(lg.distance for lg in logs) / 1000.0
    collisions = sum(lg.collision for lg in logs)
    successes = sum(lg.success for lg in logs)
    succ_times = [lg.time for lg in logs if lg.success]
    cat = lambda key: np.concatenate([lg.ticks[key] for lg in logs])  # noqa: E731
    speeds, tokens = cat("speed"), cat("tokens")
    moving = np.concatenate([_moving(lg) for lg in logs])
    viol = np.concatenate([violation_flags(lg) for lg in logs])
    viol_cmd = np.concatenate([_moving(lg) & _violates(lg.ticks["speed"], lg.ticks["L_src"], lg.ticks["d_cmd"],
                                                        lg.a_brake, lg.margin) for lg in logs])
    viol_age = np.concatenate([_moving(lg) & _violates(lg.ticks["speed"], lg.ticks["age"], lg.ticks["clearance"],
                                                        lg.a_brake, lg.margin) for lg in logs])
    causes = np.concatenate([violation_causes(lg) for lg in logs])
    n_viol = int(viol.sum())
    breakdown = {c: (float((causes == c).sum() / n_viol) if n_viol else 0.0) for c in CAUSES}
    lat = np.concatenate([lg.ticks["L_act"][lg.ticks["requested"].astype(bool)] for lg in logs])
    n_mov = max(int(moving.sum()), 1)
    return Metrics(
        episodes=n,
        success_rate=successes / n, success_ci=wilson(successes, n),
        collision_rate=collisions / n, collision_ci=wilson(collisions, n),
        collisions_per_km=collisions / km if km > 0 else float("nan"),
        time_to_goal=float(np.mean(succ_times)) if succ_times else float("nan"),
        mean_speed=float(speeds.mean()) if len(speeds) else 0.0,
        tokens_per_tick=float(tokens.mean()) if len(tokens) else 0.0,
        violation_rate=float(viol.sum() / n_mov),
        violation_rate_cmd=float(viol_cmd.sum() / n_mov),
        violation_rate_age=float(viol_age.sum() / n_mov),
        violation_rate_rel=float(np.mean(np.concatenate([lg.ticks["rel_slack"] < 0 for lg in logs])))
        if all("rel_slack" in lg.ticks for lg in logs) else float("nan"),
        violation_causes=breakdown,
        mean_latency=float(lat.mean()) if len(lat) else 0.0,
        mean_age=float(cat("age")[moving].mean()) if moving.any() else 0.0,
        stall_rate=sum(lg.stalls for lg in logs) / max(len(speeds), 1),
        km=km,
    )
