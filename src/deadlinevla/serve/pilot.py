"""The per-tick control loop: the only place the budget module meets the model.

Every tick: (1) estimate clearance, (2) compute the deadline, (3) weight views and allocate,
(4) hand obs and allocation to the executor, (5) take the executor's next action and govern it.
Baselines run through the same loop by swapping step (3), so every method shares one code path.

Modes: ``deadline`` uses ``budget.objective`` (v2 ``speed`` by default); ``deadline_v1`` always uses
the original quality objective, kept as an ablation; the rest are baselines B1-B4.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np

from deadlinevla.budget.allocator import camera_axes, choose_allocation, quality_table, view_weights
from deadlinevla.budget.clearance import estimate_clearance
from deadlinevla.budget.deadline import max_latency
from deadlinevla.budget.governor import cap_speed, govern, safe_speed
from deadlinevla.budget.latency_model import LatencyModel
from deadlinevla.config import Config, a_brake, set_field, tau_track
from deadlinevla.eval.baselines import BASELINES, baseline_alloc
from deadlinevla.policy.base import PolicyAdapter
from deadlinevla.serve.executor import ChunkExecutor

MODES = ("deadline", "deadline_v1") + BASELINES


class Pilot:
    def __init__(self, cfg: Config, runtime: Any, latency: LatencyModel, mode: str = "deadline",
                 use_governor: bool = True, adapter: Optional[PolicyAdapter] = None):
        if mode not in MODES:
            raise ValueError(f"unknown pilot mode {mode!r}; known: {MODES}")
        self.cfg = cfg
        self.latency = latency
        self.mode = mode
        self.use_governor = use_governor
        self.adapter = adapter
        self.a = a_brake(cfg)
        self.tau = tau_track(cfg) if cfg.budget.governor_delays else 0.0
        self.axes = camera_axes(cfg.cameras.cam_yaws)
        self.quality = quality_table(cfg.budget.quality_fn, cfg.tier_tokens)
        self.tokens = np.asarray(cfg.tier_tokens)
        self.executor = ChunkExecutor(runtime, cfg.dt, cfg.serve.mode)
        self.reset()

    def set_speed_cap(self, v_max: float) -> None:
        self.cfg = set_field(self.cfg, "drone.v_max", float(v_max))
        if self.adapter is not None:
            self.adapter.set_speed_cap(v_max)

    def reset(self) -> None:
        self.executor.reset()
        self.alloc = np.zeros(self.cfg.cameras.n_cams, dtype=np.int64)
        self.L_pred = 0.0
        self.L_act = 0.0
        self.log: dict[str, list] = defaultdict(list)

    def choose_alloc(self, v_body: np.ndarray, L_max: float) -> tuple[np.ndarray, float]:
        b = self.cfg.budget
        if self.mode in ("deadline", "deadline_v1"):
            w = view_weights(v_body, self.axes, b.kappa, b.epsilon, b.hover_speed)
            objective = b.objective if self.mode == "deadline" else "quality"
            return choose_allocation(w, L_max, self.latency, self.quality, objective, b.qmin_mid_weight)
        imp = self.adapter.view_importance() if self.adapter is not None else None
        alloc = baseline_alloc(self.mode, {"importance": imp}, self.latency, self.cfg)
        return alloc, self.latency.predict(alloc)

    def tick(self, obs: dict[str, Any], state: Optional[dict[str, Any]] = None) -> np.ndarray:
        cfg = self.cfg
        v_body = np.asarray(obs["vel_body"], dtype=np.float64)
        speed = float(np.linalg.norm(v_body))
        v_hat = v_body / speed if speed >= cfg.budget.hover_speed else None
        # (1) clearance and (2) deadline, before any pixel reaches the model
        d = estimate_clearance(obs, v_hat, cfg)
        L_max = max_latency(speed, d, self.a, cfg.drone.margin)
        # (3) allocation and (4) inference, whenever the executor can take a request
        requested = self.executor.can_request()
        if requested:
            self.alloc, self.L_pred = self.choose_alloc(v_body, L_max)
            self.L_act = self.executor.request(obs, self.alloc, {"L_pred": self.L_pred})
        # (5) next action, governed with the latency of the allocation actually chosen and the
        # clearance in the direction the command would take the drone (so it can always back away)
        action, stalled = self.executor.next_action()
        n = float(np.linalg.norm(action[:3]))
        d_cmd = estimate_clearance(obs, action[:3] / n, cfg) if n >= cfg.budget.hover_speed else d
        # The governor acts once per control period and the controller tracks its command with a lag, so
        # its blind time is L + dt + tau_track: with L alone, v* falls faster than the drone sheds speed.
        L_gov = self.L_pred + (cfg.dt + self.tau if cfg.budget.governor_delays else 0.0)
        # While moving, a turn sweeps from the current velocity to the command, so both directions must be
        # safe. At hover speed only the command counts, so a drone inside the margin can still back away.
        d_gov = d_cmd if v_hat is None else min(d_cmd, d)
        v_star = min(safe_speed(L_gov, d_gov, self.a, cfg.drone.margin), cfg.drone.v_max)
        out = govern(action, L_gov, d_gov, cfg) if self.use_governor else cap_speed(action, cfg.drone.v_max)
        ex = self.executor

        lg = self.log
        lg["speed"].append(speed)
        lg["clearance"].append(d)
        lg["L_max"].append(L_max)
        lg["alloc"].append(self.alloc.copy())
        lg["tokens"].append(int(self.tokens[self.alloc].sum()))
        lg["L_pred"].append(self.L_pred)
        lg["L_act"].append(self.L_act)
        lg["L_src"].append(ex.source_latency)                    # latency of the inference being flown
        lg["L_pred_src"].append(ex.source_meta.get("L_pred", 0.0))
        lg["age"].append(ex.information_age())                   # real blind time of the flown action
        lg["d_cmd"].append(d_cmd)
        lg["v_star"].append(v_star)
        lg["requested"].append(requested)
        lg["stalled"].append(stalled)
        lg["v_cmd"].append(float(np.linalg.norm(action[:3])))
        lg["v_gov"].append(float(np.linalg.norm(out[:3])))
        return out
