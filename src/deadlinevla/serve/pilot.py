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
from deadlinevla.budget.relative import (
    Contacts,
    camera_closing,
    closing_weights,
    contacts_from_cameras,
    contacts_from_kinematics,
    depth_camera_clearances,
    govern_relative,
    min_slack,
    relative_deadline,
)
from deadlinevla.config import Config, a_brake, set_field, tau_track
from deadlinevla.eval.baselines import BASELINES, baseline_alloc
from deadlinevla.perception import visible
from deadlinevla.policy.base import PolicyAdapter
from deadlinevla.serve.executor import ChunkExecutor
from deadlinevla.sim.scenes import obstacle_kinematics

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
        self._prev_d_cam: Optional[np.ndarray] = None
        self.log: dict[str, list] = defaultdict(list)

    def choose_alloc(self, v_body: np.ndarray, L_max: float,
                     weights: Optional[np.ndarray] = None) -> tuple[np.ndarray, float]:
        b = self.cfg.budget
        if self.mode in ("deadline", "deadline_v1"):
            w = weights if weights is not None else view_weights(v_body, self.axes, b.kappa, b.epsilon,
                                                                 b.hover_speed)
            objective = b.objective if self.mode == "deadline" else "quality"
            return choose_allocation(w, L_max, self.latency, self.quality, objective, b.qmin_mid_weight)
        imp = self.adapter.view_importance() if self.adapter is not None else None
        alloc = baseline_alloc(self.mode, {"importance": imp}, self.latency, self.cfg)
        return alloc, self.latency.predict(alloc)

    def contacts(self, obs: dict[str, Any], v_body: np.ndarray) -> Contacts:
        """Everything whose clearance can shrink: ground truth per obstacle, or per camera from depth."""
        cfg = self.cfg
        priv = obs.get("privileged")
        if cfg.budget.clearance_source == "oracle" and priv is not None:
            kin = obstacle_kinematics(priv["scene"], priv["pos"], priv["vel"], priv["time"], cfg.drone.radius,
                                      cfg.sim.max_range)
            return contacts_from_kinematics(kin, priv["yaw"])
        d_cam, c_cam = depth_camera_clearances(
            np.asarray(obs["depth"]), float(obs["state"][6]), cfg.cameras.fov, cfg.cameras.cam_yaws,
            cfg.budget.cone_deg, cfg.drone.radius, cfg.sim.max_range, cfg.budget.floor_ignore_z,
            self._prev_d_cam, cfg.dt)
        self._prev_d_cam = d_cam
        con = contacts_from_cameras(d_cam, c_cam, self.axes, v_body, cfg.sim.max_range)
        con.size = np.full(len(con), cfg.perception.nominal_size)
        return con

    def seen(self, con: Contacts, alloc: np.ndarray) -> Contacts:
        """The contacts some camera detects at its tier under ``alloc`` (all of them for the oracle governor)."""
        if self.cfg.perception.governor == "oracle":
            return con
        return con.subset(visible(con.p, con.size, con.d, alloc, self.cfg))

    def tick(self, obs: dict[str, Any], state: Optional[dict[str, Any]] = None) -> np.ndarray:
        cfg, b = self.cfg, self.cfg.budget
        v_body = np.asarray(obs["vel_body"], dtype=np.float64)
        speed = float(np.linalg.norm(v_body))
        v_hat = v_body / speed if speed >= b.hover_speed else None
        # (1) clearance and (2) deadline, before any pixel reaches the model
        d = estimate_clearance(obs, v_hat, cfg)
        weights = None
        if b.relative:
            # v3: every contact's closing speed counts, so a threat from behind tightens the deadline
            # The deadline and the weights can only use what the cameras detected with the allocation in force
            # (perception.governor = camera); the oracle ablation sees everything.
            con = self.contacts(obs, v_body)
            con_prev = self.seen(con, self.alloc)
            L_max = relative_deadline(con_prev, v_body, self.a, cfg.drone.margin, b.cone_deg)
            weights = closing_weights(camera_closing(con_prev, v_body, self.axes), b.kappa, b.epsilon,
                                      b.hover_speed)
        else:
            L_max = max_latency(speed, d, self.a, cfg.drone.margin)
        # (3) allocation and (4) inference, whenever the executor can take a request
        requested = self.executor.can_request()
        if requested:
            self.alloc, self.L_pred = self.choose_alloc(v_body, L_max, weights)
            self.L_act = self.executor.request(obs, self.alloc, {"L_pred": self.L_pred})
        # (5) next action, governed with the latency of the allocation actually chosen
        action, stalled = self.executor.next_action()
        n = float(np.linalg.norm(action[:3]))
        d_cmd = estimate_clearance(obs, action[:3] / n, cfg) if n >= b.hover_speed else d
        # The governor acts once per control period and the controller tracks its command with a lag, so
        # its blind time is L + dt + tau_track: with L alone, v* falls faster than the drone sheds speed.
        L_gov = self.L_pred + (cfg.dt + self.tau if b.governor_delays else 0.0)
        # While moving, a turn sweeps from the current velocity to the command, so both directions must be
        # safe. At hover speed only the command counts, so a drone inside the margin can still back away.
        d_gov = d_cmd if v_hat is None else min(d_cmd, d)
        v_star = min(safe_speed(L_gov, d_gov, self.a, cfg.drone.margin), cfg.drone.v_max)
        if not self.use_governor:
            out = cap_speed(action, cfg.drone.v_max)
        elif b.relative:
            out = govern_relative(action, L_gov, self.seen(con, self.alloc), self.a, cfg.drone.margin,
                                  cfg.drone.v_max, v_now=v_body, cone_deg=b.cone_deg)
        else:
            out = govern(action, L_gov, d_gov, cfg)
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
        if b.relative:
            # relative-deadline slack with the flown chunk's latency (negative: violated), counted even when
            # the drone is still: a threat closing on a hovering drone is exactly what it must catch
            # measured against ground truth, so threats the cameras missed show up as violations
            lg["rel_slack"].append(min_slack(con, v_body, ex.source_latency, self.a, cfg.drone.margin, b.cone_deg))
            closing = (con.closing(v_body) > 1e-6) & con.threat(v_body, cfg.drone.margin, b.cone_deg)
            lg["unseen_threats"].append(int(closing.sum()) - int((closing & visible(
                con.p, con.size, con.d, self.alloc, cfg)).sum()) if cfg.perception.governor == "camera" else 0)
        return out
