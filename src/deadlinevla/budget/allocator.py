"""Method steps 4 and 5: direction weights and exact view allocation.

    w_j = eps + max(0, v_hat . u_j)^kappa

    r* = argmax_r sum_j w_j q(r_j)   s.t.   L(r) <= L_max(v, d),   r_j >= 1 for all j if feasible

With 4096 candidates and a precomputed latency vector the search is one small matrix-vector product.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np

from deadlinevla.budget.latency_model import LatencyModel


def camera_axes(cam_yaws_deg: Sequence[float]) -> np.ndarray:
    """(V, 3) unit optical axes in the body frame (x forward, y left, z up)."""
    yaw = np.radians(np.asarray(cam_yaws_deg, dtype=np.float64))
    return np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)


def view_weights(v_body: np.ndarray, cam_axes: np.ndarray, kappa: float, eps: float,
                 hover_speed: float = 0.3) -> np.ndarray:
    """Method step 4. Uniform weights below the hover threshold, where v_hat is undefined."""
    v = np.asarray(v_body, dtype=np.float64)[: cam_axes.shape[1]]
    speed = float(np.linalg.norm(v))
    if speed < hover_speed:
        return np.ones(len(cam_axes))
    v_hat = v / speed
    return eps + np.maximum(0.0, cam_axes @ v_hat) ** kappa


# --------------------------------------------------------------------------------------------------
# Tier quality q(r)

_QUALITY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "log1p_tokens": lambda n: np.log1p(n),
    "sqrt_tokens": lambda n: np.sqrt(n),
    "linear_tokens": lambda n: n.astype(np.float64),
}


def quality_table(name: str, tier_tokens: Sequence[int]) -> np.ndarray:
    """q(r) for r = 0..3, concave in tokens for the default ``log1p_tokens``."""
    if name not in _QUALITY:
        raise KeyError(f"unknown quality_fn {name!r}; known: {sorted(_QUALITY)}")
    return _QUALITY[name](np.asarray(tier_tokens, dtype=np.float64))


# --------------------------------------------------------------------------------------------------
# Exact search


def search(weights: np.ndarray, limit: float, allocs: np.ndarray, costs: np.ndarray,
           quality: np.ndarray, floor: bool = True) -> Optional[int]:
    """Index of the best allocation with cost <= limit, or None.

    With ``floor`` the all-views-on candidates are tried first. Ties go to the lower cost.
    """
    score = quality[allocs] @ np.asarray(weights, dtype=np.float64)
    feasible = costs <= limit
    masks = [feasible & (allocs >= 1).all(axis=1), feasible] if floor else [feasible]
    for mask in masks:
        idx = np.flatnonzero(mask)
        if len(idx):
            s = score[idx]
            best = idx[s >= s.max() - 1e-9]
            return int(best[np.argmin(costs[best])])
    return None


def required_quality(weights: np.ndarray, quality_fn: np.ndarray, mid_weight: float = 0.25) -> float:
    """Q_min = sum_j w_j q(r_req_j): views with w_j >= ``mid_weight`` need mid tier, the rest low.

    Weighting by w_j means that when flying fast the forward views must stay sharp, while the
    savings come from views facing away.
    """
    w = np.asarray(weights, dtype=np.float64)
    req = np.where(w >= mid_weight, 2, 1)
    return float(quality_fn[req] @ w)


def coverage_mask(allocs: np.ndarray, lat_all: np.ndarray, v_body: np.ndarray, cam_axes: np.ndarray,
                  det_by_tier: np.ndarray, o_max: float, a: float, m: float, extra_blind: float) -> np.ndarray:
    """Derived floor: can every camera see far enough to survive the worst credible threat from its side?

        c_j^max = max(0, v . u_j) + o_max                     worst closing speed toward camera j
        d_req,j(r) = m + tau(r) c_j^max + (c_j^max)^2 / 2a    detection distance needed to survive it
        tau(r) = L(r) + extra_blind                           blind time of this allocation
        covered  <=>  d_det(r_j) >= min(d_req,j(r), d_det(high))   for every j

    When even the high tier cannot see far enough the camera is required at high and the governor slows
    the drone, so all-high is always feasible. Returns a mask over ``allocs``.
    """
    c_max = np.maximum(0.0, cam_axes[:, :2] @ np.asarray(v_body, dtype=np.float64)[:2]) + o_max   # (V,)
    tau = np.asarray(lat_all, dtype=np.float64) + extra_blind                                     # (N,)
    d_req = m + tau[:, None] * c_max[None] + c_max[None] ** 2 / (2 * a)                           # (N, V)
    det = np.asarray(det_by_tier, dtype=np.float64)
    return (det[allocs] >= np.minimum(d_req, det[3]) - 1e-9).all(axis=1)


def allocate_for_speed(weights: np.ndarray, table: LatencyModel, quality_fn: np.ndarray,
                       mid_weight: float = 0.25, floor: Optional[np.ndarray] = None) -> tuple[np.ndarray, float]:
    """Method step 5, v2: speed is the objective, perception the constraint.

        r* = argmax_r v*(L(r), D)   s.t.   sum_j w_j q(r_j) >= Q_min

    v* falls monotonically with L for every D > 0 (and is 0 for all r when D <= 0), so this is the
    cheapest allocation that still sees enough; D drops out of the argmax. Ties go to higher quality.

    The 360 degree floor is a hard constraint. ``floor`` is a mask over allocations: the derived coverage
    floor (``coverage_mask``), or by default the fixed rule "every view at least low". Without any floor
    the weighted Q_min alone would switch rear views off and pay for it with a sharper front view, and a
    threat from behind would never reach the model.
    """
    allocs = table.allocs
    lat = table.predict_all()
    w = np.asarray(weights, dtype=np.float64)
    score = quality_fn[allocs] @ w
    fl = (allocs >= 1).all(axis=1) if floor is None else floor
    ok = (score >= required_quality(w, quality_fn, mid_weight) - 1e-9) & fl
    if not ok.any():
        ok = score >= required_quality(w, quality_fn, mid_weight) - 1e-9
    idx = np.flatnonzero(ok)
    cheapest = idx[lat[idx] <= lat[idx].min() + 1e-12]
    i = int(cheapest[np.argmax(score[cheapest])])
    return np.array(allocs[i], dtype=np.int64), float(lat[i])


def choose_allocation(weights: np.ndarray, L_max: float, table: LatencyModel, quality_fn: np.ndarray,
                      objective: str = "speed", mid_weight: float = 0.25,
                      floor: Optional[np.ndarray] = None) -> tuple[np.ndarray, float]:
    """Dispatch on ``budget.objective``: ``speed`` (v2) or ``quality`` (v1, the original doc)."""
    if objective == "speed":
        return allocate_for_speed(weights, table, quality_fn, mid_weight, floor)
    if objective == "quality":
        return allocate(weights, L_max, table, quality_fn)
    raise ValueError(f"unknown allocation objective {objective!r}")


def allocate(weights: np.ndarray, L_max: float, table: LatencyModel,
             quality_fn: np.ndarray) -> tuple[np.ndarray, float]:
    """Method step 5, v1. Best feasible allocation and its predicted latency.

    If nothing meets the deadline, returns the cheapest allocation; the governor then slows the drone
    to v* for that latency.
    """
    allocs = table.allocs
    lat = table.predict_all()
    limit = L_max if not math.isnan(L_max) else -math.inf
    i = search(weights, limit, allocs, lat, quality_fn)
    if i is None:
        i = int(np.argmin(lat))
    return np.array(allocs[i], dtype=np.int64), float(lat[i])


class CoverageAllocator:
    """v6: requirements in metres, per direction (design.md, "4-5 (v6)"). No weights, no kappa, no Q_min.

    Every direction theta (5 deg grid) and every detected contact must be covered by some camera whose
    detection range reaches what the drone needs to cancel the closing speed from there:

        d_req(c, r) = min(m + tau(r) c + c^2 / 2a, d_det(high)),   tau(r) = L(r) + extra_blind
        r* = argmin L(r)  s.t.  reach(theta, r) >= d_req(c(theta), r) for all theta and contacts,  r_j >= 1

    Capping at the high tier keeps all-high feasible: where nothing suffices the camera is required at high
    and the governor slows the drone. Ties go to fewer tokens.
    """

    def __init__(self, table: LatencyModel, cam_yaws_deg: Sequence[float], fov_deg: float,
                 det_by_tier: np.ndarray, tier_tokens: Sequence[int], step_deg: float = 5.0):
        self.allocs = table.allocs
        self.lat = table.predict_all()
        self.tokens = np.asarray(tier_tokens)[self.allocs].sum(axis=1)
        self.yaws = np.radians(np.asarray(cam_yaws_deg, dtype=np.float64))
        self.half_fov = math.radians(fov_deg) / 2
        self.det = np.asarray(det_by_tier, dtype=np.float64)
        self.det_alloc = self.det[self.allocs]                                    # (N, V)
        self.floor = (self.allocs >= 1).all(axis=1)
        self.theta = np.radians(np.arange(0.0, 360.0, step_deg))
        self.u = np.stack([np.cos(self.theta), np.sin(self.theta)], axis=1)       # (D, 2)
        self.reach_grid = self.reach(self.theta)                                  # (N, D)

    def cover(self, bearings: np.ndarray) -> np.ndarray:
        off = (np.asarray(bearings)[:, None] - self.yaws[None] + math.pi) % (2 * math.pi) - math.pi
        return np.abs(off) <= self.half_fov + 1e-9                                # (D, V)

    def reach(self, bearings: np.ndarray) -> np.ndarray:
        cov = self.cover(bearings)
        return (self.det_alloc[:, None, :] * cov[None]).max(axis=2)              # (N, D)

    def required(self, closing: np.ndarray, a: float, m: float, extra_blind: float) -> np.ndarray:
        c = np.maximum(np.asarray(closing, dtype=np.float64), 0.0)
        tau = self.lat + extra_blind
        req = m + tau[:, None] * c[None] + c[None] ** 2 / (2 * a)
        return np.minimum(req, self.det[3])                                       # (N, D)

    def feasible(self, v_body: np.ndarray, a: float, m: float, extra_blind: float,
                 contact_bearings: Sequence[float] = (), contact_closing: Sequence[float] = ()) -> np.ndarray:
        v = np.asarray(v_body, dtype=np.float64)[:2]
        ok = self.floor & (self.reach_grid >= self.required(self.u @ v, a, m, extra_blind) - 1e-9).all(axis=1)
        cb = np.asarray(contact_bearings, dtype=np.float64)
        cc = np.asarray(contact_closing, dtype=np.float64)
        sel = cc > 1e-6
        if sel.any():
            ok &= (self.reach(cb[sel]) >= self.required(cc[sel], a, m, extra_blind) - 1e-9).all(axis=1)
        return ok

    def allocate(self, v_body: np.ndarray, a: float, m: float, extra_blind: float,
                 contact_bearings: Sequence[float] = (), contact_closing: Sequence[float] = ()
                 ) -> tuple[np.ndarray, float]:
        ok = self.feasible(v_body, a, m, extra_blind, contact_bearings, contact_closing)
        idx = np.flatnonzero(ok)
        if not len(idx):                     # cannot happen (all-high is always feasible), but never go blind
            idx = np.flatnonzero((self.allocs == 3).all(axis=1))
        best = idx[np.lexsort((self.tokens[idx], self.lat[idx]))[0]]
        return np.array(self.allocs[best], dtype=np.int64), float(self.lat[best])
