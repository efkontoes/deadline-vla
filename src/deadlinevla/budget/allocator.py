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


def allocate_for_speed(weights: np.ndarray, table: LatencyModel, quality_fn: np.ndarray,
                       mid_weight: float = 0.25) -> tuple[np.ndarray, float]:
    """Method step 5, v2: speed is the objective, perception the constraint.

        r* = argmax_r v*(L(r), D)   s.t.   sum_j w_j q(r_j) >= Q_min

    v* falls monotonically with L for every D > 0 (and is 0 for all r when D <= 0), so this is the
    cheapest allocation that still sees enough; D drops out of the argmax. Ties go to higher quality.

    The 360 degree floor (every view at least low) is kept as a hard constraint: with a weighted Q_min
    alone, the optimiser would switch rear views off and pay for it with a sharper front view, and a
    threat from behind would never reach the model.
    """
    allocs = table.allocs
    lat = table.predict_all()
    w = np.asarray(weights, dtype=np.float64)
    score = quality_fn[allocs] @ w
    ok = (score >= required_quality(w, quality_fn, mid_weight) - 1e-9) & (allocs >= 1).all(axis=1)
    idx = np.flatnonzero(ok)
    cheapest = idx[lat[idx] <= lat[idx].min() + 1e-12]
    i = int(cheapest[np.argmax(score[cheapest])])
    return np.array(allocs[i], dtype=np.int64), float(lat[i])


def choose_allocation(weights: np.ndarray, L_max: float, table: LatencyModel, quality_fn: np.ndarray,
                      objective: str = "speed", mid_weight: float = 0.25) -> tuple[np.ndarray, float]:
    """Dispatch on ``budget.objective``: ``speed`` (v2) or ``quality`` (v1, the original doc)."""
    if objective == "speed":
        return allocate_for_speed(weights, table, quality_fn, mid_weight)
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
