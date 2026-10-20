"""Baseline allocations. None of them uses the physics.

B1 all_high       every view at tier 3
B2 uniform_fixed  every view at one tier (VLA-ACL style fixed K per view)
B3 importance     tiers from the last step's per-view importance under a fixed token budget
                  (FastVLA style: budget from where the model looked, not from the deadline)
B4 front_only     front view at tier 3, the rest off (AerialVLA style)

Each is evaluated with and without the governor, so the governor's own effect is separable.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from deadlinevla.budget.allocator import quality_table, search
from deadlinevla.budget.latency_model import LatencyModel, tokens_of
from deadlinevla.config import Config

BASELINES = ("all_high", "uniform_fixed", "importance", "front_only")


def baseline_alloc(name: str, state: dict[str, Any], table: LatencyModel, cfg: Config) -> np.ndarray:
    V = cfg.cameras.n_cams
    if name == "all_high":
        return np.full(V, 3, dtype=np.int64)
    if name == "uniform_fixed":
        return np.full(V, cfg.eval.uniform_tier, dtype=np.int64)
    if name == "front_only":
        a = np.zeros(V, dtype=np.int64)
        a[0] = 3
        return a
    if name == "importance":
        imp: Optional[np.ndarray] = state.get("importance")
        w = np.ones(V) if imp is None else np.asarray(imp, dtype=np.float64)
        w = w / max(w.sum(), 1e-9)
        budget = V * cfg.tier_tokens[cfg.eval.uniform_tier]       # same tokens as B2
        costs = tokens_of(table.allocs, cfg.tier_tokens).astype(np.float64)
        q = quality_table(cfg.budget.quality_fn, cfg.tier_tokens)
        i = search(w, budget, table.allocs, costs, q, floor=False)
        return np.array(table.allocs[i if i is not None else 0], dtype=np.int64)
    raise ValueError(f"unknown baseline {name!r}; known: {BASELINES}")
