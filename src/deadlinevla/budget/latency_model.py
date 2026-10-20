"""Method step 3: latency model of the onboard computer.

    L(r) = L_0 + sum_j e(r_j) + c sum_j n(r_j) + K L_head

Measured once per board over all 4^6 = 4096 allocations. ``fit_linear`` fits the form above and
reports its worst residual; if that is too large, ``LatencyModel.predict`` falls back to the
measured table.

Also defines the latency *sources* a runtime can sample from in simulation: ``SyntheticBoard`` (a
stand-in for a Jetson, used before a real board is available) and ``TableBoard`` (replays a table
measured on the real board, so simulated rollouts see the board's latency, not the desktop's).
"""

from __future__ import annotations

import functools
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

N_TIERS = 4  # 0 = off, 1 = low, 2 = mid, 3 = high
STAGES = ("fixed", "encode", "prefill", "head")


@functools.lru_cache(maxsize=8)
def all_allocs(n_views: int = 6) -> np.ndarray:
    """Every allocation, shape (4^V, V), int8. Row index is the base-4 number r_0 r_1 ... r_{V-1}."""
    arr = np.array(list(itertools.product(range(N_TIERS), repeat=n_views)), dtype=np.int8)
    arr.setflags(write=False)
    return arr


def alloc_index(alloc: Sequence[int]) -> int:
    idx = 0
    for r in alloc:
        idx = idx * N_TIERS + int(r)
    return idx


def tokens_of(allocs: np.ndarray, tier_tokens: Sequence[int]) -> np.ndarray:
    """Total visual tokens per allocation (works for one alloc or a batch)."""
    return np.asarray(tier_tokens)[np.asarray(allocs, dtype=np.int64)].sum(axis=-1)


def tier_counts(allocs: np.ndarray) -> np.ndarray:
    """(N, 3): how many views sit at tier 1, 2, 3."""
    a = np.asarray(allocs)
    return np.stack([(a == r).sum(axis=-1) for r in (1, 2, 3)], axis=-1).astype(np.float64)


# --------------------------------------------------------------------------------------------------
# Measured table


@dataclass
class LatencyTable:
    allocs: np.ndarray                  # (N, V) int8
    p50: np.ndarray                     # (N,) seconds
    p95: np.ndarray
    pmax: np.ndarray
    stages_p50: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def stat(self, name: str) -> np.ndarray:
        return {"p50": self.p50, "p95": self.p95, "max": self.pmax}[name]

    @property
    def n_views(self) -> int:
        return int(self.allocs.shape[1])

    def lookup(self, alloc: Sequence[int], stat: str = "p95") -> float:
        if len(self.allocs) == N_TIERS ** self.n_views:
            return float(self.stat(stat)[alloc_index(alloc)])
        hit = np.flatnonzero((self.allocs == np.asarray(alloc)).all(axis=1))
        if not len(hit):
            raise KeyError(f"allocation {list(alloc)} not in latency table")
        return float(self.stat(stat)[hit[0]])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stages = {f"stage_{k}": v for k, v in self.stages_p50.items()}
        np.savez(path, allocs=self.allocs, p50=self.p50, p95=self.p95, pmax=self.pmax,
                 meta=np.array(json.dumps(self.meta)), **stages)
        return path

    @classmethod
    def load(cls, path: str | Path) -> LatencyTable:
        z = np.load(path, allow_pickle=False)
        stages = {k[len("stage_"):]: z[k] for k in z.files if k.startswith("stage_")}
        return cls(allocs=z["allocs"], p50=z["p50"], p95=z["p95"], pmax=z["pmax"],
                   stages_p50=stages, meta=json.loads(str(z["meta"])))


def profile_allocations(runtime: Any, allocs: np.ndarray, iters: int, obs: Any = None,
                        warmup: int = 1, meta: Optional[dict] = None) -> LatencyTable:
    """Run the real runtime for each allocation and record p50, p95 and max end-to-end latency.

    ``runtime`` must offer ``infer(obs, alloc)`` and ``timings() -> {"total": s, <stage>: s, ...}``.
    On the target board this is ``TrtRuntime``; in simulation it is a ``SimulatedRuntime``.
    """
    allocs = np.asarray(allocs, dtype=np.int8)
    n = len(allocs)
    p50, p95, pmax = np.zeros(n), np.zeros(n), np.zeros(n)
    stage_acc: dict[str, np.ndarray] = {}
    for i, alloc in enumerate(allocs):
        for _ in range(warmup):
            runtime.infer(obs, alloc)
        totals = np.empty(iters)
        stage_samples: dict[str, list[float]] = {}
        for k in range(iters):
            runtime.infer(obs, alloc)
            t = runtime.timings()
            totals[k] = t["total"]
            for s in STAGES:
                if s in t:
                    stage_samples.setdefault(s, []).append(t[s])
        p50[i], p95[i], pmax[i] = np.percentile(totals, 50), np.percentile(totals, 95), totals.max()
        for s, vals in stage_samples.items():
            stage_acc.setdefault(s, np.zeros(n))[i] = float(np.median(vals))
    return LatencyTable(allocs=allocs, p50=p50, p95=p95, pmax=pmax, stages_p50=stage_acc,
                        meta={"iters": iters, **(meta or {})})


# --------------------------------------------------------------------------------------------------
# Linear model


@dataclass
class LinearModel:
    """Total latency = intercept + sum_r count_r * per_view[r].

    ``per_view[r]`` folds e(r) + c n(r) together; the two are only separable from per-stage timings,
    in which case ``e``, ``c``, ``L0`` and ``head`` are filled in as well.
    """

    intercept: float
    per_view: np.ndarray                   # (3,) for tiers 1..3
    worst_residual: float
    rms_residual: float
    stat: str
    e: Optional[np.ndarray] = None         # (3,)
    c: Optional[float] = None
    L0: Optional[float] = None
    head: Optional[float] = None           # K * L_head

    def predict_counts(self, counts: np.ndarray) -> np.ndarray:
        return self.intercept + counts @ self.per_view

    def to_dict(self) -> dict:
        d = {"intercept": self.intercept, "per_view": self.per_view.tolist(),
             "worst_residual": self.worst_residual, "rms_residual": self.rms_residual, "stat": self.stat}
        if self.e is not None:
            d.update(e=self.e.tolist(), c=self.c, L0=self.L0, head=self.head)
        return d


def fit_linear(table: LatencyTable, stat: str = "p95",
               tier_tokens: Optional[Sequence[int]] = None) -> LinearModel:
    """Least-squares fit of Method step 3; reports the worst residual over all allocations."""
    counts = tier_counts(table.allocs)
    y = table.stat(stat)
    X = np.column_stack([np.ones(len(y)), counts])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = X @ coef - y
    model = LinearModel(intercept=float(coef[0]), per_view=coef[1:].copy(),
                        worst_residual=float(np.abs(res).max()), rms_residual=float(np.sqrt((res ** 2).mean())),
                        stat=stat)
    st = table.stages_p50
    if tier_tokens is not None and {"encode", "prefill"} <= set(st):
        e, *_ = np.linalg.lstsq(counts, st["encode"], rcond=None)
        toks = tokens_of(table.allocs, tier_tokens).astype(np.float64)
        Xc = np.column_stack([np.ones(len(toks)), toks])
        (pre0, c), *_ = np.linalg.lstsq(Xc, st["prefill"], rcond=None)
        head = float(np.median(st["head"])) if "head" in st else None
        fixed = float(np.median(st["fixed"])) if "fixed" in st else 0.0
        model.e, model.c, model.L0, model.head = e, float(c), fixed + float(pre0), head
    return model


class LatencyModel:
    """Predicts the deadline latency of an allocation: linear form if it fits, else the table."""

    def __init__(self, table: LatencyTable, stat: str = "p95", max_residual: float = 0.005,
                 tier_tokens: Optional[Sequence[int]] = None):
        self.table = table
        self.stat = stat
        self.linear = fit_linear(table, stat, tier_tokens)
        self.use_table = self.linear.worst_residual > max_residual
        self.allocs = table.allocs
        if self.use_table:
            self._all = table.stat(stat).astype(np.float64)
        else:
            self._all = self.linear.predict_counts(tier_counts(self.allocs))
        self._all.setflags(write=False)

    def predict(self, alloc: Sequence[int]) -> float:
        """p95 (or the configured stat) latency of one allocation, seconds."""
        if self.use_table:
            return self.table.lookup(alloc, self.stat)
        return float(self.linear.predict_counts(tier_counts(np.asarray(alloc)[None]))[0])

    def predict_all(self) -> np.ndarray:
        return self._all

    def describe(self) -> dict:
        return {"source": "table" if self.use_table else "linear", **self.linear.to_dict(),
                "board": self.table.meta.get("board"), "synthetic": self.table.meta.get("synthetic", False)}


# --------------------------------------------------------------------------------------------------
# Latency sources for simulation


class LatencySource:
    """Samples an end-to-end latency (and per-stage split) for an allocation."""

    name = "abstract"
    synthetic = True

    def sample(self, alloc: Sequence[int], rng: np.random.Generator) -> dict[str, float]:
        raise NotImplementedError


# Rough Jetson-class numbers for a ~450M-parameter VLA; placeholders until M3 measures the board.
_BOARDS: dict[str, dict[str, Any]] = {
    "synthetic_orin_nx": dict(L0=0.015, e=(0.0, 0.006, 0.012, 0.030), c=1.2e-4, quad=2.0e-8,
                              K=10, head=0.003, sigma=0.06, spike_p=0.02, spike=1.5),
    "synthetic_agx_orin": dict(L0=0.010, e=(0.0, 0.0035, 0.007, 0.017), c=6.5e-5, quad=1.1e-8,
                               K=10, head=0.0018, sigma=0.05, spike_p=0.02, spike=1.4),
    "synthetic_desktop": dict(L0=0.004, e=(0.0, 0.001, 0.002, 0.004), c=1.5e-5, quad=2.0e-9,
                              K=10, head=0.0005, sigma=0.04, spike_p=0.01, spike=1.3),
}


class SyntheticBoard(LatencySource):
    """A made-up board with a mildly non-linear (quadratic attention) prefill and lognormal jitter."""

    def __init__(self, board: str, tier_tokens: Sequence[int]):
        if board not in _BOARDS:
            raise KeyError(f"unknown synthetic board {board!r}; known: {sorted(_BOARDS)}")
        self.name = board
        self.p = _BOARDS[board]
        self.tier_tokens = np.asarray(tier_tokens)

    def nominal(self, alloc: Sequence[int]) -> dict[str, float]:
        p = self.p
        a = np.asarray(alloc, dtype=np.int64)
        T = float(self.tier_tokens[a].sum())
        enc = float(np.asarray(p["e"])[a].sum())
        pre = p["c"] * T + p["quad"] * T * T
        head = p["K"] * p["head"]
        return {"fixed": p["L0"], "encode": enc, "prefill": pre, "head": head}

    def sample(self, alloc, rng):
        st = self.nominal(alloc)
        factor = float(np.exp(rng.normal(0.0, self.p["sigma"])))
        if rng.random() < self.p["spike_p"]:
            factor *= self.p["spike"]
        out = {k: v * factor for k, v in st.items()}
        out["total"] = sum(out.values())
        return out


class TableBoard(LatencySource):
    """Replays a measured table: lognormal with the table's p50 and p95, clipped at its max."""

    synthetic = False

    def __init__(self, table: LatencyTable):
        self.table = table
        self.name = table.meta.get("board", "measured")
        self.synthetic = bool(table.meta.get("synthetic", False))

    def sample(self, alloc, rng):
        p50 = self.table.lookup(alloc, "p50")
        p95 = self.table.lookup(alloc, "p95")
        pmax = self.table.lookup(alloc, "max")
        sigma = max(math.log(max(p95, 1e-9) / max(p50, 1e-9)) / 1.645, 1e-6)
        total = min(p50 * math.exp(rng.normal(0.0, sigma)), pmax)
        return {"total": total}


def make_latency_source(board: str, tier_tokens: Sequence[int],
                        table: Optional[LatencyTable] = None) -> LatencySource:
    if board in _BOARDS:
        return SyntheticBoard(board, tier_tokens)
    if table is None:
        raise ValueError(f"board {board!r} is not synthetic; a measured latency table is required")
    return TableBoard(table)


def synthetic_prior(board: str, tier_tokens: Sequence[int], n_views: int = 6) -> LatencyModel:
    """Deterministic (no-jitter) table from a synthetic board's nominal latencies.

    Used by the training sampler before the real board has been profiled.
    """
    name = board if board in _BOARDS else "synthetic_orin_nx"
    src = SyntheticBoard(name, tier_tokens)
    allocs = all_allocs(n_views)
    nominal = [src.nominal(a) for a in allocs]
    tot = np.array([sum(s.values()) for s in nominal])
    table = LatencyTable(allocs=allocs, p50=tot, p95=tot * 1.1, pmax=tot * 1.3,
                         meta={"board": name, "synthetic": True, "prior": True})
    return LatencyModel(table, "p95", max_residual=math.inf)


