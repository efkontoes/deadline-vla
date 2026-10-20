"""Paired statistics over episodes. Every method flies the same scenes, so compare scene by scene.

paired_bootstrap  CI of the mean per-scene difference (e.g. speed of method A minus B on the same scene)
mcnemar_exact     two-sided exact McNemar test on paired binary outcomes (e.g. collided or not)
compare           both, for method A against a reference method, from episode summaries
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence

import numpy as np


def paired_bootstrap(diffs: Sequence[float], n_boot: int = 10000, seed: int = 0,
                     level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap of the mean paired difference."""
    x = np.asarray(diffs, dtype=np.float64)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    alpha = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p-value; b = only A had the event, c = only B had it."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare_runs(eps_a: Iterable[dict[str, Any]], eps_b: Iterable[dict[str, Any]], method: str,
                 governor: bool = True, families: Optional[Sequence[str]] = None,
                 cap: Optional[float] = None) -> dict[str, Any]:
    """Paired comparison of the same method across two runs (an ablation) on identical scenes."""
    tag = lambda eps, name: [{**e, "method": name} for e in eps if e["method"] == method]  # noqa: E731
    return compare(tag(eps_a, "A") + tag(eps_b, "B"), "A", "B", governor, families, cap)


def episode_speed(e: dict[str, Any]) -> float:
    return e["distance"] / e["time"] if e["time"] > 0 else 0.0


def _key(e: dict[str, Any]) -> tuple:
    return (e["family"], e["seed"], e["speed_cap"])


def compare(episodes: Iterable[dict[str, Any]], method: str, reference: str, governor: bool = True,
            families: Optional[Sequence[str]] = None, cap: Optional[float] = None,
            seed: int = 0) -> dict[str, Any]:
    """Paired comparison of ``method`` against ``reference`` on identical (family, seed, cap) scenes."""
    def pick(m):
        return {_key(e): e for e in episodes
                if e["method"] == m and e["governor"] == governor
                and (families is None or e["family"] in families) and (cap is None or e["speed_cap"] == cap)}
    a, r = pick(method), pick(reference)
    keys = sorted(set(a) & set(r))
    if not keys:
        return {"n": 0}
    dv = [episode_speed(a[k]) - episode_speed(r[k]) for k in keys]
    mean, lo, hi = paired_bootstrap(dv, seed=seed)
    out = {"n": len(keys), "speed_diff": mean, "speed_diff_ci": [lo, hi],
           "speed_a": float(np.mean([episode_speed(a[k]) for k in keys])),
           "speed_ref": float(np.mean([episode_speed(r[k]) for k in keys]))}
    for event in ("collision", "success"):
        b = sum(a[k][event] and not r[k][event] for k in keys)
        c = sum(r[k][event] and not a[k][event] for k in keys)
        out[event] = {"a": float(np.mean([a[k][event] for k in keys])),
                      "ref": float(np.mean([r[k][event] for k in keys])),
                      "only_a": int(b), "only_ref": int(c), "p_mcnemar": mcnemar_exact(b, c)}
    return out
