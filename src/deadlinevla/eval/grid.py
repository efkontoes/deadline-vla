"""Threat grid: threat size x threat speed with the intercepting generator, governed, camera-coupled.

Each cell is rear_threat with a hunter of that size and speed; every variant flies the same scenes.
Variants:
    derived (matched)   derived floor whose threat envelope (o_max, size) is the cell's threat
    derived (default)   derived floor with the config's default envelope, kept fixed across the grid
    fixed floor         every camera at least low tier (v2 to v4 rule)
    B2 uniform, B1 all-high, B4 front-only

Paired against the fixed floor with exact McNemar on collisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from deadlinevla.config import Config, set_field
from deadlinevla.eval.metrics import summarize, wilson
from deadlinevla.eval.rollout import build_eval_scenes
from deadlinevla.eval.stats import mcnemar_exact
from deadlinevla.eval.sweep import run_sweep

SIZES = (0.3, 0.6, 1.0)          # threat diameter, m
SPEEDS = (3.0, 6.0, 9.0)         # threat speed, m/s
VARIANTS = (                     # label, pilot mode, envelope: "matched" | "default" | None
    ("derived_matched", "deadline", "matched"),
    ("derived_default", "deadline", "default"),
    ("fixed_floor", "deadline_fixed", None),
    ("uniform_fixed", "uniform_fixed", None),
    ("all_high", "all_high", None),
    ("front_only", "front_only", None),
)
REFERENCE = "fixed_floor"


def hard(size: float, speed: float) -> bool:
    """The small-fast corner gets the large paired sample."""
    return (size <= 0.3 and speed >= 6.0) or (size <= 0.6 and speed >= 9.0)


def run_grid(cfg: Config, table_path: str | Path, out_dir: str | Path, cap: float = 8.0,
             n_hard: int = 450, n_easy: int = 150, sizes: Sequence[float] = SIZES,
             speeds: Sequence[float] = SPEEDS, workers: int = 4,
             log: Callable[[str], Any] = print) -> list[dict[str, Any]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = cfg
    for k, v in (("scenes.threat_mode", "intercept"), ("budget.relative", True), ("perception.governor", "camera"),
                 ("eval.governor", (True,)), ("eval.speed_caps", (float(cap),))):
        base = set_field(base, k, v)
    rows: list[dict[str, Any]] = []
    for size in sizes:
        for speed in speeds:
            cell = set_field(set_field(base, "scenes.threat_radius", size / 2), "scenes.threat_speed", speed)
            n = n_hard if hard(size, speed) else n_easy
            scenes = build_eval_scenes(cell, ["rear_threat"], per_family=n)
            logs_by: dict[str, list] = {}
            for label, mode, envelope in VARIANTS:
                vcfg = set_field(cell, "eval.methods", (mode,))
                if envelope == "matched":
                    vcfg = set_field(set_field(vcfg, "budget.o_max", speed), "budget.floor_threat_size", size)
                (_, logs), = run_sweep(vcfg, scenes, table_path, None, workers)
                logs_by[label] = logs
            ref = {(lg.family, lg.seed): lg.collision for lg in logs_by[REFERENCE]}
            for label, logs in logs_by.items():
                m = summarize(logs)
                mine = {(lg.family, lg.seed): lg.collision for lg in logs}
                b = sum(mine[k] and not ref[k] for k in mine)
                c = sum(ref[k] and not mine[k] for k in mine)
                k_coll = int(sum(lg.collision for lg in logs))
                row = {"size": size, "speed": speed, "variant": label, "n": len(logs),
                       "collision_rate": m.collision_rate, "collision_ci": list(wilson(k_coll, len(logs))),
                       "success_rate": m.success_rate, "tokens_per_tick": m.tokens_per_tick,
                       "mean_speed": m.mean_speed, "unseen_threat_rate": m.unseen_threat_rate,
                       "violation_rate_rel": m.violation_rate_rel,
                       "vs_fixed_only_this": int(b), "vs_fixed_only_fixed": int(c),
                       "p_vs_fixed": mcnemar_exact(b, c)}
                rows.append(row)
                log(f"  s={size:.1f} m  v={speed:.0f} m/s  n={len(logs):3d}  {label:16s} "
                    f"coll={m.collision_rate:.3f} [{row['collision_ci'][0]:.3f},{row['collision_ci'][1]:.3f}] "
                    f"tok={m.tokens_per_tick:5.0f} unseen={m.unseen_threat_rate:.2f} "
                    f"p_vs_fixed={row['p_vs_fixed']:.3g}")
    with open(out / "grid.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


LABELS = {"derived_matched": ("#2a78d6", "o", "derived floor (matched envelope)"),
          "derived_default": ("#6250d6", "h", "derived floor (default 5 m/s, 0.5 m)"),
          "fixed_floor": ("#008300", "p", "fixed floor (tier >= 1)"),
          "uniform_fixed": ("#1baf7a", "^", "B2 uniform"),
          "all_high": ("#eb6834", "s", "B1 all high"),
          "front_only": ("#e87ba4", "v", "B4 front only")}


def plot_grid(rows: Sequence[dict[str, Any]], out_dir: str | Path, synthetic: bool = True) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from deadlinevla.eval.report import INK, INK_2, SURFACE, _style_axes

    sizes = sorted({r["size"] for r in rows})
    fig, axes = plt.subplots(2, len(sizes), figsize=(4.0 * len(sizes), 6.6), dpi=150, sharey="row", squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for col, size in enumerate(sizes):
        for row_i, key in enumerate(("collision_rate", "tokens_per_tick")):
            ax = axes[row_i][col]
            _style_axes(ax)
            for label, (color, marker, name) in LABELS.items():
                pts = sorted((r for r in rows if r["size"] == size and r["variant"] == label), key=lambda r: r["speed"])
                if not pts:
                    continue
                x = np.array([r["speed"] for r in pts]) + (list(LABELS).index(label) - 2.5) * 0.08
                y = np.array([r[key] for r in pts])
                if key == "collision_rate":
                    lo = np.array([r["collision_ci"][0] for r in pts])
                    hi = np.array([r["collision_ci"][1] for r in pts])
                    ax.errorbar(x, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], fmt=marker + "-",
                                color=color, markersize=6, capsize=3, linewidth=1.5, label=name)
                else:
                    ax.plot(x, y, marker + "-", color=color, markersize=6, linewidth=1.5, label=name)
            if row_i == 0:
                ax.set_title(f"threat size {size:g} m", color=INK, fontsize=10, loc="left")
            if row_i == 1:
                ax.set_xlabel("threat speed (m/s)", color=INK, fontsize=9)
            if col == 0:
                ax.set_ylabel("collision rate (95% CI)" if row_i == 0 else "visual tokens per tick", color=INK,
                              fontsize=9)
            ax.set_ylim(bottom=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    title = "Intercepting threat from behind, governed, camera-coupled perception"
    if synthetic:
        title += "\nSYNTHETIC latency (not a measured board)"
    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    fig.text(0.99, 0.01, "error bars: Wilson 95%", color=INK_2, fontsize=7, ha="right")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    path = Path(out_dir) / "threat_grid.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path
