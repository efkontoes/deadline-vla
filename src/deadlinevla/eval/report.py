"""Figures and the markdown report.

Headline (``speed_vs_compute``): for each method, the fastest mean flight speed it achieves at any
speed cap where collisions per km stay at or below the threshold, against the compute it spends
(mean visual tokens per tick). Up-and-left is better.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# Fixed method -> colour/marker assignment (reference categorical order; never cycled).
# Five methods exceed the three-slot all-pairs cap for scatter plots, so marker shape and a direct
# label carry identity as well as colour.
STYLE = {
    "deadline": ("#2a78d6", "o", "deadline (ours)"),
    "all_high": ("#eb6834", "s", "B1 all high"),
    "uniform_fixed": ("#1baf7a", "^", "B2 uniform"),
    "importance": ("#eda100", "D", "B3 importance"),
    "front_only": ("#e87ba4", "v", "B4 front only"),
}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _overall(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in results if r.get("family", "all") == "all"]


def safe_speed_points(results: Sequence[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Per (method, governor): best mean speed among caps meeting the collision threshold."""
    groups: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for r in _overall(results):
        groups.setdefault((r["method"], r["governor"]), []).append(r)
    pts = []
    for (method, gov), rows in sorted(groups.items()):
        ok = [r for r in rows if (r["collisions_per_km"] or 0.0) <= threshold]
        best = max(ok, key=lambda r: r["mean_speed"]) if ok else None
        pts.append({"method": method, "governor": gov,
                    "safe_speed": best["mean_speed"] if best else 0.0,
                    "speed_cap": best["speed_cap"] if best else None,
                    "tokens_per_tick": (best or max(rows, key=lambda r: r["speed_cap"]))["tokens_per_tick"]})
    return pts


def speed_vs_compute(results: Sequence[dict[str, Any]], threshold: float = 1.0, title: str = "") -> Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    pts = safe_speed_points(results, threshold)
    x_span = max([p["tokens_per_tick"] for p in pts] + [1.0])
    y_span = max([p["safe_speed"] for p in pts] + [1.0])
    # one label per group of coincident points ("B2 uniform · B3 importance")
    groups: list[tuple[float, float, list[str]]] = []
    for p in pts:
        color, marker, label = STYLE.get(p["method"], ("#6250d6", "o", p["method"]))
        face = color if p["governor"] else SURFACE
        x, y = p["tokens_per_tick"], p["safe_speed"]
        ax.scatter(x, y, s=70, marker=marker, facecolor=face, edgecolor=color, linewidth=2, zorder=3)
        text = f"{label}{'' if p['governor'] else ' (no gov.)'}"
        for gx, gy, names in groups:
            if abs(x - gx) < 0.03 * x_span and abs(y - gy) < 0.03 * y_span:
                names.append(text)
                break
        else:
            groups.append((x, y, [text]))
    placed: list[tuple[float, float]] = []
    for x, y, names in groups:
        # labels that would run into each other sideways are stacked upward
        n_near = sum(abs(x - px) < 0.3 * x_span and abs(y - py) < 0.04 * y_span for px, py in placed)
        placed.append((x, y))
        ax.annotate(" · ".join(names), (x, y), xytext=(9, 3 + 11 * n_near), textcoords="offset points",
                    fontsize=8, color=INK)
    ax.set_xlabel("compute: mean visual tokens per tick", color=INK, fontsize=10)
    ax.set_ylabel("safe flight speed (m/s)", color=INK, fontsize=10)
    ax.set_title(title or f"Safe speed vs compute (collisions/km ≤ {threshold:g}); filled = with governor",
                 color=INK, fontsize=10, loc="left")
    ax.set_ylim(0, y_span * 1.15)
    ax.set_xlim(0, x_span * 1.3)
    fig.tight_layout()
    return fig


def pareto(results: Sequence[dict[str, Any]]) -> Figure:
    """Time to goal against collisions per km, one point per (method, governor, speed cap)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    seen = set()
    for r in _overall(results):
        if r["time_to_goal"] is None or r["collisions_per_km"] is None:
            continue
        color, marker, label = STYLE.get(r["method"], ("#6250d6", "o", r["method"]))
        key = (r["method"], r["governor"])
        lab = f"{label}{'' if r['governor'] else ' (no gov.)'}" if key not in seen else None
        seen.add(key)
        ax.scatter(r["collisions_per_km"], r["time_to_goal"], s=50, marker=marker,
                   facecolor=color if r["governor"] else SURFACE, edgecolor=color, linewidth=2, label=lab, zorder=3)
    ax.set_xlabel("collisions per km", color=INK, fontsize=10)
    ax.set_ylabel("time to goal (s), successful episodes", color=INK, fontsize=10)
    ax.set_title("Time to goal vs collision rate (each point = one speed cap)", color=INK, fontsize=10, loc="left")
    if seen:
        ax.legend(fontsize=7, frameon=False, labelcolor=INK, ncol=2)
    fig.tight_layout()
    return fig


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def write_report(results: Sequence[dict[str, Any]], out_dir: str | Path, threshold: float,
                 notes: Sequence[str] = ()) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, fig in (("speed_vs_compute", speed_vs_compute(results, threshold)), ("pareto", pareto(results))):
        fig.savefig(out / f"{name}.png")
        fig.savefig(out / f"{name}.pdf")
        plt.close(fig)
    cols = ["method", "governor", "speed_cap", "success_rate", "collisions_per_km", "time_to_goal", "mean_speed",
            "tokens_per_tick", "violation_rate", "mean_latency", "stall_rate"]
    lines = ["# DeadlineVLA evaluation report", ""]
    lines += [f"> {n}" for n in notes] + ([""] if notes else [])
    lines += ["![speed vs compute](speed_vs_compute.png)", "", "![pareto](pareto.png)", "",
              "## Safe speed per method", "", "| method | governor | safe speed (m/s) | at cap | tokens/tick |",
              "|---|---|---|---|---|"]
    for p in safe_speed_points(results, threshold):
        lines.append(f"| {p['method']} | {p['governor']} | {_fmt(p['safe_speed'])} | {_fmt(p['speed_cap'])} | "
                     f"{_fmt(p['tokens_per_tick'])} |")
    lines += ["", "## All cells (all families)", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(_overall(results), key=lambda r: (r["method"], not r["governor"], r["speed_cap"])):
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    fams = sorted({r["family"] for r in results if r.get("family", "all") != "all"})
    for fam in fams:
        lines += ["", f"## Family: {fam}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        rows = [r for r in results if r.get("family") == fam]
        for r in sorted(rows, key=lambda r: (r["method"], not r["governor"], r["speed_cap"])):
            lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
