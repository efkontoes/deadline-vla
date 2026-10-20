"""Figures and the markdown report.

Headline (``speed_vs_compute``): for each method, the fastest mean flight speed it achieves at any
speed cap where collisions per km stay at or below the threshold, against the compute it spends
(mean visual tokens per tick). Up-and-left is better.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# Fixed method -> colour/marker assignment (reference categorical order; never cycled).
# Five methods exceed the three-slot all-pairs cap for scatter plots, so marker shape and a direct
# label carry identity as well as colour.
STYLE = {
    "deadline": ("#2a78d6", "o", "deadline (ours)"),
    "deadline_v1": ("#6250d6", "h", "deadline v1 (quality obj.)"),
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


def _headline(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows the safe-speed plot uses: the static families when there are any, else all families."""
    static = [r for r in results if r.get("family") == "static"]
    return static or _overall(results)


def safe_speed_points(results: Sequence[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Per (method, governor): best mean speed among caps meeting the collision threshold."""
    groups: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for r in _headline(results):
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


def speed_vs_compute(results: Sequence[dict[str, Any]], threshold: float = 1.0, title: str = "",
                     tag: str = "") -> Figure:
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
    scope = "static families" if any(r.get("family") == "static" for r in results) else "all families"
    ax.set_title(title or f"Safe speed vs compute, {scope} (coll/km ≤ {threshold:g}); filled = governor{tag}",
                 color=INK, fontsize=10, loc="left")
    ax.set_ylim(0, y_span * 1.15)
    ax.set_xlim(0, x_span * 1.3)
    fig.tight_layout()
    return fig


def rear_threat_collisions(results: Sequence[dict[str, Any]], tag: str = "") -> Optional[Figure]:
    """Collision rate (Wilson 95% interval) in rear_threat at the highest speed cap, governed, vs compute."""
    rows = [r for r in results if r.get("family") == "rear_threat" and r["governor"]]
    if not rows:
        return None
    cap = max(r["speed_cap"] for r in rows)
    rows = sorted((r for r in rows if r["speed_cap"] == cap), key=lambda r: r["tokens_per_tick"])
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    for i, r in enumerate(rows):
        color, marker, label = STYLE.get(r["method"], ("#6250d6", "o", r["method"]))
        lo, hi = r["collision_ci"]
        p = r["collision_rate"]
        ax.errorbar(i, p, yerr=[[max(p - lo, 0.0)], [max(hi - p, 0.0)]],
                    fmt=marker, color=color, markersize=8, capsize=4, linewidth=1.5, zorder=3)
        ax.annotate(f"{r['tokens_per_tick']:.0f} tok", (i, hi), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(rows)), [STYLE.get(r["method"], (None, None, r["method"]))[2] for r in rows],
                  fontsize=8, rotation=15)
    ax.set_ylabel("collision rate (95% CI)", color=INK, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_title(f"rear_threat (obstacle from behind), governed, cap {cap:g} m/s, n={rows[0]['episodes']}{tag}",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def pareto(results: Sequence[dict[str, Any]], tag: str = "") -> Figure:
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
    ax.set_title(f"Time to goal vs collision rate (one point per speed cap){tag}", color=INK, fontsize=10, loc="left")
    if seen:
        ax.legend(fontsize=7, frameon=False, labelcolor=INK, ncol=2)
    fig.tight_layout()
    return fig


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def _rate_ci(r: dict[str, Any], key: str) -> str:
    ci = r.get(key.replace("_rate", "_ci"))
    if ci is None:
        return _fmt(r.get(key))
    return f"{r[key]:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def _row(r: dict[str, Any], cols: Sequence[str]) -> str:
    cells = [_rate_ci(r, c) if c in ("success_rate", "collision_rate") else _fmt(r.get(c)) for c in cols]
    return "| " + " | ".join(cells) + " |"


def headline_lines(results: Sequence[dict[str, Any]], episodes: Sequence[dict[str, Any]], ours: str = "deadline",
                   reference: str = "uniform_fixed") -> list[str]:
    """Tokens for equal safety and speed: paired comparison of each method against B2 at the top speed cap."""
    from deadlinevla.eval.stats import compare

    gov = [e for e in episodes if e["governor"]]
    if not gov or not any(e["method"] == reference for e in gov):
        return []
    cap = max(e["speed_cap"] for e in gov)
    fams = sorted({e["family"] for e in gov})
    static = [f for f in fams if f != "rear_threat"]
    tok = {(r["method"], r["governor"]): r["tokens_per_tick"] for r in results
           if r.get("family") == "all" and r["speed_cap"] == cap}

    def cmp_rows(families):
        rows = []
        for m in sorted({e["method"] for e in gov}):
            c = compare(gov, m, reference, True, families, cap)
            if c.get("n"):
                rows.append((m, c))
        return rows

    lines = ["## Headline: tokens for equal safety and speed", "",
             f"Governed, speed cap {cap:g} m/s, each method paired scene-by-scene against B2 `{reference}`. Speed "
             "difference: mean per-scene difference with a 95% paired-bootstrap interval. Event rates: McNemar "
             "exact p on the discordant pairs.", ""]
    for title, families in (("Static families (" + ", ".join(static) + ")", static), ("rear_threat", ["rear_threat"])):
        rows = cmp_rows(families) if families and all(f in fams for f in families) else []
        if not rows:
            continue
        lines += [f"**{title}**", "", "| method | n | tokens/tick (vs B2) | speed m/s | Δspeed vs B2 [95% CI] | "
                  "success | collision | p (collision) |", "|---|---|---|---|---|---|---|---|"]
        for m, c in rows:
            t, t0 = tok.get((m, True)), tok.get((reference, True))
            tok_s = "-" if t is None else (f"{t:.0f} ({(t / t0 - 1) * 100:+.0f}%)" if t0 else f"{t:.0f}")
            lo, hi = c["speed_diff_ci"]
            dv = f"{c['speed_diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            lines.append(f"| {m} | {c['n']} | {tok_s} | {c['speed_a']:.2f} | {dv} | {c['success']['a']:.3f} | "
                         f"{c['collision']['a']:.3f} | {c['collision']['p_mcnemar']:.3g} |")
        lines.append("")
        if title.startswith("Static") and any(m == ours for m, _ in rows):
            c = dict(rows)[ours]
            lo, hi = c["speed_diff_ci"]
            same = lo <= 0 <= hi
            t, t0 = tok.get((ours, True)), tok.get((reference, True))
            if t and t0:
                sa, sr = c["success"]["a"], c["success"]["ref"]
                ca, cr = c["collision"]["a"], c["collision"]["ref"]
                lines += [f"> `{ours}` vs B2 on static scenes: {(1 - t / t0) * 100:.0f}% fewer tokens per tick, "
                          f"success {sa:.3f} vs {sr:.3f}, collisions {ca:.3f} vs {cr:.3f}, "
                          f"speed {c['speed_diff']:+.3f} m/s [{lo:+.3f}, {hi:+.3f}]"
                          f"{' (interval includes zero: equal speed, not faster)' if same else ''}.", ""]
    return lines


def write_report(results: Sequence[dict[str, Any]], out_dir: str | Path, threshold: float,
                 notes: Sequence[str] = (), synthetic: bool = False,
                 summary: Optional[dict[str, dict[str, Any]]] = None,
                 episodes: Sequence[dict[str, Any]] = ()) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "\nSYNTHETIC latency (not a measured board)" if synthetic else ""
    figs = [("speed_vs_compute", speed_vs_compute(results, threshold, tag=tag)), ("pareto", pareto(results, tag=tag)),
            ("rear_threat", rear_threat_collisions(results, tag=tag))]
    for name, fig in figs:
        if fig is None:
            continue
        fig.savefig(out / f"{name}.png")
        fig.savefig(out / f"{name}.pdf")
        plt.close(fig)
    cols = ["method", "governor", "speed_cap", "episodes", "success_rate", "collision_rate", "collisions_per_km",
            "time_to_goal", "mean_speed", "tokens_per_tick", "violation_rate", "mean_latency", "mean_age"]
    lines = ["# DeadlineVLA evaluation report", ""]
    lines += [f"> {n}" for n in notes] + ([""] if notes else [])
    lines += headline_lines(results, episodes)
    lines += ["Rates show the Wilson 95% interval in brackets.", "",
              "![speed vs compute](speed_vs_compute.png)", "",
              *(["![rear threat](rear_threat.png)", ""] if (out / "rear_threat.png").exists() else []),
              "![pareto](pareto.png)", "",
              "## Safe speed per method", "", "| method | governor | safe speed (m/s) | at cap | tokens/tick |",
              "|---|---|---|---|---|"]
    for p in safe_speed_points(results, threshold):
        lines.append(f"| {p['method']} | {p['governor']} | {_fmt(p['safe_speed'])} | {_fmt(p['speed_cap'])} | "
                     f"{_fmt(p['tokens_per_tick'])} |")
    if summary:
        from deadlinevla.eval.metrics import CAUSES

        lines += ["", "## Deadline violations and their causes (all caps pooled)", "",
                  "Rates are per moving tick: `doc` uses clearance along the velocity and the latency of the "
                  "flown inference; `cmd` uses clearance along the command (what the governor judged); `age` "
                  "uses the real information age; `rel` is the v3 relative deadline (any contact closing too fast "
                  "for its clearance, over all ticks). Causes are fractions of the `doc` violations.", "",
                  "| method | governor | doc | cmd | age | rel | " + " | ".join(CAUSES) + " |",
                  "|---|---|---|---|---|---|" + "---|" * len(CAUSES)]
        for key in sorted(summary):
            m = summary[key]
            method, gov = key.split("|")
            causes = m.get("violation_causes", {})
            lines.append(f"| {method} | {gov == 'gov'} | {_fmt(m.get('violation_rate'))} | "
                         f"{_fmt(m.get('violation_rate_cmd'))} | {_fmt(m.get('violation_rate_age'))} | "
                         f"{_fmt(m.get('violation_rate_rel'))} | "
                         + " | ".join(_fmt(causes.get(c, 0.0)) for c in CAUSES) + " |")
    lines += ["", "## All cells (all families)", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(_overall(results), key=lambda r: (r["method"], not r["governor"], r["speed_cap"])):
        lines.append(_row(r, cols))
    fams = sorted({r["family"] for r in results if r.get("family", "all") != "all"})
    for fam in fams:
        lines += ["", f"## Family: {fam}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        rows = [r for r in results if r.get("family") == fam]
        for r in sorted(rows, key=lambda r: (r["method"], not r["governor"], r["speed_cap"])):
            lines.append(_row(r, cols))
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
