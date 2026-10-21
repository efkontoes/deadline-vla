# ruff: noqa: E501  (report-template strings)
"""Score the v6 pre-registered predictions (docs/preregistration.md, amendment 1) and draw the figure.

    python scripts/analyze_v6.py            # reads runs/light_coverage, runs/light_highspeed, runs/light_highspeed_s03

Writes docs/results_v6.md, docs/results_v6.json and docs/media/highspeed.png. Tests exactly as written:
non-inferiority by paired bootstrap bounds (P1, P3), exact McNemar (P2b), Holm across P1, P2b, P3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deadlinevla.eval.stats import mcnemar_exact  # noqa: E402

STATIC = ("goal_nav", "sparse_nav", "slalom")
OURS, V4 = "deadline", "deadline_v4"
B = 10000


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]


def paired(eps, a, b, fams, cap):
    key = lambda e: (e["family"], e["seed"])  # noqa: E731
    A = {key(e): e for e in eps if e["method"] == a and e["family"] in fams and e["speed_cap"] == cap and e["governor"]}
    R = {key(e): e for e in eps if e["method"] == b and e["family"] in fams and e["speed_cap"] == cap and e["governor"]}
    ks = sorted(set(A) & set(R))
    return [A[k] for k in ks], [R[k] for k in ks]


def boot(diff, margin=None, seed=0):
    x = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    out = {"mean": float(x.mean()), "ci": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}
    return out, means


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj.tolist()


def tok(rows, method, family, cap):
    r = next(r for r in rows if r["method"] == method and r["family"] == family and r["speed_cap"] == cap
             and r["governor"])
    return r["tokens_per_tick"]


def main() -> int:
    cov, hs, s03 = ROOT / "runs" / "light_coverage", ROOT / "runs" / "light_highspeed", ROOT / "runs" / "light_highspeed_s03"
    ce, cr = jl(cov / "episodes.jsonl"), jl(cov / "results.jsonl")
    he, hr = jl(hs / "episodes.jsonl"), jl(hs / "results.jsonl")
    caps_hs = sorted({e["speed_cap"] for e in he})
    res: dict = {}

    # P1: static families, cap 10: fewer tokens, collisions and success non-inferior
    a, b = paired(ce, OURS, V4, STATIC, 10.0)
    dc, mc = boot([x["collision"] - y["collision"] for x, y in zip(a, b)])
    ds, ms = boot([x["success"] - y["success"] for x, y in zip(a, b)])
    p1 = {"n": len(a), "tokens_v6": tok(cr, OURS, "static", 10.0), "tokens_v4": tok(cr, V4, "static", 10.0),
          "collision_diff": dc, "success_diff": ds,
          "collision_v6": float(np.mean([x["collision"] for x in a])), "collision_v4": float(np.mean([y["collision"] for y in b])),
          "success_v6": float(np.mean([x["success"] for x in a])), "success_v4": float(np.mean([y["success"] for y in b])),
          "p_noninf": float(max(np.mean(mc >= 0.01), np.mean(ms <= -0.02)))}
    p1["pass"] = bool(p1["tokens_v6"] < p1["tokens_v4"] and dc["ci"][1] <= 0.01 and ds["ci"][0] >= -0.02)
    res["P1"] = p1

    # P2a: tokens rise with the cap; P2b: fewer collisions at 15; P2c: more tokens at 15
    t6 = [tok(hr, OURS, "all", c) for c in caps_hs]
    res["P2a"] = {"caps": caps_hs, "tokens_v6": t6, "pass": bool(all(x < y for x, y in zip(t6, t6[1:])))}
    a, b = paired(he, OURS, V4, ("highspeed",), max(caps_hs))
    only_a = sum(x["collision"] and not y["collision"] for x, y in zip(a, b))
    only_b = sum(y["collision"] and not x["collision"] for x, y in zip(a, b))
    cv6, cv4 = float(np.mean([x["collision"] for x in a])), float(np.mean([y["collision"] for y in b]))
    p2b = {"n": len(a), "collision_v6": cv6, "collision_v4": cv4, "only_v6": only_a, "only_v4": only_b,
           "p": mcnemar_exact(only_a, only_b)}
    p2b["pass_raw"] = bool(p2b["p"] < 0.05 and cv6 < cv4)
    res["P2b"] = p2b
    res["P2c"] = {"tokens_v6": tok(hr, OURS, "all", max(caps_hs)), "tokens_v4": tok(hr, V4, "all", max(caps_hs))}
    res["P2c"]["pass"] = bool(res["P2c"]["tokens_v6"] > res["P2c"]["tokens_v4"])

    # P3: rear_threat, cap 10: collisions non-inferior
    a, b = paired(ce, OURS, V4, ("rear_threat",), 10.0)
    d3, m3 = boot([x["collision"] - y["collision"] for x, y in zip(a, b)])
    res["P3"] = {"n": len(a), "collision_v6": float(np.mean([x["collision"] for x in a])),
                 "collision_v4": float(np.mean([y["collision"] for y in b])), "collision_diff": d3,
                 "p_noninf": float(np.mean(m3 >= 0.01)), "pass": bool(d3["ci"][1] <= 0.01)}

    adj = holm([res["P1"]["p_noninf"], res["P2b"]["p"], res["P3"]["p_noninf"]])
    res["holm"] = {"P1": adj[0], "P2b": adj[1], "P3": adj[2]}
    res["P1"]["pass_holm"] = bool(res["P1"]["pass"] and adj[0] < 0.05)
    res["P2b"]["pass_holm"] = bool(res["P2b"]["pass_raw"] and adj[1] < 0.05)
    res["P3"]["pass_holm"] = bool(res["P3"]["pass"] and adj[2] < 0.05)

    # Descriptive: every method per cap in highspeed; S1 sensitivity
    methods = [m for m in (OURS, V4, "uniform_fixed", "all_high", "front_only") if any(r["method"] == m for r in hr)]
    table = {m: {c: next(r for r in hr if r["method"] == m and r["family"] == "all" and r["speed_cap"] == c)
                 for c in caps_hs} for m in methods}
    res["highspeed"] = {m: {str(c): {k: table[m][c][k] for k in ("collision_rate", "collision_ci", "success_rate",
                                                                 "mean_speed", "tokens_per_tick", "time_to_goal")}
                            for c in caps_hs} for m in methods}
    if (s03 / "results.jsonl").exists():
        sr = jl(s03 / "results.jsonl")
        res["S1"] = {str(c): {k: next(r for r in sr if r["family"] == "all" and r["speed_cap"] == c)[k]
                              for k in ("collision_rate", "success_rate", "mean_speed", "tokens_per_tick")}
                     for c in caps_hs}
    static = {m: next(r for r in cr if r["method"] == m and r["family"] == "static" and r["speed_cap"] == 10.0
                      and r["governor"]) for m in methods}
    res["static_cap10"] = {m: {k: static[m][k] for k in ("collision_rate", "success_rate", "mean_speed",
                                                          "tokens_per_tick")} for m in methods}

    (ROOT / "docs" / "results_v6.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    figure(table, caps_hs, res.get("S1"))
    write_md(res, caps_hs, methods)
    print(json.dumps({k: res[k].get("pass", res[k].get("pass_raw")) for k in ("P1", "P2a", "P2b", "P2c", "P3")}))
    return 0


NAMES = {OURS: "v6 coverage (ours)", V4: "v4 allocator", "uniform_fixed": "B2 uniform", "all_high": "B1 all-high",
         "front_only": "B4 front-only"}
COLORS = {OURS: "#2a78d6", V4: "#6250d6", "uniform_fixed": "#1baf7a", "all_high": "#eb6834", "front_only": "#e87ba4"}


def figure(table, caps, s1):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from deadlinevla.eval.report import INK, SURFACE, _style_axes

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, label) in zip(axes, (("tokens_per_tick", "visual tokens per tick"),
                                       ("collision_rate", "collision rate (Wilson 95%)"),
                                       ("mean_speed", "mean speed (m/s)"))):
        _style_axes(ax)
        for m, rows in table.items():
            y = np.array([rows[c][key] for c in caps])
            if key == "collision_rate":
                lo = np.array([rows[c]["collision_ci"][0] for c in caps])
                hi = np.array([rows[c]["collision_ci"][1] for c in caps])
                ax.errorbar(caps, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], color=COLORS[m], marker="o",
                            ms=5, lw=1.8, capsize=3, label=NAMES[m])
            else:
                ax.plot(caps, y, color=COLORS[m], marker="o", ms=5, lw=1.8, label=NAMES[m])
        if s1 and key in ("tokens_per_tick", "collision_rate"):
            ax.plot(caps, [s1[str(c)][key] for c in caps], color=COLORS[OURS], ls="--", marker="o", ms=4, lw=1.2,
                    label="v6, design size 0.3 m (S1)")
        ax.set_xlabel("speed cap (m/s)", color=INK, fontsize=9)
        ax.set_title(label, color=INK, fontsize=10, loc="left")
        ax.set_ylim(bottom=0)
        ax.set_xticks(caps)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("High-speed family, governed, camera-coupled perception: SYNTHETIC latency, surrogate policy",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    out = ROOT / "docs" / "media" / "highspeed.png"
    fig.savefig(out)
    plt.close(fig)


def pct(x):
    return f"{100 * x:.1f}%"


def write_md(res, caps, methods):
    p1, p2a, p2b, p2c, p3, h = res["P1"], res["P2a"], res["P2b"], res["P2c"], res["P3"], res["holm"]
    ok = lambda b: "**held**" if b else "**failed**"  # noqa: E731
    L = ["# v6 results: coverage allocation, scored against the pre-registered predictions", "",
         "Predictions: docs/preregistration.md, amendment 1 (committed in `006411b` before any v6 code or run).",
         "Simulation only: synthetic Jetson latency, surrogate policy, camera-coupled relative governor.", "",
         "| ID | Prediction | Result | Verdict |", "|---|---|---|---|",
         f"| P1 | v6 uses fewer tokens than v4 on the v4 families, collisions and success non-inferior (cap 10, "
         f"n = {p1['n']}) | tokens {p1['tokens_v6']:.0f} vs {p1['tokens_v4']:.0f}; collisions {pct(p1['collision_v6'])} "
         f"vs {pct(p1['collision_v4'])}, paired difference {100 * p1['collision_diff']['mean']:+.1f} pp "
         f"[{100 * p1['collision_diff']['ci'][0]:+.1f}, {100 * p1['collision_diff']['ci'][1]:+.1f}] (margin +1); success "
         f"{pct(p1['success_v6'])} vs {pct(p1['success_v4'])}, difference [{100 * p1['success_diff']['ci'][0]:+.1f}, "
         f"{100 * p1['success_diff']['ci'][1]:+.1f}] pp (margin −2); Holm p = {h['P1']:.3g} | {ok(p1['pass_holm'])} |",
         f"| P2a | v6's tokens rise with the cap (highspeed) | {' → '.join(f'{t:.0f}' for t in p2a['tokens_v6'])} at caps "
         f"{', '.join(f'{c:g}' for c in p2a['caps'])} | {ok(p2a['pass'])} |",
         f"| P2b | v6 collides less than v4 at 15 m/s (highspeed, n = {p2b['n']}) | {pct(p2b['collision_v6'])} vs "
         f"{pct(p2b['collision_v4'])}; discordant {p2b['only_v6']} (v6 only) vs {p2b['only_v4']} (v4 only); McNemar "
         f"p = {p2b['p']:.3g}, Holm {h['P2b']:.3g} | {ok(p2b['pass_holm'])} |",
         f"| P2c | v6 spends more tokens than v4 at 15 m/s | {p2c['tokens_v6']:.0f} vs {p2c['tokens_v4']:.0f} | "
         f"{ok(p2c['pass'])} |",
         f"| P3 | v6 not worse than v4 on rear_threat (cap 10, n = {p3['n']}) | {pct(p3['collision_v6'])} vs "
         f"{pct(p3['collision_v4'])}, difference [{100 * p3['collision_diff']['ci'][0]:+.1f}, "
         f"{100 * p3['collision_diff']['ci'][1]:+.1f}] pp (margin +1); Holm p = {h['P3']:.3g} | {ok(p3['pass_holm'])} |",
         "", "![high-speed results](media/highspeed.png)", "",
         "## High-speed family, every method (governed)", "",
         "| method | cap | tokens/tick | collision rate [95% CI] | success | mean speed | time to goal |",
         "|---|---|---|---|---|---|---|"]
    for m in methods:
        for c in caps:
            r = res["highspeed"][m][str(c)]
            ttg = "-" if r["time_to_goal"] is None else f"{r['time_to_goal']:.1f} s"
            L.append(f"| {NAMES[m]} | {c:g} | {r['tokens_per_tick']:.0f} | {pct(r['collision_rate'])} "
                     f"[{pct(r['collision_ci'][0])}, {pct(r['collision_ci'][1])}] | {pct(r['success_rate'])} | "
                     f"{r['mean_speed']:.2f} m/s | {ttg} |")
    L += ["", "## v4 families, cap 10 m/s, static scenes (governed)", "",
          "| method | tokens/tick | collision rate | success | mean speed |", "|---|---|---|---|---|"]
    for m in methods:
        r = res["static_cap10"][m]
        L.append(f"| {NAMES[m]} | {r['tokens_per_tick']:.0f} | {pct(r['collision_rate'])} | {pct(r['success_rate'])} | "
                 f"{r['mean_speed']:.2f} m/s |")
    if "S1" in res:
        L += ["", "## S1 sensitivity: design obstacle size 0.3 m (v6 only, highspeed)", "",
              "| cap | tokens/tick | collision rate | success | mean speed |", "|---|---|---|---|---|"]
        for c in caps:
            r = res["S1"][str(c)]
            L.append(f"| {c:g} | {r['tokens_per_tick']:.0f} | {pct(r['collision_rate'])} | {pct(r['success_rate'])} | "
                     f"{r['mean_speed']:.2f} m/s |")
    (ROOT / "docs" / "results_v6.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
