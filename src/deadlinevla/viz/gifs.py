"""Build the README replay GIFs from finished runs. Scenes are picked by a fixed rule, not by eye.

static   v4 run, static families, cap 10: the first scene (eval order) where B4 front-only collides and ours
         reaches the goal
rear     v4 run, rear_threat, cap 10: same rule
hunter   v5 threat grid, 0.3 m threat at 9 m/s, cap 8: the first scene where B1 all-high is caught while
         front-only and ours (fixed floor) are not: the negative result
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from deadlinevla.config import Config, load_config, set_field
from deadlinevla.eval.rollout import build_eval_scenes
from deadlinevla.viz.replay import Trace, animate_side_by_side, check_against_log, replay_cell

STATIC = ("goal_nav", "sparse_nav", "slalom")
NAMES = {"deadline": "ours (deadline allocator)", "deadline_fixed": "ours (fixed floor)",
         "front_only": "B4 front-only", "all_high": "B1 all-high", "uniform_fixed": "B2 uniform"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def _pick(scenes, episodes, cap, families, bad: str, good: Sequence[str]) -> int:
    out = {(e["method"], e["family"], e["seed"]): e for e in episodes if e["speed_cap"] == cap and e["governor"]}
    for i, sc in enumerate(scenes):
        if sc.family not in families:
            continue
        b = out.get((bad, sc.family, sc.seed))
        g = [out.get((m, sc.family, sc.seed)) for m in good]
        if b and b["collision"] and all(x and x["success"] for x in g):
            return i
    raise LookupError(f"no scene where {bad} collides and {good} succeed")


def _replay_one(cfg, scenes, idx, methods, cap, table, episodes=None) -> list[Trace]:
    want = {(scenes[idx].family, scenes[idx].seed)}
    traces = []
    for m in methods:
        cell = replay_cell(cfg, scenes[: idx + 1], m, cap, table, keep=want)
        if episodes is not None:
            check_against_log(cell, episodes, cap)          # every replayed episode up to idx matches the log
        traces.append(cell[-1])
    return traces


def _agg(rows, family, cap, methods) -> str:
    parts = []
    for m in methods:
        r = next((r for r in rows if r.get("family") == family and r["speed_cap"] == cap and r["governor"]
                  and r["method"] == m), None)
        if r:
            parts.append(f"{NAMES.get(m, m)}: {r['collision_rate'] * 100:.1f}% collisions, "
                         f"{r['tokens_per_tick']:.0f} tokens/tick")
    return "; ".join(parts)


def build_replays(run_dir: str | Path, out_dir: str | Path, cfg_grid: Config | None = None,
                  log: Callable[[str], Any] = print) -> list[Path]:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    cfg = set_field(load_config(run_dir / "config.resolved.yaml"), "sim.render", False)
    table = run_dir / "latency_table.npz"
    episodes = _jsonl(run_dir / "episodes.jsonl")
    results = _jsonl(run_dir / "results.jsonl")
    scenes = build_eval_scenes(cfg)
    cap = max(cfg.eval.speed_caps)
    methods = ("deadline", "front_only", "all_high")
    made = []

    for name, fams, agg_family, label in (("replay_static", STATIC, "static", "static"),
                                          ("replay_rear", ("rear_threat",), "rear_threat", "rear_threat")):
        idx = _pick(scenes, episodes, cap, fams, "front_only", ("deadline",))
        sc = scenes[idx]
        log(f"{name}: {sc.family} seed {sc.seed} (scene {idx}); replaying {len(methods)} methods")
        traces = _replay_one(cfg, scenes, idx, methods, cap, table, episodes)
        n = sum(1 for s in scenes if s.family in fams)
        footer = (f"Selected by rule: the first {label} scene where front-only collides and ours reaches the goal. "
                  f"Over all {n} {label} scenes at {cap:g} m/s: " + _agg(results, agg_family, cap, methods) + ".")
        made.append(animate_side_by_side(
            traces, [NAMES[m] for m in methods], cfg, out_dir / f"{name}.gif",
            headline=f"Same scene, three allocators ({sc.family}, seed {sc.seed}, cap {cap:g} m/s)",
            footer=footer))

    if cfg_grid is not None:
        made.append(build_hunter(cfg_grid, run_dir, out_dir, log))
    return made


def build_hunter(cfg: Config, run_dir: Path, out_dir: Path, log: Callable[[str], Any] = print,
                 size: float = 0.3, speed: float = 9.0, cap: float = 8.0) -> Path:
    """The v5 negative result: an ideal pursuer catches the allocator that sees it best."""
    grid = _jsonl(run_dir / "threat_grid" / "grid.jsonl")
    cell = cfg
    for k, v in (("scenes.threat_mode", "intercept"), ("budget.relative", True), ("perception.governor", "camera"),
                 ("eval.governor", (True,)), ("eval.speed_caps", (cap,)), ("scenes.threat_radius", size / 2),
                 ("scenes.threat_speed", speed), ("sim.render", False)):
        cell = set_field(cell, k, v)
    n = next(r["n"] for r in grid if r["size"] == size and r["speed"] == speed)
    scenes = build_eval_scenes(cell, ["rear_threat"], per_family=n)
    table = run_dir / "latency_table.npz"
    methods = ("deadline_fixed", "all_high", "front_only")
    labels = {"deadline_fixed": "fixed_floor", "all_high": "all_high", "front_only": "front_only"}
    outcomes = {}
    for m in methods:
        cellrun = replay_cell(cell, scenes, m, cap, table, keep=set())
        rate = sum(t.collision for t in cellrun) / len(cellrun)
        logged = next(r for r in grid if r["size"] == size and r["speed"] == speed and r["variant"] == labels[m])
        if abs(rate - logged["collision_rate"]) > 1e-9:
            raise AssertionError(f"hunter replay for {m}: {rate:.3f} collisions, grid logged "
                                 f"{logged['collision_rate']:.3f}")
        outcomes[m] = cellrun
        log(f"replay_hunter: {m} reproduces the grid ({rate:.3f} collisions over {len(cellrun)} scenes)")
    idx = next(i for i in range(len(scenes)) if outcomes["all_high"][i].collision
               and outcomes["front_only"][i].success and outcomes["deadline_fixed"][i].success)
    sc = scenes[idx]
    traces = [replay_cell(cell, scenes[: idx + 1], m, cap, table, keep={(sc.family, sc.seed)})[-1] for m in methods]
    rate_of = {r["variant"]: r["collision_rate"] for r in grid if r["size"] == size and r["speed"] == speed}
    agg = "; ".join(f"{NAMES[m]}: {rate_of[labels[m]] * 100:.0f}% caught" for m in methods)
    footer = (f"Negative result. Selected by rule: first scene where all-high is caught and the others are not. "
              f"Over all {n} scenes ({size:g} m threat at {speed:g} m/s): {agg}.")
    return animate_side_by_side(traces, [NAMES[m] for m in methods], cell, out_dir / "replay_hunter.gif",
                                headline=f"A hunting threat ({size:g} m, {speed:g} m/s): seeing it more did not help",
                                footer=footer)
