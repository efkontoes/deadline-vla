"""Command line: env-check, record, train, export, profile, eval, report (and ``all``).

    deadlinevla <command> --config configs/light.yaml [--run-dir runs/x] [section.field=value ...]

Each command reads the previous stage's outputs from ``<run-dir>/manifest.json`` and writes
``config.resolved.yaml`` next to its outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from deadlinevla.config import Config, load_config, save_resolved, set_field
from deadlinevla.manifest import Manifest

COMMANDS = ("env-check", "record", "train", "export", "profile", "eval", "report")
EXTRA = ("compare",)


def log(msg: str) -> None:
    print(msg, flush=True)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _resolve(cfg: Config, man: Manifest) -> Config:
    """Fill 'measured' fields from earlier stages."""
    for name in ("a_brake", "tau_track"):
        if getattr(cfg.drone, name) == "measured" and man.has("env-check", name):
            cfg = set_field(cfg, f"drone.{name}", float(man.get("env-check", name)))
    return cfg


def _require_a_brake(cfg: Config, run_dir: Path, man: Manifest) -> Config:
    if "measured" in (cfg.drone.a_brake, cfg.drone.tau_track):
        log("a_brake / tau_track not measured yet: running env-check first")
        cmd_env_check(cfg, run_dir, man)
        cfg = _resolve(cfg, man)
    return cfg


def _latency_table_path(cfg: Config, man: Manifest) -> Optional[Path]:
    if cfg.budget.latency_table != "measured":
        return Path(cfg.budget.latency_table)
    if man.has("profile", "latency_table"):
        return man.path_of("profile", "latency_table")
    return None


def _latency_model(cfg: Config, man: Manifest, required: bool = True):
    from deadlinevla.budget.latency_model import LatencyModel, LatencyTable

    path = _latency_table_path(cfg, man)
    if path is None:
        if required:
            raise SystemExit("no latency table: run 'profile' first or set budget.latency_table")
        return None, None
    table = LatencyTable.load(path)
    return LatencyModel(table, cfg.budget.latency_stat, cfg.budget.max_residual, cfg.tier_tokens), table


def _load_adapter(cfg: Config, man: Manifest):
    from deadlinevla.policy.base import make_policy

    adapter = make_policy(cfg)
    if adapter.trainable:
        ckpt = man.path_of("train", "checkpoint")
        adapter.load(ckpt, cfg.model.device)
    return adapter


# --------------------------------------------------------------------------------------------------


def cmd_env_check(cfg: Config, run_dir: Path, man: Manifest) -> None:
    from deadlinevla.sim.drone_env import make_env, measure_braking_details
    from deadlinevla.sim.scenes import build_scene

    cfg = set_field(cfg, "sim.render", True)
    backends = {m: _has(m) for m in ("isaaclab", "torch", "lerobot", "tensorrt", "gr00t")}
    log("backends: " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in backends.items()))
    env = make_env(cfg)
    braking = measure_braking_details(env)
    log(f"measured braking deceleration a = {braking['a_brake']:.3f} m/s^2 "
        f"(fit a={braking['fit_a']}, controller lag {braking['fit_lag_s']:.3f} s); "
        f"velocity-tracking time constant tau = {braking['tau_track']:.3f} s")
    cfg = set_field(set_field(cfg, "drone.a_brake", braking["a_brake"]), "drone.tau_track", braking["tau_track"])
    env = make_env(cfg)
    obs = env.reset(build_scene(cfg.scenes.families[0], cfg.scenes.seed, cfg))
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        obs, _ = env.step(np.zeros(4))
    fps = n / (time.perf_counter() - t0)
    log(f"six-camera step rate: {fps:.1f} steps/s at {cfg.tier_res[3]} px (control rate {cfg.drone.ctrl_hz:g} Hz)")

    import matplotlib

    matplotlib.use("Agg")  # never a GUI backend: Tk objects collected from worker threads crash the process
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, cfg.cameras.n_cams, figsize=(2 * cfg.cameras.n_cams, 2.2), dpi=120)
    for j, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(obs["images"][j])
        ax.set_title(f"cam{j} ({cfg.cameras.cam_yaws[j]:g}°)", fontsize=8)
        ax.axis("off")
    fig.suptitle(obs["instruction"], fontsize=9)
    fig.tight_layout()
    run_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(run_dir / "env_check_views.png")
    plt.close(fig)

    checks = {
        "six_cameras_render_at_control_rate": bool(fps >= cfg.drone.ctrl_hz),
        "smolvla_variable_views": None if not backends["lerobot"] else "run tests/manual/smolvla_views.py",
        "jetson_board_available": bool(backends["tensorrt"]),
    }
    log("go / no-go: " + json.dumps(checks))
    man.record("env-check", a_brake=braking["a_brake"], tau_track=braking["tau_track"], braking=braking,
               step_rate=fps, backends=backends,
               go_no_go=checks, views="env_check_views.png", backend=cfg.sim.backend)


def cmd_record(cfg: Config, run_dir: Path, man: Manifest) -> None:
    from deadlinevla.data.record import record_demos
    from deadlinevla.sim.drone_env import make_env
    from deadlinevla.sim.expert import Expert

    cfg = set_field(_require_a_brake(cfg, run_dir, man), "sim.render", True)
    env = make_env(cfg)
    root = run_dir / "dataset"
    log(f"recording {cfg.data.episodes} expert episodes -> {root}")
    record_demos(env, Expert(cfg), cfg.data.episodes, cfg.data.repo_id, root, seed0=cfg.scenes.seed, log=log)
    man.record("record", dataset=man.rel(root), episodes=cfg.data.episodes)


def cmd_train(cfg: Config, run_dir: Path, man: Manifest) -> None:
    from deadlinevla.train.sft import train_sft

    cfg = _require_a_brake(cfg, run_dir, man)
    latency, _ = _latency_model(cfg, man, required=False)
    if latency is None:
        log("no measured latency table yet: the allocator-biased sampler uses the synthetic board prior")
    path = train_sft(cfg, run_dir, man.path_of("record", "dataset"), latency, log=log)
    man.record("train", checkpoint=man.rel(path), policy=cfg.model.policy)


def cmd_export(cfg: Config, run_dir: Path, man: Manifest) -> None:
    from deadlinevla.serve.runtime import export_engines

    adapter = _load_adapter(cfg, man)
    path = export_engines(adapter, cfg, run_dir / "engines")
    log(f"export manifest: {path}")
    man.record("export", export=man.rel(path), engines=man.rel(path.parent))


def cmd_profile(cfg: Config, run_dir: Path, man: Manifest) -> None:
    from deadlinevla.budget.latency_model import (
        LatencyModel,
        SyntheticBoard,
        all_allocs,
        profile_allocations,
    )
    from deadlinevla.serve.runtime import SimulatedRuntime, TrtRuntime

    rng = np.random.default_rng(cfg.run.seed)
    allocs = all_allocs(cfg.cameras.n_cams)
    if cfg.serve.runtime == "simulated":
        board = SyntheticBoard(cfg.serve.board, cfg.tier_tokens)
        runtime, obs = SimulatedRuntime(None, board, rng, latency_only=True), None
        meta = {"board": board.name, "synthetic": True, "runtime": "simulated"}
        log(f"profiling SYNTHETIC board {board.name}: {len(allocs)} allocations x {cfg.serve.profile_iters} iters")
    else:  # pragma: no cover - on the Jetson
        from deadlinevla.sim.drone_env import make_env
        from deadlinevla.sim.scenes import build_scene

        cfg = _require_a_brake(cfg, run_dir, man)
        adapter = _load_adapter(cfg, man)
        runtime = TrtRuntime(adapter, man.path_of("export", "engines"))
        obs = make_env(cfg).reset(build_scene("goal_nav", 0, cfg))
        meta = {"board": cfg.serve.board, "synthetic": False, "runtime": cfg.serve.runtime}
        log(f"profiling {cfg.serve.board}: {len(allocs)} allocations x {cfg.serve.profile_iters} iters")
    table = profile_allocations(runtime, allocs, cfg.serve.profile_iters, obs=obs, meta=meta)
    path = table.save(run_dir / "latency_table.npz")
    model = LatencyModel(table, cfg.budget.latency_stat, cfg.budget.max_residual, cfg.tier_tokens)
    desc = model.describe()
    (run_dir / "latency_fit.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")
    log(f"linear fit worst residual {desc['worst_residual'] * 1000:.2f} ms (rms {desc['rms_residual'] * 1000:.2f} ms)"
        f" -> predict uses the {desc['source']}")
    log(f"latency range p95: {table.p95.min() * 1000:.1f} .. {table.p95.max() * 1000:.1f} ms")
    man.record("profile", latency_table=man.rel(path), fit=desc)


def cmd_eval(cfg: Config, run_dir: Path, man: Manifest) -> None:
    from deadlinevla.budget.latency_model import make_latency_source
    from deadlinevla.eval.metrics import summarize
    from deadlinevla.eval.rollout import build_eval_scenes
    from deadlinevla.eval.sweep import run_sweep

    cfg = _require_a_brake(cfg, run_dir, man)
    _, table = _latency_model(cfg, man)
    source = make_latency_source(cfg.serve.board, cfg.tier_tokens, table)
    ckpt = man.path_of("train", "checkpoint") if man.has("train", "checkpoint") else None
    scenes = build_eval_scenes(cfg)
    log(f"eval: {len(scenes)} scenes x {len(cfg.eval.methods)} methods x {len(cfg.eval.governor)} governor "
        f"settings x {len(cfg.eval.speed_caps)} speed caps on {cfg.eval.workers} worker(s); "
        f"latency source {source.name}{' (SYNTHETIC)' if source.synthetic else ''}")
    results: list[dict] = []
    by_method: dict[str, list] = {}
    ep_file = open(run_dir / "episodes.jsonl", "w", encoding="utf-8")

    def on_cell(cell, logs):
        method, gov, cap = cell
        by_method.setdefault(f"{method}|{'gov' if gov else 'nogov'}", []).extend(logs)
        for lg in logs:
            ep_file.write(json.dumps(lg.summary()) + "\n")
        m = summarize(logs).to_dict()
        results.append({"method": method, "governor": gov, "speed_cap": cap, "family": "all", **m})
        for fam in sorted({lg.family for lg in logs}):
            fm = summarize([lg for lg in logs if lg.family == fam]).to_dict()
            results.append({"method": method, "governor": gov, "speed_cap": cap, "family": fam, **fm})
        # "static": every family without a moving obstacle. The forward-cone governor can stop for what is
        # ahead, but not for a threat closing from behind, so the two kinds of failure are reported apart.
        static = [lg for lg in logs if lg.family != "rear_threat"]
        if static and len(static) < len(logs):
            sm = summarize(static).to_dict()
            results.append({"method": method, "governor": gov, "speed_cap": cap, "family": "static", **sm})
        cpk = m["collisions_per_km"] if m["collisions_per_km"] is not None else float("nan")
        log(f"  {method:14s} gov={str(gov):5s} cap={cap:4.1f}  success={m['success_rate']:.2f} "
            f"coll/km={cpk:6.2f} speed={m['mean_speed']:.2f} tokens={m['tokens_per_tick']:.0f} "
            f"viol={m['violation_rate']:.3f}")

    try:
        run_sweep(cfg, scenes, _latency_table_path(cfg, man), ckpt, cfg.eval.workers, on_cell)
    finally:
        ep_file.close()
    summary = {k: summarize(v).to_dict() for k, v in by_method.items()}
    notes = _duplicate_baseline_check(cfg, by_method)
    for n in notes:
        log("WARNING: " + n)
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    man.record("eval", results="results.jsonl", summary="summary.json", episodes="episodes.jsonl",
               latency_source=source.name, synthetic_latency=source.synthetic, scenes=len(scenes), notes=notes)


def _duplicate_baseline_check(cfg: Config, by_method: dict[str, list]) -> list[str]:
    """Flag B3 when it is just B2 in disguise (no usable importance signal from the policy)."""
    notes = []
    uniform = np.full(cfg.cameras.n_cams, cfg.eval.uniform_tier)
    for key, logs in by_method.items():
        if not key.startswith("importance|") or not logs:
            continue
        allocs = np.concatenate([lg.ticks["alloc"] for lg in logs])
        same = float((allocs == uniform).all(axis=1).mean())
        if same > 0.95:
            notes.append(f"{key}: B3 chose the B2 allocation on {same:.0%} of ticks; it is not a distinct baseline "
                         f"with policy '{cfg.model.policy}'")
    return notes


def cmd_report(cfg: Config, run_dir: Path, man: Manifest) -> int:
    from deadlinevla.eval.report import write_report
    from deadlinevla.ops.gates import check_gates

    results = [json.loads(line) for line in man.path_of("eval", "results").read_text().splitlines() if line]
    notes = []
    if man.get("eval", "synthetic_latency", False):
        notes.append("Latency comes from a SYNTHETIC board model, not a measured Jetson table.")
    if cfg.model.policy in ("mock", "surrogate"):
        notes.append(f"Policy is the '{cfg.model.policy}' stand-in, not a VLA.")
    notes += man.get("eval", "notes", [])
    summary = json.loads(man.path_of("eval", "summary").read_text())
    episodes = [json.loads(line) for line in man.path_of("eval", "episodes").read_text().splitlines() if line]
    path = write_report(results, run_dir / "report", cfg.eval.collision_threshold, notes,
                        synthetic=bool(man.get("eval", "synthetic_latency", False)), summary=summary,
                        episodes=episodes)
    log(f"report: {path}")
    ours = summary.get("deadline|gov")
    gate = None
    if ours is not None:
        baseline = None
        if cfg.gates.baseline:
            baseline = json.loads(Path(cfg.gates.baseline).read_text()).get("deadline|gov")
        gate = check_gates(ours, baseline, cfg)
        (run_dir / "gates.json").write_text(json.dumps(gate.__dict__, indent=2, default=str), encoding="utf-8")
        log("gates: " + ("PASS" if gate.passed else "FAIL: " + "; ".join(gate.reasons)))
    man.record("report", report=man.rel(path), gates_passed=None if gate is None else gate.passed)
    return 0 if gate is None or gate.passed else 1


def cmd_compare(run_a: Path, run_b: Path) -> None:
    """Paired ablation table: the same methods and scenes in two run directories (A minus B)."""
    from deadlinevla.eval.stats import compare_runs

    load = lambda d: [json.loads(x) for x in (Path(d) / "episodes.jsonl").read_text().splitlines() if x]  # noqa: E731
    ea, eb = load(run_a), load(run_b)
    cap = max(e["speed_cap"] for e in ea)
    fams = sorted({e["family"] for e in ea})
    groups = [("static", [f for f in fams if f != "rear_threat"])] + ([("rear_threat", ["rear_threat"])]
                                                                       if "rear_threat" in fams else [])
    log(f"A = {run_a}   B = {run_b}   governed, cap {cap:g} m/s, paired by scene")
    for name, families in groups:
        log(f"\n{name}: method | n | speed A-B [95% CI] | collision A vs B (p McNemar) | success A vs B")
        for m in sorted({e["method"] for e in ea}):
            c = compare_runs(ea, eb, m, True, families, cap)
            if not c.get("n"):
                continue
            lo, hi = c["speed_diff_ci"]
            col, suc = c["collision"], c["success"]
            log(f"  {m:14s} {c['n']:4d}  {c['speed_diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                f"{col['a']:.3f} vs {col['ref']:.3f} (p={col['p_mcnemar']:.3g})  {suc['a']:.3f} vs {suc['ref']:.3f}")


HANDLERS = {"env-check": cmd_env_check, "record": cmd_record, "train": cmd_train, "export": cmd_export,
            "profile": cmd_profile, "eval": cmd_eval, "report": cmd_report}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="deadlinevla", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS + ("all",) + EXTRA)
    ap.add_argument("--config", default=None, help="YAML config (default: built-in defaults)")
    ap.add_argument("--run-dir", default=None, help="output folder (default: run.out_dir)")
    ap.add_argument("--other", default=None, help="compare: second run directory (B)")
    ap.add_argument("overrides", nargs="*", help="section.field=value")
    args = ap.parse_args(argv)
    if args.command == "compare":
        if not (args.run_dir and args.other):
            ap.error("compare needs --run-dir A --other B")
        cmd_compare(Path(args.run_dir), Path(args.other))
        return 0

    cfg = load_config(args.config, args.overrides)
    run_dir = Path(args.run_dir or cfg.run.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    man = Manifest(run_dir)
    commands = COMMANDS if args.command == "all" else (args.command,)
    code = 0
    for c in commands:
        log(f"== {c} ==")
        cfg_c = _resolve(cfg, man)
        save_resolved(cfg_c, run_dir)
        out: Any = HANDLERS[c](cfg_c, run_dir, man)
        code = out if isinstance(out, int) else code
    return code


if __name__ == "__main__":
    sys.exit(main())
