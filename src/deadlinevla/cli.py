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


def log(msg: str) -> None:
    print(msg, flush=True)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _resolve(cfg: Config, man: Manifest) -> Config:
    """Fill 'measured' fields from earlier stages."""
    if cfg.drone.a_brake == "measured" and man.has("env-check", "a_brake"):
        cfg = set_field(cfg, "drone.a_brake", float(man.get("env-check", "a_brake")))
    return cfg


def _require_a_brake(cfg: Config, run_dir: Path, man: Manifest) -> Config:
    if cfg.drone.a_brake == "measured":
        log("a_brake not measured yet: running env-check first")
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

    backends = {m: _has(m) for m in ("isaaclab", "torch", "lerobot", "tensorrt", "gr00t")}
    log("backends: " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in backends.items()))
    env = make_env(cfg)
    braking = measure_braking_details(env)
    log(f"measured braking deceleration a = {braking['a_brake']:.3f} m/s^2 "
        f"(fit a={braking['fit_a']}, controller lag {braking['fit_lag_s']:.3f} s)")
    cfg = set_field(cfg, "drone.a_brake", braking["a_brake"])
    env = make_env(cfg)
    obs = env.reset(build_scene(cfg.scenes.families[0], cfg.scenes.seed, cfg))
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        obs, _ = env.step(np.zeros(4))
    fps = n / (time.perf_counter() - t0)
    log(f"six-camera step rate: {fps:.1f} steps/s at {cfg.tier_res[3]} px (control rate {cfg.drone.ctrl_hz:g} Hz)")

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
    man.record("env-check", a_brake=braking["a_brake"], braking=braking, step_rate=fps, backends=backends,
               go_no_go=checks, views="env_check_views.png", backend=cfg.sim.backend)


def cmd_record(cfg: Config, run_dir: Path, man: Manifest) -> None:
    from deadlinevla.data.record import record_demos
    from deadlinevla.sim.drone_env import make_env
    from deadlinevla.sim.expert import Expert

    cfg = _require_a_brake(cfg, run_dir, man)
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
    from deadlinevla.eval.rollout import build_eval_scenes, run_episodes
    from deadlinevla.serve.pilot import Pilot
    from deadlinevla.serve.runtime import SimulatedRuntime
    from deadlinevla.sim.drone_env import make_env

    cfg = _require_a_brake(cfg, run_dir, man)
    latency, table = _latency_model(cfg, man)
    adapter = _load_adapter(cfg, man)
    # The simulated loop sees the onboard computer's latency, never the desktop's.
    source = make_latency_source(cfg.serve.board, cfg.tier_tokens, table)
    env = make_env(cfg)
    scenes = build_eval_scenes(cfg)
    log(f"eval: {len(scenes)} scenes x {len(cfg.eval.methods)} methods x {len(cfg.eval.governor)} governor "
        f"settings x {len(cfg.eval.speed_caps)} speed caps; latency source {source.name}")
    results, summary = [], {}
    with open(run_dir / "episodes.jsonl", "w", encoding="utf-8") as ep_file:
        for method in cfg.eval.methods:
            for gov in cfg.eval.governor:
                all_logs = []
                for cap in cfg.eval.speed_caps:
                    rng = np.random.default_rng([cfg.run.seed, int(cap * 100)])
                    pilot = Pilot(cfg, SimulatedRuntime(adapter, source, rng), latency, method, gov, adapter)
                    logs = run_episodes(pilot, env, scenes, cap)
                    all_logs += logs
                    for lg in logs:
                        ep_file.write(json.dumps(lg.summary()) + "\n")
                    m = summarize(logs).to_dict()
                    results.append({"method": method, "governor": gov, "speed_cap": cap, "family": "all", **m})
                    for fam in sorted({lg.family for lg in logs}):
                        fm = summarize([lg for lg in logs if lg.family == fam]).to_dict()
                        results.append({"method": method, "governor": gov, "speed_cap": cap, "family": fam, **fm})
                    cpk = m["collisions_per_km"] if m["collisions_per_km"] is not None else float("nan")
                    log(f"  {method:14s} gov={str(gov):5s} cap={cap:4.1f}  success={m['success_rate']:.2f} "
                        f"coll/km={cpk:6.2f} "
                        f"speed={m['mean_speed']:.2f} tokens={m['tokens_per_tick']:.0f} viol={m['violation_rate']:.3f}")
                summary[f"{method}|{'gov' if gov else 'nogov'}"] = summarize(all_logs).to_dict()
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    man.record("eval", results="results.jsonl", summary="summary.json", episodes="episodes.jsonl",
               latency_source=source.name, synthetic_latency=source.synthetic)


def cmd_report(cfg: Config, run_dir: Path, man: Manifest) -> int:
    from deadlinevla.eval.report import write_report
    from deadlinevla.ops.gates import check_gates

    results = [json.loads(line) for line in man.path_of("eval", "results").read_text().splitlines() if line]
    notes = []
    if man.get("eval", "synthetic_latency", False):
        notes.append("Latency comes from a SYNTHETIC board model, not a measured Jetson table.")
    if cfg.model.policy in ("mock", "surrogate"):
        notes.append(f"Policy is the '{cfg.model.policy}' stand-in, not a VLA.")
    path = write_report(results, run_dir / "report", cfg.eval.collision_threshold, notes)
    log(f"report: {path}")
    summary = json.loads(man.path_of("eval", "summary").read_text())
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


HANDLERS = {"env-check": cmd_env_check, "record": cmd_record, "train": cmd_train, "export": cmd_export,
            "profile": cmd_profile, "eval": cmd_eval, "report": cmd_report}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="deadlinevla", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS + ("all",))
    ap.add_argument("--config", default=None, help="YAML config (default: built-in defaults)")
    ap.add_argument("--run-dir", default=None, help="output folder (default: run.out_dir)")
    ap.add_argument("overrides", nargs="*", help="section.field=value")
    args = ap.parse_args(argv)

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
