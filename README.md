# DeadlineVLA

Omnidirectional drone VLA under a physical latency deadline. Six cameras cover 360°; every control
tick, the drone's speed and clearance set a latency deadline, the deadline sets a visual-token budget,
the budget is spent mostly in the direction of travel, and a governor slows the drone when no budget
is safe. Full design: [docs/design.md](docs/design.md).

```
clearance d ─┐
speed v ─────┼─> L_max(v,d) ─> allocate tokens over 6 views ─> VLA (async chunks) ─> govern speed ─> drone
latency table┘        (deadline.py)      (allocator.py)          (executor.py)      (governor.py)
```

## Status

| Part | State |
|---|---|
| Deadline, governor, latency model, exact allocator, clearance (oracle + depth) | Implemented and unit-tested, including the design doc's worked examples |
| Light simulator (numpy multirotor, 6 ray-cast cameras, 3 scene families), A* expert | Runs here; expert solves all three families collision-free |
| Recording (LeRobot when installed, LeRobot-shaped npz otherwise), budget-randomised training, offline eval | Runs here with the `mock` policy |
| Pilot loop, async/sync chunk executor, baselines B1–B4, metrics, report, CI gates, CLI | Runs here end to end |
| Isaac Lab 3.0 env, SmolVLA and GR00T N1.7 adapters, TensorRT runtime | Written against the upstream APIs, **never executed** (no GPU/Jetson here). Validating them is milestone M0 |

Everything run so far used a **synthetic** Jetson latency model (`serve.board=synthetic_orin_nx`), and the
policies were stand-ins (`mock`, `surrogate`), not a VLA. Reports say so in a banner.

## Install

```bash
python -m venv .venv
.venv/Scripts/pip install -e .[dev]        # light path: numpy, pyyaml, matplotlib, pytest, ruff
# real backends:  pip install -e .[lerobot]  (SmolVLA),  .[trt] (TensorRT export)
# Isaac Lab 3.0 and Isaac-GR00T: install from NVIDIA's instructions
```

## Run

One command reproduces a full light-path run (about 15 min on a laptop CPU):

```bash
deadlinevla all --config configs/light_surrogate.yaml        # -> runs/light_surrogate/report/report.md
```

Stages can run one at a time; each reads the previous one's outputs from `manifest.json`:

```bash
deadlinevla env-check --config configs/light.yaml   # measures a_brake, checks render rate, go/no-go
deadlinevla record    --config configs/light.yaml   # expert demos, all six views at the top tier
deadlinevla train     --config configs/light.yaml   # budget-randomised fine-tuning + offline eval
deadlinevla export    --config configs/light.yaml   # per-tier TensorRT engines (or a note for simulated)
deadlinevla profile   --config configs/light.yaml   # latency table over all 4096 allocations + linear fit
deadlinevla eval      --config configs/light.yaml   # methods x governor x speed caps x scenes
deadlinevla report    --config configs/light.yaml   # figures, report.md, gates (exit 1 if a gate fails)
```

Any field can be overridden: `deadlinevla eval --config configs/light.yaml drone.margin=0.8 "eval.speed_caps=[4, 8]"`.
Each run directory gets `config.resolved.yaml`.

| Config | What it is |
|---|---|
| `configs/default.yaml` | The real setup: Isaac Lab, SmolVLA, TensorRT on a Jetson Orin NX |
| `configs/light.yaml` | Laptop CPU: numpy sim, `mock` learned policy, simulated synthetic board |
| `configs/light_surrogate.yaml` | As light, with the perception-limited `surrogate` policy |

The **surrogate** policy reads ground truth, but only what the allocation lets it see: an obstacle is
detected if it is inside an active camera's field of view and within that tier's range
(`model.surrogate_ranges`). It lets the deadline module be studied before a real VLA exists. The
**mock** policy is a small numpy MLP that exercises the training contract (variable views, view/tier
codes, per-sample allocations); it is a plumbing test, not a pilot.

## Light-path results, v2 (synthetic board, surrogate policy)

200 scenes per cell (50 per family × 4 families), 6 methods × governor on/off × 5 speed caps =
12,000 episodes; about 2.5 min on 4 laptop cores. Reproduce with
`deadlinevla all --config configs/light_surrogate.yaml`. Full tables, Wilson intervals and the
violation-cause breakdown are in `runs/light_surrogate/report/report.md`.

![speed vs compute](runs/light_surrogate/report/speed_vs_compute.png)
![rear threat](runs/light_surrogate/report/rear_threat.png)

- **Static families (goal_nav, sparse_nav, slalom), governed: zero collisions in 6,000 episodes.** The v2
  allocator is the fastest method with 100% success (3.54 m/s at the 10 m/s cap, versus 3.42 for uniform
  B2) on ~215 tokens per tick, about 45% fewer than B2's 384. B4 front-only is about as fast but succeeds
  only 88% of the time (goals behind it). v1 and all-high are the slowest (≈2.8 m/s) and time out more.
- **rear_threat, governed:** every governed collision happens here, because the governor watches only
  the cone ahead and cannot react to an obstacle closing from behind. v2 collides in 10% of episodes
  [4–21%], the same as B2/B3; v1 and all-high collide in 22%. With n = 50 the v2-vs-uniform difference
  is not resolvable.
- **Governed violation rate: 0.2–1%** (was 19–32% before the governor fixes below), well under the ~5%
  expected from the p95 latency tail.
- **Without the governor**, every method crashes in 25–37% of episodes at caps ≥ 8 m/s.

All latency here comes from an invented board, so the numbers say nothing yet about an Orin.

### What changed from v1, and why

| Change | Evidence |
|---|---|
| Allocator objective is now *speed* (`budget.objective: speed`): the cheapest allocation meeting a weighted perception floor Q_min, with every view at least low | v1 filled any slack with ~1,300 tokens; with the governor on, latency costs speed (a = 8.4, D = 5 m: v* 7.85 → 6.99 m/s from L = 0.17 → 0.30 s). v1 kept as `deadline_v1` |
| 360° floor kept as a hard constraint in v2 | Without it, the weighted Q_min switched rear views off and paid with a sharper front view (`[3,1,0,0,0,1]`) |
| Governor blind time L + dt + τ_track, with τ_track measured by `env-check` (0.09 s) | Violation-cause log: 85–90% of governed violations were `dynamics_lag`, where v* fell ~0.4 m/s per tick but the velocity controller closed only ~63% of the gap |
| Governor clearance = min(command, velocity) while moving, command only at hover | A turn sweeps both directions; at hover the drone must still be able to back away |
| Per-tick violation causes, Wilson intervals, moving-ticks-only violation rate | Needed to tell bugs from physics |
| B3 uses per-camera obstacle proximity; eval warns if B3 ≈ B2 on > 95% of ticks | B3 was identical to B2 in every cell |
| `sparse_nav` family (quarter clutter); 50 scenes per family; `rear_threat` reported separately from the static families | Governor always bound in clutter; 6 scenes per cell was too few; rear-threat failures hid the static-family result |
| `sim.render: false` for the surrogate, parallel sweep (`eval.workers`) | 12,000 episodes went from hours to minutes |

## Implementation notes (where the code goes beyond or interprets the design doc)

- **Braking.** `measure_braking` returns the conservative `min_v v²/(2 s(v))`, so `v²/(2a)` bounds every
  measured stop including controller lag (light sim: a ≈ 8.4 m/s²).
- **Governor.** The deadline uses clearance along the current velocity (doc step 1). The governor uses
  min(command, velocity) clearance while moving and command clearance at hover, so a drone inside the
  margin can be commanded away. Its blind time is L + dt + τ_track (see the v2 table above).
- **Latency in simulation.** Rollouts sample latency from the board (synthetic, or the measured Jetson
  table replayed as a lognormal with its p50/p95, clipped at max), never from the desktop. This is the
  "injected delay in sim" answer to the hardware-in-the-loop risk.
- **Async splicing.** If latency exceeds the chunk horizon, the chunk's last action is kept rather than
  nothing.
- **Violation rate** counts moving ticks only (v ≥ hover speed), since while hovering L_max is infinite. It uses the latency of the inference whose chunk is actually being flown.
- **Linear latency fit.** `c` and `e(r)` are only separable with per-stage timings (`encode`, `prefill`);
  from totals alone the fit reports an effective per-view cost per tier. The synthetic board has a
  quadratic attention term, so its fit falls back to the table, as the doc anticipates.
- **TensorRT.** Vision encoders are compiled per tier (the allocation-dependent cost); backbone prefill
  and the action head run in fp16 PyTorch for now. Full-TRT backbone/head is an M3 task.
- **SmolVLA plumbing.** The adapter wraps `prepare_images`/`embed_image` on the loaded instance only;
  `fixed_slots=True` is the doc's fallback (six slots, masked off views, no compute saving).

## Layout

```
src/deadlinevla/
  config.py manifest.py cli.py
  sim/     scenes.py drone_env.py light_env.py isaac_env.py expert.py
  data/    record.py dataset.py
  budget/  deadline.py governor.py latency_model.py allocator.py clearance.py
  policy/  base.py views.py mock.py surrogate.py smolvla.py gr00t.py
  train/   sft.py
  serve/   runtime.py executor.py pilot.py
  eval/    baselines.py rollout.py sweep.py metrics.py report.py
  ops/     gates.py
tests/     80 tests incl. an end-to-end CLI run (pytest, ~20 s)
```

## Next steps (milestone M0, go / no-go)

1. Isaac Lab 3.0: run `deadlinevla env-check` with `sim.backend=isaac`; fix `sim/isaac_env.py` against the installed API.
2. SmolVLA: load `lerobot/smolvla_base` through `policy/smolvla.py` with views dropped and tiers mixed.
3. Jetson: `export` + `profile` with `serve.runtime=trt`, then set `serve.board` to the board name so eval replays its table.
4. Novelty sweep (Risks) before M1.

## Licenses

Code: MIT. SmolVLA and LeRobot are open source; GR00T N1.7 weights are under the NVIDIA Open Model
License. Sim-to-real is out of scope for v1: nothing here has flown.
