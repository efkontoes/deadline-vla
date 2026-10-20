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

## First light-path result (synthetic board, surrogate policy, 12 scenes per cell)

![speed vs compute](runs/light_surrogate/report/speed_vs_compute.png)

Read honestly, this toy setup does **not** yet support the headline claim:

- The **governor** is what buys safety: without it, every method collides at ≥ 30 collisions/km once
  the cap exceeds 4–6 m/s; with it, most cells reach zero.
- The **deadline allocator** spends heavily whenever there is slack (≈1100–1400 tokens/tick, ~0.3 s
  latency), and in this setup a cheap uniform allocation (B2, 384 tokens, ~0.17 s) flies faster. The
  objective rewards tokens, not freshness, and the surrogate's value of a tier is only its range.
- Deadline-violation rates stay at 10–35 % even with the governor (see below).

These are properties of the method as specified meeting a toy perception model, which is exactly what
the light path is for; they are the first things to look at (quality function, a latency term in the
objective, κ/ε, `surrogate_ranges`) before M5 runs it with a real model and board latency.

## Implementation notes (where the code goes beyond or interprets the design doc)

- **Braking.** `measure_braking` returns the conservative `min_v v²/(2 s(v))`, so `v²/(2a)` bounds every
  measured stop including controller lag (light sim: a ≈ 8.4 m/s²).
- **Governor clearance.** The deadline uses clearance along the current velocity (doc step 1). The
  governor uses clearance along the *commanded* direction; otherwise a drone inside the margin can never
  be commanded away and deadlocks.
- **Latency in simulation.** Rollouts sample latency from the board (synthetic, or the measured Jetson
  table replayed as a lognormal with its p50/p95, clipped at max), never from the desktop. This is the
  "injected delay in sim" answer to the hardware-in-the-loop risk.
- **Async splicing.** If latency exceeds the chunk horizon, the chunk's last action is kept rather than
  nothing.
- **Violation rate** counts moving ticks only (v ≥ hover speed); while hovering L_max is infinite.
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
  eval/    baselines.py rollout.py metrics.py report.py
  ops/     gates.py
tests/     69 tests incl. an end-to-end CLI run (pytest, ~20 s)
```

## Next steps (milestone M0, go / no-go)

1. Isaac Lab 3.0: run `deadlinevla env-check` with `sim.backend=isaac`; fix `sim/isaac_env.py` against the installed API.
2. SmolVLA: load `lerobot/smolvla_base` through `policy/smolvla.py` with views dropped and tiers mixed.
3. Jetson: `export` + `profile` with `serve.runtime=trt`, then set `serve.board` to the board name so eval replays its table.
4. Novelty sweep (Risks) before M1.

## Licenses

Code: MIT. SmolVLA and LeRobot are open source; GR00T N1.7 weights are under the NVIDIA Open Model
License. Sim-to-real is out of scope for v1: nothing here has flown.
