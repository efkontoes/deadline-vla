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

## Light-path results, v3 (synthetic board, surrogate policy)

Per cell: 150 static scenes (50 each of goal_nav, sparse_nav, slalom) plus 450 rear_threat scenes (threats at
up to 10 m/s), every method on identical scenes. Sweep: 6 methods × governor on/off × 5 speed caps =
36,000 episodes, about 14 min on 4 laptop cores. Reproduce with
`deadlinevla all --config configs/light_surrogate.yaml`. Headline numbers below are governed, at the 10 m/s
cap, paired scene by scene against B2 uniform; full tables are in `runs/light_surrogate/report/report.md`.

**Headline: equal safety and speed for about a third fewer tokens.**

| | tokens/tick | success | collisions | speed vs B2, paired 95% CI |
|---|---|---|---|---|
| static families, deadline (ours) | 247 (−36%) | 1.000 | 0.000 | +0.078 m/s [+0.061, +0.096] |
| static families, B2 uniform | 384 | 1.000 | 0.000 | (reference, 4.28 m/s) |
| rear_threat, deadline (ours) | 247 (−36%) | 0.993 | 0.002 | +0.071 m/s [+0.057, +0.086] |
| rear_threat, B2 uniform | 384 | 0.984 | 0.011 (McNemar p = 0.13 vs ours) | (reference, 4.53 m/s) |

- The speed gain over B2 is real but small (~2%); the efficiency is the result. On a real Orin, ~36% fewer
  visual tokens is headroom for a smaller board or a bigger model.
- **Part of this is by construction.** The surrogate sees an obstacle if an active camera at that tier would,
  so the link between tokens and perception is something we wrote, not measured. A real VLA may need more
  tokens to see the same thing. The efficiency number is a hypothesis until M4.
- v1 (quality objective) and B1 all-high fly ~0.77 m/s slower and succeed less often (88–89%): spending tokens
  costs speed through the governor.
- **Relative vs own-motion governor** (`deadlinevla compare`, same scenes): rear_threat collisions fall for
  every method, e.g. ours 4.0% → 0.2% (p = 1.5e-5), all-high 21% → 1.6%. It is also slightly faster
  (+0.04 m/s on static scenes, CI above zero), with no static collisions either way.
- **Confound:** the relative governor works from ground truth (or, in the ablation, from a separate depth
  sensor), not from the VLA's tokens, so it protects every allocation from behind, front-only included.
  rear_threat therefore tests the governor, not the allocator; 360° VLA perception has to show its value on
  the policy side.
- At the 2 m/s cap, rear-threat collisions rise for every method: a drone capped at 2 m/s cannot outrun a
  faster threat. This is physics, not a bug.

All latency here comes from an invented board, so the numbers say nothing yet about an Orin.

![speed vs compute](runs/light_surrogate/report/speed_vs_compute.png)
![rear threat](runs/light_surrogate/report/rear_threat.png)

### What changed from v2, and why

| Change | Evidence |
|---|---|
| Relative deadline per contact (`budget.relative`): L_max,i = (d_i − m − c_i²/2a)/c_i with closing speed c_i; reduces to the original for a static obstacle ahead | In v2, every governed collision was in rear_threat: own-motion reasoning never sees a threat from behind |
| Closing-speed camera weights | In a static world they equal the old weights; a threat from behind raises the rear weight early |
| Escape governor: each threat constrains the commanded closing speed; ahead → slow down, otherwise → push away along −r̂; inside the margin, slide but don't approach | A threat from behind can't be braked away from; v2's scaling-only governor had no response to it |
| Threat = on a collision course **or** inside the 30° cone ahead of velocity or command | Course-only threats let a grazed slalom obstacle become a collision course at 1.8 m and 5 m/s after one heading change (13–19 of 50 slalom runs collided). The cone restores v2's safety ahead |
| Surrogate passes obstacles outside the governor's margin | Otherwise every planned pass looks like a collision course (goal_nav speed fell to 1.3 m/s) |
| rear_threat at threat_speed 5 and 450 scenes; paired bootstrap and McNemar; headline section; `deadlinevla compare` | Two-proportion power analysis: ~434 scenes to tell 10% from 5%; speed needs a paired CI before claiming "fastest" |

### What changed from v1 to v2, and why

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
  budget/  deadline.py governor.py relative.py latency_model.py allocator.py clearance.py
  policy/  base.py views.py mock.py surrogate.py smolvla.py gr00t.py
  train/   sft.py
  serve/   runtime.py executor.py pilot.py
  eval/    baselines.py rollout.py sweep.py metrics.py stats.py report.py
  ops/     gates.py
tests/     89 tests incl. an end-to-end CLI run (pytest, ~40 s)
```

## Next steps (milestone M0, go / no-go)

1. Isaac Lab 3.0: run `deadlinevla env-check` with `sim.backend=isaac`; fix `sim/isaac_env.py` against the installed API.
2. SmolVLA: load `lerobot/smolvla_base` through `policy/smolvla.py` with views dropped and tiers mixed.
3. Jetson: `export` + `profile` with `serve.runtime=trt`, then set `serve.board` to the board name so eval replays its table.
4. Novelty sweep (Risks) before M1.

## Licenses

Code: MIT. SmolVLA and LeRobot are open source; GR00T N1.7 weights are under the NVIDIA Open Model
License. Sim-to-real is out of scope for v1: nothing here has flown.
