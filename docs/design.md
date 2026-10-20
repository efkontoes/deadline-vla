# DeadlineVLA: Omnidirectional Drone VLA under a Physical Latency Deadline

Oct 20, 2026 · Efthymios Kontoes

## What this project is

DeadlineVLA is a vision-language-action policy for a drone with six cameras covering 360 degrees, where how much the model looks, and in which direction, is set every control tick by a physical safety deadline.

**The claim.** On a small onboard computer, a drone cannot afford to process six cameras at full resolution. The faster it flies and the closer the obstacles, the less time each inference may take. We turn that physics into an explicit visual-token budget, spend the budget mostly in the direction of travel, and slow the drone down when no budget is safe. The result we want: higher safe flight speed at the same compute, or the same speed on a smaller computer.

### What we build

1. **Simulator task suite.** A multirotor in Isaac Lab 3.0 with six cameras, procedurally generated cluttered scenes, and language goals that can lie in any direction ("fly to the red backpack behind you").
2. **Expert and dataset.** A privileged scripted expert that flies collision-free using ground-truth geometry; its demonstrations recorded in LeRobot format with all six views.
3. **Budget-aware VLA.** A small open VLA fine-tuned to act from any allocation of tokens across views, including views turned off or at low resolution.
4. **The deadline module.** Latency model of the onboard computer, deadline from speed and clearance, exact view allocator, and a speed governor.
5. **Onboard runtime.** TensorRT engines per resolution tier, async chunk execution, all measured on a Jetson-class board.
6. **Evaluation.** Success, collisions per km, time to goal, deadline violations, against fixed-budget and importance-based baselines. Headline plot: safe speed versus compute.

### Out of scope for v1

- Real flight. Everything runs in simulation; the onboard computer is real, the drone is not.
- Low-level flight control. The policy outputs body-frame velocity commands; a standard controller tracks them.
- Training a VLA from scratch. We fine-tune an existing open model.

Working title: DeadlineVLA. The repo and doc names can change once the novelty check in Risks is done.

## Gap analysis

Each ingredient exists on its own; no work we found combines an aerial VLA, omnidirectional vision and a token budget derived from a physical deadline.

| Area | Existing work | What it does not do |
|---|---|---|
| Drone VLA navigation | AerialVLA fuses front and down views; AutoFly, Exp2VLA | One or two forward-facing views, no compute budgeting |
| Onboard drone VLA | VLA-AN on Jetson Orin NX; LiteVLA-H on AGX Orin, finds latency is prefill dominated | Single camera; latency optimized, not tied to flight state |
| Aerial manipulation VLA | AIR-VLA benchmark, AeroManip-VLA | Manipulation focus, not omnidirectional |
| 360 degree VLA | PanoFuse (pano camera beside a table), PanoVLA (wheeled bimanual), UniTrackPLA (ground navigation) | Ground robots, server-class compute |
| 360 degree drone flight | Fly360: panoramic obstacle avoidance | No language, not a VLA |
| Multi-view token budgets | VLA-ACL fixed K per view; MVPruner budget from instruction text; FastVLA budget from attention spread | Budget never derived from dynamics or a deadline |
| Speed-aware foveation | Fisheye-VLA uses a reference speed to place its high-res budget | Manipulator arm, one fisheye, no deadline |

The empty intersection: aerial platform, 360 degree multi-camera input, and a per-tick token budget computed from speed, clearance and the measured latency of the onboard computer, with a governor that trades speed for safety when compute runs out.

This is "not found in our searches", not "proven untouched". A deeper search is the first task in Risks.

## Method

Every control tick runs five steps: estimate clearance, compute the deadline, allocate tokens across views, run the VLA, cap the speed. Each step is one equation below.

### 1. Deadline from physics (`budget/deadline.py`)

During one inference the drone flies blind for $vL$ metres, then needs $v^2/2a$ to stop. Safe only if both fit inside the clearance:

$$vL + \frac{v^2}{2a} \le d - m \quad\Longrightarrow\quad L_{\max}(v,d) = \frac{d - m - \frac{v^2}{2a}}{v}$$

- $v$: current speed (m/s)
- $L$: observation-to-action latency, end to end (s)
- $a$: maximum braking deceleration (m/s²)
- $d$: clearance to the nearest obstacle inside a cone around the velocity direction (m)
- $m$: safety margin (m)

Illustrative: $v = 8, a = 8, d = 5, m = 0$ gives braking 4 m and $L_{\max} = 1/8 = 0.125$ s. At $v = 4$ the same scene allows $L_{\max} = 1$ s.

### 2. Speed governor (`budget/governor.py`)

The same inequality read the other way gives the fastest safe speed for a latency $L$. Multiply by $2a$ and solve the quadratic in $v$:

$$v^2 + 2aL\,v - 2a(d-m) = 0 \quad\Longrightarrow\quad v^{*}(L,d) = -aL + \sqrt{a^2L^2 + 2a(d-m)}$$

Check with the example: $a = 8, L = 0.125, d - m = 5$ gives $-1 + \sqrt{1 + 80} = 8$ m/s. The governor caps the commanded speed at $v^*$, using the latency of the allocation actually chosen.

### 3. Latency model of the onboard computer (`budget/latency_model.py`)

Each camera $j$ gets a resolution tier $r_j \in \{0 = \text{off}, 1 = \text{low}, 2 = \text{mid}, 3 = \text{high}\}$. Measured once per board:

$$L(\mathbf{r}) = L_0 + \sum_{j=1}^{6} e(r_j) + c \sum_{j=1}^{6} n(r_j) + K\,L_{\text{head}}$$

- $e(r)$: vision-encoder time for one view at tier $r$ ($e(0) = 0$)
- $n(r)$: visual tokens one view produces at tier $r$ ($n(0) = 0$)
- $c$: backbone prefill cost per token
- $K, L_{\text{head}}$: denoising steps and cost per step of the action head
- $L_0$: fixed cost (I/O, preprocessing, transport)

The model is checked against measured latency for every allocation; if the linear form fits badly we store the full measured table instead (there are only $4^6 = 4096$ allocations).

### 4. Direction weights (`budget/allocator.py`)

$$w_j = \epsilon + \max\big(0,\ \hat{\mathbf{v}} \cdot \mathbf{u}_j\big)^{\kappa}$$

- $\mathbf{u}_j$: unit optical axis of camera $j$ in the body frame
- $\hat{\mathbf{v}}$: unit velocity direction
- $\kappa$: sharpness; $\epsilon$: floor weight for views facing away

Six cameras 60 degrees apart, flying straight ahead, $\kappa = 2$, $\epsilon = 0.05$: front 1.05, front-left and front-right 0.30 each, the three rear-facing views 0.05 each.

### 5. Allocation (`budget/allocator.py`)

$$\mathbf{r}^{*} = \arg\max_{\mathbf{r}} \sum_{j} w_j\, q(r_j) \quad \text{s.t.} \quad L(\mathbf{r}) \le L_{\max}(v,d), \quad r_j \ge 1 \ \forall j \text{ if feasible}$$

- $q(r)$: value of a tier, concave in tokens (diminishing returns), for example $q(r) = \log(1 + n(r))$
- the floor constraint keeps every view at least on low resolution when the budget allows, so a threat from behind still reaches the model

With 4096 candidates and a precomputed latency table, exact search costs microseconds. If no allocation meets the deadline, take the cheapest allocation and let the governor slow the drone to $v^*$ for its latency.

When hovering, $\hat{\mathbf{v}}$ is undefined: weights become uniform and the full 360 degree view serves the language task.

### 6. Budget-randomized training (`train/sft.py`)

A pretrained VLA has never seen views switched off or at mixed resolutions. During fine-tuning we sample a random allocation for every training sample, with a learned embedding per view and per tier added to its tokens:

$$\mathcal{L}(\theta) = \mathbb{E}_{(\mathbf{o},\ell,\mathbf{A}) \sim \mathcal{D}}\ \mathbb{E}_{\mathbf{r} \sim p(\mathbf{r})}\ \Big[\ \mathcal{L}_{\text{FM}}\big(\theta;\ \Phi_{\mathbf{r}}(\mathbf{o}), \ell, \mathbf{A}\big)\ \Big]$$

- $\mathcal{D}$: expert demonstrations; $\mathbf{o}$: six images; $\ell$: instruction; $\mathbf{A}$: target action chunk
- $p(\mathbf{r})$: sampling distribution over allocations, biased toward the ones the allocator actually picks
- $\Phi_{\mathbf{r}}$: renders the six views at tiers $\mathbf{r}$
- $\mathcal{L}_{\text{FM}}$: the model's own flow-matching loss

Wrong reading to avoid: this is not data augmentation for robustness. It is what makes a per-tick allocation legal at test time; without it, changing the allocation is a distribution shift every tick.

### 7. Clearance estimate (`budget/clearance.py`)

v1 uses ground-truth distance from the simulator (oracle $d$), isolating the allocation question. v2 replaces it with the minimum depth inside a cone around $\hat{\mathbf{v}}$ from a cheap depth source, reported as an ablation.

## Feasibility check

The project is feasible in simulation with tools that exist today; the two real costs are building the drone environment ourselves and generating our own dataset.

| Need | Option | Verdict |
|---|---|---|
| Multi-camera rendering | Isaac Lab 3.0 cameras: pinhole and fisheye models, RGB and depth, tiled rendering batches all cameras into one render pass (camera docs, Isaac Lab paper) | Use. Depth output also gives us oracle clearance |
| Drone dynamics | Pegasus Simulator: PX4 and ArduPilot integration, last release targets Isaac Sim 4.2; a forum report shows one drone with three extra sensors dropping to about 12 FPS (NVIDIA forum) | Avoid for v1. Version lock and camera cost |
| Drone dynamics | OmniDrones: RL drone platform on Isaac Sim 4.1, several drone models | Reference only. Borrow model parameters, not the stack |
| Drone dynamics | Our own multirotor in Isaac Lab 3.0: rigid body plus a velocity-tracking controller | Use. The policy outputs velocity commands, so a simplified model is enough; braking deceleration $a$ must be measured from it |
| Training data | No public 360 degree drone VLA dataset found | Generate. Scripted expert with privileged geometry, as in Exp2VLA's expert distillation |
| Onboard base model | SmolVLA (about 450M parameters, in LeRobot, already deployed on Jetson by Tether) | Primary. Small enough for a drone-class board |
| Upper-bound model | GR00T N1.7: Qwen3-VL backbone that encodes images at native resolution; inference needs 16 GB+ VRAM, fine-tuning 40 GB+ (repo) | Desktop only, as the large-model reference |
| Onboard computer | Jetson Orin NX 16 GB (used onboard by VLA-AN) or AGX Orin | Required. The latency table must be measured on the real board |

**Go / no-go before writing method code (week 1):**

- [ ] Six cameras on one drone render in Isaac Lab 3.0 at the frame rate the expert needs
- [ ] SmolVLA accepts six views, with views dropped and resolutions mixed, without code surgery beyond its image loop
- [ ] Access to a Jetson Orin-class board for latency profiling

## Control loop

Clearance and speed fix the deadline before any pixel reaches the model; the governor then uses the same clearance and the chosen allocation's latency to cap the speed.

## Code map

Read it top to bottom: the simulator and expert produce demonstrations; the budget package supplies random allocations for training and live allocations plus the speed cap at run time; the pilot is the only place they meet the model. Evaluation drives the same pilot, so baselines and our method share one code path.

## Parameters

One YAML file, parsed into `Config` with one sub-config per section; every field can be overridden from the command line. "Measured" means derived from the simulator or the board, never typed in; "ours" is a starting value to tune.

| Section | Field | Meaning | Start value |
|---|---|---|---|
| drone | a_brake (a) | max braking deceleration | measured from the dynamics model |
| drone | v_max | hard speed cap | ours |
| drone | margin (m) | safety margin added to braking distance | 0.5 m (ours) |
| drone | ctrl_hz | rate velocity commands are sent | ours |
| cameras | n_cams | cameras around the body | 6 |
| cameras | cam_yaws | optical axes u_j, degrees | 0, 60, 120, 180, 240, 300 |
| cameras | model, fov | pinhole or fisheye, field of view | pinhole, about 90 degrees (ours) |
| cameras | tiers | resolution per tier 1, 2, 3 | model dependent, set at M1 |
| scenes | families | task families to generate | goal_nav, rear_threat, slalom |
| scenes | clutter_density, seed | obstacle density and RNG seed | ours |
| expert | planner | global path planner on the occupancy grid | A* |
| expert | speed_profile | expert respects the same braking model | on |
| data | episodes, repo_id | demonstrations recorded, LeRobot repo | grow by learning curve |
| model | policy | smolvla (onboard) or gr00t_n17 (desktop) | smolvla |
| model | chunk_size (H), num_denoise_steps (K) | | from the checkpoint |
| train | alloc_sampler | p(r): uniform, allocator-biased, or mixed | mixed (ours) |
| train | p_view_off | chance a view is switched off in a sample | ours |
| train | lr, batch_size, max_steps | optimizer and length | ours |
| budget | kappa, epsilon | weight sharpness and floor | 2, 0.05 (ours) |
| budget | quality_fn | q(r) | log(1 + n(r)) |
| budget | cone_deg | cone around v-hat used for clearance d | ours |
| budget | clearance_source | oracle (v1) or depth (v2) | oracle |
| budget | latency_table | path to the measured per-allocation table | measured |
| serve | mode | sync or async chunk execution | async |
| serve | precision | TensorRT precision per engine | fp16 |
| eval | methods | ours and baselines B1 to B4 | all |
| eval | episodes_per_cell | rollouts per (method, family, speed cap) | ours |
| eval | speed_caps | v_max values swept for the speed plot | ours |
| gates | max_violation_rate | CI fails above this deadline-violation rate | set after first baseline |

## Function specifications

Each function has one job and typed inputs. Shapes: V = 6 views, H = chunk length, d_a = 4 (body-frame vx, vy, vz, yaw rate). Alloc is a length-6 integer vector of tiers.

### deadlinevla/config.py
- `load_config(path, overrides) -> Config` — Parses YAML, applies key.sub=value overrides, rejects unknown keys, freezes the result, writes config.resolved.yaml next to every output.

### deadlinevla/sim/drone_env.py
- `make_env(cfg) -> DroneEnv` — Builds the Isaac Lab 3.0 scene: one multirotor, six cameras at cam_yaws (RGB plus depth), velocity-tracking controller, num_envs parallel copies.
- `DroneEnv.step(v_cmd: float[d_a]) -> (obs, info)` — Applies the command for one control period. obs holds six RGB images, body velocity, attitude, instruction. info holds collision flag, goal reached, ground-truth clearance.
- `DroneEnv.clearance(v_hat, cone_deg) -> float` — Minimum distance to geometry inside a cone around v_hat, from ground-truth depth. This is the oracle d.
- `measure_braking(env) -> float` — Flies at several speeds, commands zero velocity, fits the deceleration. Writes a_brake; nothing downstream uses a hand-typed value.

### deadlinevla/sim/scenes.py
- `build_scene(family, seed) -> Scene` — Procedurally places obstacles at clutter_density, a start pose and a goal object. goal_nav puts goals in any direction, including behind the start; rear_threat adds a moving obstacle approaching from a side or rear camera; slalom forces fast flight through gaps.
- `make_instruction(scene) -> str` — Templated language goal referencing the goal object's attributes and, sometimes, its relative direction.

### deadlinevla/sim/expert.py
- `plan_path(scene, start, goal) -> Path` — A* on the scene's occupancy grid, smoothed.
- `expert_action(state, path) -> float[d_a]` — Pure-pursuit velocity along the path, with speed capped by the same braking model the policy is judged by, so demonstrations never violate the deadline.

### deadlinevla/data/record.py
- `record_demos(env, expert, n_episodes, repo_id) -> Path` — Rolls out the expert, writes all six views at the highest tier plus state, action, instruction and per-step clearance into a LeRobot dataset. Lower tiers are rendered later by downsampling, so one recording serves every allocation.

### deadlinevla/data/dataset.py
- `load_dataset(cfg) -> Dataset`, `make_dataloader(ds, cfg, sampler) -> DataLoader` — Episode-level train/val split; each sample returns the next H actions. The dataloader asks sampler for an allocation per sample and renders views accordingly.
- `sample_alloc(cfg, rng) -> Alloc` — Draws r from p(r): uniform over tiers, plus allocator-biased draws (random velocity and clearance pushed through the real allocator), plus random views switched off with p_view_off.

### deadlinevla/budget/deadline.py
- `max_latency(v, d, a, m) -> float` — Returns L_max from Method step 1. Returns infinity when v is near zero; returns a negative number when even instant inference is unsafe, which the caller treats as "must slow down now".

### deadlinevla/budget/governor.py
- `safe_speed(L, d, a, m) -> float` — Returns v* from Method step 2; zero if d is at or below m.
- `govern(v_cmd, L_alloc, d, cfg) -> float[d_a]` — Rescales the policy's commanded linear velocity so its norm is at most min(v*, v_max), keeping its direction and yaw rate.

### deadlinevla/budget/latency_model.py
- `profile_allocations(runtime, allocs, iters) -> LatencyTable` — On the target board, runs the real runtime for each allocation and records p50, p95 and max end-to-end latency. Uses p95 for the deadline.
- `fit_linear(table) -> LinearModel` — Fits L_0, e(r), c from Method step 3 and reports the worst residual. If it is too large, predict falls back to the table.
- `predict(alloc) -> float` — Looks up or computes the p95 latency of an allocation.

### deadlinevla/budget/allocator.py
- `view_weights(v_body, cam_axes, kappa, eps) -> float[V]` — Method step 4. Uniform weights when speed is below a hover threshold.
- `allocate(weights, L_max, table, quality_fn) -> (Alloc, float)` — Exact search over all 4096 allocations, first with the all-views-on floor, then without. Returns the best feasible allocation and its predicted latency, or the cheapest allocation if none is feasible. Tie-break toward lower latency.

### deadlinevla/budget/clearance.py
- `estimate_clearance(obs, v_hat, cfg) -> float` — oracle: reads info.clearance from the simulator. depth: minimum depth in the cone from the depth source. Same interface, so the ablation swaps one config field.

### deadlinevla/policy/base.py
- `class PolicyAdapter` — every model hides behind this.
  - `load(path, device)`: weights plus the model's own config.
  - `forward_loss(batch, alloc) -> (loss, metrics)`: flow-matching loss with views rendered at alloc.
  - `predict_chunk(obs, alloc) -> float[H, d_a]`: PyTorch reference inference.
  - `spec() -> PolicySpec`: H, K, tier resolutions, tokens per tier n(r).

### deadlinevla/policy/views.py
- `render_views(images, alloc, tiers) -> list[Tensor]` — Downsamples each view to its tier, drops views at tier 0. Identical code in training and serving.
- `ViewTierEmbedding` — Learned vectors indexed by (view j, tier r_j), added to that view's visual tokens, so the model knows which direction and resolution each token came from.

### deadlinevla/policy/smolvla.py, gr00t.py
- Adapters over the upstream LeRobot SmolVLA and Isaac-GR00T code. They only change the image loop to accept a variable list of views and inject ViewTierEmbedding; no model code is copied.

### deadlinevla/train/sft.py
- `train_sft(cfg) -> Path` — Fine-tunes with sample_alloc drawing a fresh allocation per sample (Method step 6). Logs loss per tier and per number of active views, so we see whether low budgets are being learned.
- `offline_eval(adapter, val, allocs) -> dict` — Action error against the expert at a fixed grid of allocations: all high, forward-only, uniform low, allocator picks. A cheap check before any rollout.

### deadlinevla/serve/runtime.py
- `class TrtRuntime`
  - `load(engines, device)`: one vision engine per tier, backbone and head engines, buffers allocated once.
  - `infer(obs, alloc) -> float[H, d_a]`: encodes only active views at their tier, concatenates tokens, runs prefill and K head steps.
  - `timings() -> dict`: per-stage times of the last call, for profile_allocations.

### deadlinevla/serve/executor.py
- `class ChunkExecutor` — Async chunk execution: keeps sending actions while the next chunk is computed, splices the new chunk by dropping the actions already in the past. Reports stalls.

### deadlinevla/serve/pilot.py
- `class Pilot`: the per-tick control loop, the heart of the system.
  - `tick(obs, state) -> float[d_a]`: (1) estimate_clearance, (2) max_latency, (3) view_weights then allocate, (4) hand obs and alloc to the executor for inference, (5) take the executor's next action and pass it through govern. Logs every quantity for evaluation.
  - mode: deadline (ours) or a baseline name, so baselines run through the same loop.

### deadlinevla/eval/baselines.py
- `baseline_alloc(name, state, table) -> Alloc` — B1 all_high: every view at tier 3. B2 uniform_fixed: every view at one tier, VLA-ACL style. B3 importance: tiers from attention importance of the last step, FastVLA style, no physics. B4 front_only: front view at tier 3, others off, AerialVLA style.
- Each baseline is run with and without the governor, so the governor's own effect is separable.

### deadlinevla/eval/rollout.py
- `run_episodes(pilot, env, scenes, speed_cap) -> list[EpisodeLog]` — Fixed seeds per scene so every method faces identical scenes. Logs per tick: speed, clearance, allocation, predicted and actual latency, commanded and governed velocity.

### deadlinevla/eval/metrics.py
- `summarize(logs) -> Metrics` — Success rate, collisions per km, time to goal, mean speed, tokens per tick, and the deadline-violation rate: the fraction of ticks where v L_actual + v²/(2a) exceeds d − m.

### deadlinevla/eval/report.py
- `speed_vs_compute(results) -> Figure` — Headline plot: highest speed cap at which each method keeps collisions per km below a threshold, against compute (tokens per tick or board). Also a time-to-goal versus collision-rate Pareto plot.

### deadlinevla/ops/gates.py
- `check_gates(metrics, baseline, cfg) -> GateResult` — CI fails if the violation rate exceeds max_violation_rate or success drops against the stored baseline.

### deadlinevla/cli.py
- `main(argv)` — Commands: env-check, record, train, export, profile, eval, report. Each reads the previous stage's outputs from manifest.json.

## Experiments and milestones

The project succeeds if the deadline-driven pilot flies faster than every baseline at the same collision rate on the same board, and the speed-versus-compute plot shows it.

**Questions the experiments answer**

1. Does deadline-driven allocation raise the safe speed over fixed-budget and importance-based allocation? (ours vs B1 to B4)
2. How much comes from allocation and how much from the governor alone? (every method with and without the governor)
3. Does budget-randomized training matter? (same model fine-tuned with all views at high tier only)
4. Does 360 degree input pay off where it should? (rear_threat and behind-the-start goal_nav, against front_only)
5. How much is lost with estimated clearance instead of oracle? (clearance_source = depth)
6. Sensitivity to kappa and epsilon.

**Milestones, each ending in something runnable**

1. M0 Go / no-go. The three checks in Feasibility.
2. M1 Environment. Drone with six cameras in Isaac Lab, measure_braking, three scene families, expert flying collision-free.
3. M2 Dataset. Recorded demonstrations in LeRobot format; offline replay check.
4. M3 Latency table. SmolVLA tiers exported to TensorRT on the Jetson; profile_allocations over all 4096 allocations; linear-fit residual reported.
5. M4 Budget-aware model. Fine-tuning with random allocations; offline_eval across the allocation grid.
6. M5 Pilot. Full control loop in simulation with the real board's latency injected; first deadline-violation numbers.
7. M6 Baselines and plots. All methods, all families, speed sweep; speed-versus-compute and Pareto plots.
8. M7 Ablations and write-up. Questions 2 to 6; GR00T N1.7 on desktop as the large-model reference; README with one-command reproduction.

## Risks and open questions

- [ ] **Novelty.** Run a dedicated literature sweep (arXiv, CoRL 2026, IROS, RA-L) for deadline- or speed-conditioned token budgets on aerial or multi-camera VLAs before M1.
- [ ] **Hardware-in-the-loop latency.** The simulator runs on a desktop GPU while latency comes from the Jetson table. The pilot must inject the measured latency into the simulated loop, not the desktop's. Decide: injected delay in sim, or the policy physically served from the board.
- [ ] **Linear latency model may not hold.** Tiled encoders and attention cost can be non-linear in tokens. Mitigated by the measured table fallback.
- [ ] **Clearance estimation.** The oracle makes v1 clean but optimistic. The depth ablation shows the real gap.
- [ ] **Base model plumbing.** If SmolVLA's image pipeline fights variable views, fall back to a fixed six-slot input with zeroed tokens for off views.
- [ ] **Sim-to-real.** Out of scope for v1. Flag it clearly in the write-up.
- [ ] **Licenses.** SmolVLA and LeRobot code are open; GR00T N1.7 weights under the NVIDIA Open Model License. Fine for a research repo; state it in the README.

## Sources

- Drone VLA: AerialVLA, AutoFly, Exp2VLA, VLA-AN, LiteVLA-H
- Aerial manipulation: AIR-VLA, AeroManip-VLA
- Panoramic: PanoFuse, PanoVLA, UniTrackPLA, Fly360
- Token budgets: VLA-ACL, MVPruner, FastVLA, Fisheye-VLA
- Simulation: Isaac Lab cameras, Isaac Lab paper, Pegasus Simulator, OmniDrones, multi-sensor FPS report
- Models: Isaac-GR00T
