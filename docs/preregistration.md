# Pre-registration (DRAFT)

**Status: draft.** The numbers marked *(proposed)* are suggestions, not decisions. Edit them, then sign
below (date + commit) **before** any confirmatory run. After signing, any change goes in the amendments
table with a reason; nothing is silently re-tuned.

Why this exists: in v1 to v5 the threat or governor model was changed after each result. Every change was
defensible and is documented, but doing it again to make a method win is how a simulator gets tuned until
it agrees. From here on, models and success criteria are fixed first and results are reported whatever
they are.

## 1. Frozen laptop simulator

Frozen at commit `a2754eb` (v5). Frozen means no changes to the code paths below, except for bugs, which
must be logged as amendments with a test that shows the bug:

- `sim/` (scenes, light env, expert), `policy/surrogate.py`, `perception.py`
- `budget/` (deadline, governor, relative, allocator, latency model), `serve/pilot.py`, `serve/executor.py`
- `eval/` metric definitions; `configs/light_surrogate.yaml`

Laptop results are final for that model: the v4 headline (36% fewer tokens than B2, equal success and
safety, camera-coupled governor) and the v5 negative result (derived floor never beat the fixed floor).

## 2. Confirmatory claims (on the real stack, M5/M6)

Setting: Isaac Lab environment (M1), fine-tuned budget-aware SmolVLA (M4), latency replayed from the
measured Jetson table (M3), camera-coupled relative governor, fixed floor, `budget.objective: speed`.
Scenes: goal_nav, sparse_nav, slalom (static) and rear_threat (linear threat, 5 m/s, 1.0 m), evaluation
seeds from `eval.seed_offset = 10000`, every method on identical scenes, governed, top speed cap.

| ID | Claim | Test | Success criterion |
|---|---|---|---|
| H1 | Efficiency: the deadline allocator matches B2 uniform on safety and success with fewer tokens | paired, ≥ 300 static scenes *(proposed)* | tokens/tick ≥ 20% *(proposed)* below B2 **and** success non-inferior (lower 95% bound of the paired difference ≥ −2 pp *(proposed)*) **and** collisions non-inferior (upper bound ≤ +1 pp *(proposed)*) |
| H2 | 360° coverage matters when safety depends on the cameras | paired McNemar, ≥ 150 static scenes | front-only collides more than ours, p < 0.05 |
| H3 | The governor carries most of the safety | each method with and without governor, paired | collision rate lower with the governor for every method, p < 0.05 |
| H4 | Real tokens buy perception the way the surrogate assumes | offline eval of the M4 model at the allocator's picks vs all-high | action error at allocator picks within 10% *(proposed)* of all-high |

H4 is the direct test of the surrogate assumption. If H4 fails, H1 is reported but flagged as resting on
an assumption the real model does not meet.

Statistics: paired bootstrap (10,000 resamples) for differences, exact McNemar for paired binary
outcomes, Holm correction across H1 to H4. Sample sizes are fixed in advance: no optional stopping.

## 3. Exploratory: the derived floor (later chapter only)

The v5 result showed that earlier detection only helps if something can turn extra time into a miss.
If this is revisited, the models are fixed **now**:

- Threat: a pursuer with a **finite turn rate** of 90°/s *(proposed)*, one pass, speeds {3, 6, 9} m/s,
  sizes {0.3, 0.6, 1.0} m; the 450/150 paired-scene design from v5.
- Escape: a lateral-dodge governor (velocity-obstacle style: the velocity that maximises miss distance),
  implemented and unit-tested on development seeds (< 10000) **before** any grid run.
- Success: the derived floor (matched envelope) has a collision rate at least 5 pp *(proposed)* lower than
  the fixed floor in at least two of the three small-fast cells, Holm-corrected McNemar α = 0.05.
  Otherwise the v5 negative result stands and is reported as such.

## 4. Rules

- Development uses seeds below 10000; confirmatory runs use seeds from 10000 up, once.
- Every confirmatory run is reported, including failures and crashes (with the crash reason).
- Latency results are labelled with the board they came from; synthetic numbers are never presented as
  board numbers.

## 5. Amendment 1 (2026-10-21): coverage allocation (v6), predictions written before running

The allocator is changed: the frozen code paths `budget/allocator.py`, `serve/pilot.py` and `sim/scenes.py`
get additions only, and the v4 code paths keep their behaviour, which the v4 replays check against the
logged episodes. Spec: design.md, "4–5 (v6)". This section was committed before any simulation run of v6.

Setting: light simulator, surrogate policy, camera-coupled relative governor, synthetic Jetson latency,
measured $a$ and $\tau_{\text{track}}$; design obstacle size $s = 0.5$ m, $p_{\min} = 8$. Every method on
identical scenes, governed. Comparison: `deadline_v4` (speed objective, fixed floor: the v4 allocator).

| ID | Prediction | Data | Test and criterion |
|---|---|---|---|
| P1 | On the v4 families, v6 uses fewer tokens than v4 (predicted 96–160 per tick vs 240) | 150 static scenes, cap 10 m/s | tokens below v4 **and** collisions non-inferior (upper 95% bound of the paired difference ≤ +1 pp) **and** success non-inferior (lower bound ≥ −2 pp) |
| P2a | In `highspeed`, v6's tokens per tick rise with the speed cap | 150 scenes per cap, caps 6/9/12/15 | mean tokens strictly increasing across caps |
| P2b | In `highspeed` at 15 m/s, v6 collides less than v4 | same, cap 15 | paired McNemar, two-sided, p < 0.05, with v6's rate lower |
| P2c | In `highspeed` at 15 m/s, v6 spends more tokens than v4 | same | mean tokens above v4's |
| P3 | On `rear_threat`, v6 is not worse than v4 (no improvement is predicted: unseen threats cannot raise the rear) | 450 scenes, cap 10 | collisions non-inferior (upper bound ≤ +1 pp) |
| S1 | Sensitivity, no hypothesis: design size s = 0.3 m | `highspeed`, v6 only | reported |

Holm correction across P1, P2b and P3 (the tests with p-values). All results are reported whatever they are.

## Sign-off

| Date | Commit | Signed by | Notes |
|---|---|---|---|
| | | | |

## Amendments

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-10-21 | 5 | Coverage allocation (v6) added as `budget.objective: coverage`; `highspeed` scene family; `deadline_v4` comparison mode | The weights' speed ratio is scale-free, so the v4 allocation could not depend on speed (found while building the explainer animation) |
