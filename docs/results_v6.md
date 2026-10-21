# v6 results: coverage allocation, scored against the pre-registered predictions

Predictions: docs/preregistration.md, amendment 1 (committed in `006411b` before any v6 code or run).
Simulation only: synthetic Jetson latency, surrogate policy, camera-coupled relative governor.

| ID | Prediction | Result | Verdict |
|---|---|---|---|
| P1 | v6 uses fewer tokens than v4 on the v4 families, collisions and success non-inferior (cap 10, n = 150) | tokens 117 vs 240; collisions 0.0% vs 0.0%, paired difference +0.0 pp [+0.0, +0.0] (margin +1); success 100.0% vs 100.0%, difference [+0.0, +0.0] pp (margin −2); Holm p = 0 | **held** |
| P2a | v6's tokens rise with the cap (highspeed) | 123 → 145 → 150 → 149 at caps 6, 9, 12, 15 | **failed** |
| P2b | v6 collides less than v4 at 15 m/s (highspeed, n = 150) | 0.0% vs 0.0%; discordant 0 (v6 only) vs 0 (v4 only); McNemar p = 1, Holm 1 | **failed** |
| P2c | v6 spends more tokens than v4 at 15 m/s | 149 vs 219 | **failed** |
| P3 | v6 not worse than v4 on rear_threat (cap 10, n = 450) | 0.2% vs 0.7%, difference [-1.1, +0.0] pp (margin +1); Holm p = 0 | **held** |

![high-speed results](media/highspeed.png)

## High-speed family, every method (governed)

| method | cap | tokens/tick | collision rate [95% CI] | success | mean speed | time to goal |
|---|---|---|---|---|---|---|
| v6 coverage (ours) | 6 | 123 | 0.0% [0.0%, 2.5%] | 100.0% | 4.96 m/s | 8.3 s |
| v6 coverage (ours) | 9 | 145 | 0.0% [0.0%, 2.5%] | 100.0% | 5.59 m/s | 7.4 s |
| v6 coverage (ours) | 12 | 150 | 0.0% [0.0%, 2.5%] | 100.0% | 5.62 m/s | 7.3 s |
| v6 coverage (ours) | 15 | 149 | 0.0% [0.0%, 2.5%] | 100.0% | 5.60 m/s | 7.4 s |
| v4 allocator | 6 | 217 | 0.0% [0.0%, 2.5%] | 100.0% | 4.85 m/s | 8.5 s |
| v4 allocator | 9 | 219 | 0.0% [0.0%, 2.5%] | 100.0% | 5.44 m/s | 7.7 s |
| v4 allocator | 12 | 219 | 0.0% [0.0%, 2.5%] | 99.3% | 5.37 m/s | 7.6 s |
| v4 allocator | 15 | 219 | 0.0% [0.0%, 2.5%] | 100.0% | 5.45 m/s | 7.7 s |
| B2 uniform | 6 | 384 | 0.0% [0.0%, 2.5%] | 100.0% | 4.74 m/s | 8.7 s |
| B2 uniform | 9 | 384 | 0.0% [0.0%, 2.5%] | 100.0% | 5.24 m/s | 8.0 s |
| B2 uniform | 12 | 384 | 0.0% [0.0%, 2.5%] | 100.0% | 5.31 m/s | 7.9 s |
| B2 uniform | 15 | 384 | 0.0% [0.0%, 2.5%] | 100.0% | 5.33 m/s | 7.8 s |
| B1 all-high | 6 | 1536 | 0.0% [0.0%, 2.5%] | 85.3% | 4.04 m/s | 11.2 s |
| B1 all-high | 9 | 1536 | 0.0% [0.0%, 2.5%] | 77.3% | 4.41 m/s | 10.1 s |
| B1 all-high | 12 | 1536 | 0.0% [0.0%, 2.5%] | 73.3% | 4.43 m/s | 10.5 s |
| B1 all-high | 15 | 1536 | 0.0% [0.0%, 2.5%] | 69.3% | 4.44 m/s | 10.6 s |
| B4 front-only | 6 | 256 | 2.0% [0.7%, 5.7%] | 98.0% | 4.90 m/s | 8.4 s |
| B4 front-only | 9 | 256 | 2.0% [0.7%, 5.7%] | 98.0% | 5.55 m/s | 7.4 s |
| B4 front-only | 12 | 256 | 1.3% [0.4%, 4.7%] | 98.7% | 5.60 m/s | 7.4 s |
| B4 front-only | 15 | 256 | 2.0% [0.7%, 5.7%] | 98.0% | 5.59 m/s | 7.4 s |

## v4 families, cap 10 m/s, static scenes (governed)

| method | tokens/tick | collision rate | success | mean speed |
|---|---|---|---|---|
| v6 coverage (ours) | 117 | 0.0% | 100.0% | 4.27 m/s |
| v4 allocator | 240 | 0.0% | 100.0% | 3.99 m/s |
| B2 uniform | 384 | 0.0% | 100.0% | 3.91 m/s |
| B1 all-high | 1536 | 0.0% | 88.0% | 3.23 m/s |
| B4 front-only | 256 | 30.7% | 69.3% | 4.11 m/s |

## S1 sensitivity: design obstacle size 0.3 m (v6 only, highspeed)

| cap | tokens/tick | collision rate | success | mean speed |
|---|---|---|---|---|
| 6 | 171 | 0.0% | 100.0% | 4.95 m/s |
| 9 | 301 | 0.0% | 100.0% | 5.52 m/s |
| 12 | 304 | 0.0% | 100.0% | 5.56 m/s |
| 15 | 304 | 0.0% | 100.0% | 5.56 m/s |

## Why P2 failed: the high-speed family did not produce high speeds

Even at a 15 m/s cap, the drone flew faster than 9 m/s on only about 5% of ticks (median peak speed per
episode 9.5 m/s, measured on 20 replayed scenes for both allocators). The caps of 12 and 15 m/s therefore
rarely bound, so episode-average tokens could not keep rising (P2a), and neither allocator collided at all
(P2b). P2c failed for the same reason: averaged over mostly slow ticks, v6 spends fewer tokens than v4.
These are failures of the test scenario as pre-registered, and they are reported as failures. The scenario
was **not** re-tuned and re-run: that is the forking path the pre-registration exists to prevent. A faster
scenario needs its own pre-registered amendment before it is run.

## Exploratory (not pre-registered): tokens by actual speed, tick level

![tokens by speed](media/tokens_by_speed.png)

All 150 highspeed scenes at the 15 m/s cap, replayed with the logged code (the replay reproduces the cell).

| actual speed (m/s) | ticks | v6 tokens/tick | v6 front camera at high | v4 tokens/tick |
|---|---|---|---|---|
| 0-2 | 709 | 96 | 0% | 319 |
| 2-4 | 2131 | 96 | 0% | 215 |
| 4-6 | 3178 | 109 | 0% | 210 |
| 6-8 | 3406 | 166 | 0% | 208 |
| 8-10 | 1648 | 267 | 36% | 207 |
| 10-14 | 104 | 474 | 100% | 205 |

Tick by tick, v6's spend follows the physics it was built from: all-low below 4 m/s, front mid from about
6 m/s, front high above 10 m/s. v4's spend is flat in speed, because its weights are a ratio of speeds.
This is the mechanism P2 was meant to test; it was observed, but only on the small share of fast ticks,
and it is exploratory evidence, not a confirmed prediction.
