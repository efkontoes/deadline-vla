# v6 diagnosis: what held the speed down, and does the front camera chatter?

Highspeed family, 15 m/s cap, all 150 scenes, governed. Every replayed cell reproduces the logged episodes exactly. Diagnosis from existing runs only: no method is re-run or re-tuned.

## v6 coverage

| what limited the speed | share of ticks |
|---|---|
| speed cap (policy at cruise) | 11.7% |
| policy: obstacle detected ahead | 24.9% |
| policy: slowing for the goal | 0.2% |
| governor | 61.7% |
| waiting for the first chunk | 1.5% |

- Ticks where the drone is still accelerating toward its governed command (> 0.5 m/s below it): 39.3%.
- Ticks above 9 m/s: 4.9%.
- Front-camera tier switches: 1.23 per second overall, 1.94 per second while flying 8–10 m/s (165 s spent in that band); 141 of 1375 switches were reversed within 3 ticks (flicker).

## v4 allocator

| what limited the speed | share of ticks |
|---|---|
| speed cap (policy at cruise) | 5.2% |
| policy: obstacle detected ahead | 22.3% |
| policy: slowing for the goal | 0.2% |
| governor | 69.7% |
| waiting for the first chunk | 2.6% |

- Ticks where the drone is still accelerating toward its governed command (> 0.5 m/s below it): 39.3%.
- Ticks above 9 m/s: 5.4%.
- Front-camera tier switches: 0.06 per second overall, 0.00 per second while flying 8–10 m/s (171 s spent in that band); 5 of 64 switches were reversed within 3 ticks (flicker).

## Conclusions

1. **The governor held the speed down, not the policy.** It bound on 62% (v6) and 70% (v4) of ticks: even
   sparse clutter puts a detected obstacle inside the 30° cone ahead often enough to cap the safe speed near
   9–11 m/s for a blind time of ~0.3 s. The policy's own obstacle term bound on ~25%, the goal approach on
   0.2%, the cap on 5–12%. (In these runs the policy is the surrogate, which sets its own speed; it does not
   imitate the expert's demonstrations.) The drone was still accelerating on 39% of ticks.
2. **So a faster scene must keep the cone ahead clear for long stretches**, not just lower the clutter
   density: obstacles near the direct line should be rare and spaced so the drone can reach the cap between
   them.
3. **v6 chatters.** The front camera changes tier 1.9 times per second while flying 8–10 m/s (v4: 0), and
   10% of switches reverse within 3 ticks. That is the loop speed → tier → latency → blind time → governor →
   speed. Hysteresis is the standard remedy, and it must be pre-registered as a variant before it is tested.

