# ruff: noqa: E501  (report-template strings)
"""Diagnose the v6 high-speed run from its logs (no new method runs): what held the speed down, and does the
front camera chatter between tiers?

    python scripts/diagnose_v6.py     # writes docs/diagnosis_v6.md and docs/diagnosis_v6.json

Every replayed cell is first checked against runs/light_highspeed/episodes.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deadlinevla.config import load_config, set_field  # noqa: E402
from deadlinevla.eval.rollout import build_eval_scenes  # noqa: E402
from deadlinevla.viz.replay import check_against_log, replay_cell  # noqa: E402

RUN = ROOT / "runs" / "light_highspeed"
CAP = 15.0
TOL = 0.05
NAMES = {"deadline": "v6 coverage", "deadline_v4": "v4 allocator"}


def binding(tr, cap):
    """Per tick: which limit set the commanded/flown speed."""
    dist_goal = np.linalg.norm(tr.pos[:, :2] - tr.scene.goal_xy[None], axis=1)
    goal_term = np.sqrt(2 * 4.0 * np.maximum(dist_goal - 0.3, 0.0)) + 0.3      # surrogate's goal approach
    lab = np.empty(len(tr.speed), dtype=object)
    for k in range(len(tr.speed)):
        vc, vg = tr.v_cmd[k], tr.v_gov[k]
        if vc < 0.3 and k < 8:
            lab[k] = "waiting for the first chunk"
        elif vg < vc - TOL:
            lab[k] = "governor"
        elif vc >= cap - TOL:
            lab[k] = "speed cap (policy at cruise)"
        elif abs(vc - goal_term[k]) < TOL:
            lab[k] = "policy: slowing for the goal"
        else:
            lab[k] = "policy: obstacle detected ahead"
    lag = tr.speed < tr.v_gov - 0.5                                              # still accelerating toward it
    return lab, lag


def chatter(tr, lo=8.0, hi=10.0, dt=0.1):
    f = tr.alloc[:, 0].astype(int)
    sw = np.flatnonzero(f[1:] != f[:-1]) + 1
    band = (tr.speed >= lo) & (tr.speed < hi)
    sw_band = [k for k in sw if band[k]]
    # flicker: a switch reversed within 3 ticks (up then down, or down then up)
    flick = sum(1 for a, b in zip(sw, sw[1:]) if b - a <= 3 and f[b] == f[a - 1])
    return len(sw), len(sw_band), int(band.sum()) * dt, flick, len(f) * dt


def main() -> int:
    cfg = set_field(load_config(RUN / "config.resolved.yaml"), "sim.render", False)
    scenes = build_eval_scenes(cfg)
    eps = [json.loads(x) for x in (RUN / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if x]
    out, md = {}, ["# v6 diagnosis: what held the speed down, and does the front camera chatter?", "",
                   f"Highspeed family, {CAP:g} m/s cap, all {len(scenes)} scenes, governed. Every replayed cell reproduces "
                   "the logged episodes exactly. Diagnosis from existing runs only: no method is re-run or re-tuned.", ""]
    for m in ("deadline", "deadline_v4"):
        traces = replay_cell(cfg, scenes, m, CAP, RUN / "latency_table.npz")
        check_against_log(traces, eps, CAP)
        labs = np.concatenate([binding(t, CAP)[0] for t in traces])
        lags = np.concatenate([binding(t, CAP)[1] for t in traces])
        speeds = np.concatenate([t.speed for t in traces])
        cats = ["speed cap (policy at cruise)", "policy: obstacle detected ahead", "policy: slowing for the goal",
                "governor", "waiting for the first chunk"]
        share = {c: float(np.mean(labs == c)) for c in cats}
        lag_share = float(np.mean(lags))
        sw = np.array([chatter(t) for t in traces], dtype=float)
        time_total, time_band = sw[:, 4].sum(), sw[:, 2].sum()
        res = {"share_binding": share, "share_accelerating": lag_share,
               "share_ticks_above_9": float(np.mean(speeds > 9)),
               "front_switches_per_s": float(sw[:, 0].sum() / time_total),
               "front_switches_per_s_8_10": float(sw[:, 1].sum() / max(time_band, 1e-9)),
               "time_in_8_10_s": float(time_band), "flickers": int(sw[:, 3].sum()), "switches": int(sw[:, 0].sum())}
        # what limited the speed on the fast-capable ticks: policy commands below 9 m/s while not near the goal?
        out[m] = res
        md += [f"## {NAMES[m]}", "", "| what limited the speed | share of ticks |", "|---|---|"]
        md += [f"| {c} | {100 * share[c]:.1f}% |" for c in cats]
        md += ["", f"- Ticks where the drone is still accelerating toward its governed command (> 0.5 m/s below it): "
               f"{100 * lag_share:.1f}%.",
               f"- Ticks above 9 m/s: {100 * res['share_ticks_above_9']:.1f}%.",
               f"- Front-camera tier switches: {res['front_switches_per_s']:.2f} per second overall, "
               f"{res['front_switches_per_s_8_10']:.2f} per second while flying 8–10 m/s "
               f"({res['time_in_8_10_s']:.0f} s spent in that band); {res['flickers']} of {res['switches']} switches "
               "were reversed within 3 ticks (flicker).", ""]
    (ROOT / "docs" / "diagnosis_v6.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (ROOT / "docs" / "diagnosis_v6.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
