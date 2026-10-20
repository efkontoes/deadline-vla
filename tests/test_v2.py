"""v2 changes: speed-objective allocator, violation causes, Wilson intervals, parallel sweep."""

import numpy as np
import pytest

from deadlinevla.budget.allocator import (
    allocate,
    allocate_for_speed,
    camera_axes,
    quality_table,
    required_quality,
    view_weights,
)
from deadlinevla.budget.governor import safe_speed
from deadlinevla.budget.latency_model import all_allocs, synthetic_prior
from deadlinevla.config import set_field
from deadlinevla.eval.metrics import CAUSES, summarize, violation_causes, wilson
from deadlinevla.eval.rollout import EpisodeLog, build_eval_scenes

YAWS = (0, 60, 120, 180, 240, 300)
TOKENS = (0, 16, 64, 256)


@pytest.fixture(scope="module")
def model():
    return synthetic_prior("synthetic_orin_nx", TOKENS)


@pytest.mark.parametrize("heading", [0.0, 75.0, 180.0, None])
def test_speed_allocator_is_cheapest_meeting_qmin_with_floor(heading, model):
    q = quality_table("log1p_tokens", TOKENS)
    v = np.zeros(3) if heading is None else 6 * np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading)), 0])
    w = view_weights(v, camera_axes(YAWS), 2, 0.05)
    alloc, lat = allocate_for_speed(w, model, q)
    qmin = required_quality(w, q)
    assert q[alloc] @ w >= qmin - 1e-9
    lats = model.predict_all()
    ok = ((q[all_allocs(6)] @ w) >= qmin - 1e-9) & (all_allocs(6) >= 1).all(axis=1)
    assert (alloc >= 1).all()
    assert lat == pytest.approx(lats[ok].min())


def test_speed_allocator_is_cheaper_than_v1_with_slack(model):
    q = quality_table("log1p_tokens", TOKENS)
    w = view_weights(np.array([4.0, 0, 0]), camera_axes(YAWS), 2, 0.05)
    a2, l2 = allocate_for_speed(w, model, q)
    a1, l1 = allocate(w, 5.0, model, q)  # lots of slack: v1 fills it with tokens
    assert l2 < l1
    assert a2[0] >= 2 and (a2 >= 1).all()   # forward stays sharp, every view at least low
    assert a2[0] >= a2[3]                   # savings come from the rear
    # the latency saved is speed gained, at any clearance
    for D in (2.0, 5.0):
        assert safe_speed(l2, D, 8.4, 0.0) > safe_speed(l1, D, 8.4, 0.0)


def test_wilson():
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi and 0.2 < lo and hi < 0.8
    assert wilson(0, 50)[0] == 0.0 and wilson(0, 50)[1] < 0.1
    assert wilson(0, 0) == (0.0, 1.0)


def _log(**ticks):
    n = len(ticks["speed"])
    base = dict(clearance=np.full(n, 5.0), d_cmd=np.full(n, 5.0), L_src=np.full(n, 0.1),
                L_pred_src=np.full(n, 0.1), L_pred=np.full(n, 0.1), age=np.full(n, 0.1), v_star=np.full(n, 9.0),
                tokens=np.zeros(n), L_act=np.full(n, 0.1), requested=np.ones(n, bool))
    base.update({k: np.asarray(v, dtype=float) for k, v in ticks.items()})
    return EpisodeLog("deadline", True, "goal_nav", 0, 8.0, False, False, False, 1.0, 10.0, 8.0, 0.5, 0, 0.1, base)


def test_violation_causes_are_attributed():
    # tick 1: clearance drops 4 m in one tick -> clearance_jump; tick 2: drone above the cap -> dynamics_lag;
    # tick 3: latency above its prediction -> latency_tail; tick 4: no violation
    d = [5.0, 1.0, 1.0, 1.0, 9.0]
    lg = _log(speed=[4, 4, 4, 4, 4], clearance=d, d_cmd=d, v_star=[9, 9, 2, 9, 9],
              L_src=[0.1, 0.1, 0.1, 0.4, 0.1])
    causes = violation_causes(lg)
    assert list(causes) == ["", "clearance_jump", "dynamics_lag", "latency_tail", ""]
    m = summarize([lg])
    assert set(m.violation_causes) == set(CAUSES)
    assert sum(m.violation_causes.values()) == pytest.approx(1.0)


def test_parallel_sweep_matches_serial(light_cfg):
    from deadlinevla.budget.latency_model import LatencyTable
    from deadlinevla.eval.sweep import run_sweep

    cfg = set_field(light_cfg, "model.policy", "surrogate")
    cfg = set_field(cfg, "sim.render", False)
    cfg = set_field(cfg, "eval.methods", ("deadline", "importance"))
    cfg = set_field(cfg, "eval.governor", (True,))
    cfg = set_field(cfg, "eval.speed_caps", (6.0,))
    m = synthetic_prior("synthetic_orin_nx", cfg.tier_tokens)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = LatencyTable(m.allocs, m.table.p50, m.table.p95, m.table.pmax, meta={"board": "synthetic_orin_nx"}
                            ).save(Path(tmp) / "t.npz")
        scenes = build_eval_scenes(cfg, ["sparse_nav", "goal_nav"], per_family=1)
        serial = run_sweep(cfg, scenes, path, None, workers=1)
        parallel = run_sweep(cfg, scenes, path, None, workers=2)
    for (c1, l1), (c2, l2) in zip(serial, parallel):
        assert c1 == c2
        assert [lg.time for lg in l1] == [lg.time for lg in l2]
    # B3 now follows the surrogate's proximity importance instead of collapsing onto B2
    imp = np.concatenate([lg.ticks["alloc"] for (c, logs) in serial if c[0] == "importance" for lg in logs])
    assert not (imp == 2).all(axis=1).all()
