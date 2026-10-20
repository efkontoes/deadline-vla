"""v5: intercepting threat."""

import math

import numpy as np
import pytest

from deadlinevla.config import set_field
from deadlinevla.sim.drone_env import make_env
from deadlinevla.sim.scenes import Mover, build_scene, intercept_time


def test_intercept_time_solves_the_quadratic():
    delta = np.array([10.0, 0.0])           # drone 10 m ahead of the threat
    v_d = np.array([2.0, 0.0])              # drone flying away at 2 m/s
    t = intercept_time(delta, v_d, 6.0)
    assert t == pytest.approx(10.0 / 4.0)   # closes at 6 - 2 = 4 m/s
    assert np.linalg.norm(delta + v_d * t) == pytest.approx(6.0 * t)
    assert intercept_time(delta, np.array([7.0, 0.0]), 6.0) is None   # drone outruns it
    t_side = intercept_time(np.array([0.0, 8.0]), np.array([3.0, 0.0]), 5.0)
    assert math.hypot(3.0 * t_side, 8.0) == pytest.approx(5.0 * t_side)


def test_hunter_tracks_a_turning_drone_and_makes_one_pass():
    m = Mover(p0=np.array([0.0, 0.0]), vel=np.array([5.0, 0.0]), r=0.3, t_start=0.0, mode="intercept")
    drone, v = np.array([10.0, 5.0]), np.array([0.0, 2.0])
    min_range, t = 1e9, 0.0
    for _ in range(80):
        m.steer(t, drone, v)
        t += 0.1
        drone = drone + v * 0.1
        min_range = min(min_range, float(np.linalg.norm(m.pos(t) - drone)))
    assert min_range < 0.5                  # it reaches the drone
    assert m.locked                         # and then flies on instead of circling


def test_linear_mode_unchanged():
    m = Mover(p0=np.array([1.0, 2.0]), vel=np.array([3.0, 0.0]), r=0.5, t_start=0.5)
    m.steer(1.0, np.zeros(3), np.zeros(3))
    assert m.pos(0.2).tolist() == [1.0, 2.0] and m.pos(1.5).tolist() == [4.0, 2.0]


def test_env_resets_the_hunter_between_episodes(light_cfg):
    cfg = set_field(set_field(light_cfg, "scenes.threat_mode", "intercept"), "scenes.threat_speed", 6.0)
    scene = build_scene("rear_threat", 1, cfg)
    env = make_env(cfg)
    env.reset(scene)
    for _ in range(20):
        env.step(np.zeros(4))
    moved = scene.mover.pos(env.t).copy()
    env.reset(scene)
    assert not scene.mover.locked and scene.mover.catchable_ticks == 0
    assert np.linalg.norm(moved - scene.mover.p0) > 1.0
    assert np.linalg.norm(scene.mover.velocity(1.0)) == pytest.approx(6.0)


def test_review_worked_example_for_the_derived_floor(light_cfg):
    """Rear camera, drone still, o_max 5, tau 0.3, a 8.4, m 0.5: d_req = 3.49 m."""
    from deadlinevla.budget.allocator import camera_axes, coverage_mask
    from deadlinevla.perception import detection_range

    axes = camera_axes((0, 60, 120, 180, 240, 300))
    allocs = np.array([[1] * 6, [2] * 6])
    lat = np.array([0.2, 0.2])                                   # + extra 0.1 -> tau = 0.3
    for size, ok in ((1.0, [True, True]), (0.3, [False, True])):
        det = detection_range(np.arange(4), np.full(4, size), light_cfg)
        got = coverage_mask(allocs, lat, np.zeros(3), axes, det, 5.0, 8.4, 0.5, 0.1)
        assert got.tolist() == ok        # 1.0 m: low sees 8 m; 0.3 m: low 2.4 m < 3.49 m, mid 4.8 m


def test_derived_floor_raises_rear_tier_only_when_needed(light_cfg):
    from deadlinevla.budget.allocator import allocate_for_speed, camera_axes, coverage_mask, quality_table
    from deadlinevla.budget.latency_model import synthetic_prior
    from deadlinevla.perception import detection_range

    model = synthetic_prior("synthetic_orin_nx", (0, 16, 64, 256))
    q = quality_table("log1p_tokens", (0, 16, 64, 256))
    axes = camera_axes((0, 60, 120, 180, 240, 300))
    v = np.array([3.0, 0.0, 0.0])
    w = 0.05 + np.maximum(0, axes[:, 0]) ** 2
    fixed, _ = allocate_for_speed(w, model, q)
    for size, expect_rear in ((2.0, 1), (0.3, 2)):
        det = detection_range(np.arange(4), np.full(4, size), light_cfg)
        mask = coverage_mask(model.allocs, model.predict_all(), v, axes, det, 5.0, 8.4, 0.5, 0.19)
        derived, _ = allocate_for_speed(w, model, q, floor=mask)
        assert derived[3] >= expect_rear
    assert fixed[3] == 1
