"""v3 relative deadline: special cases, weights, escape governor."""

import math

import numpy as np
import pytest

from deadlinevla.budget.allocator import camera_axes, view_weights
from deadlinevla.budget.deadline import max_latency
from deadlinevla.budget.governor import safe_speed
from deadlinevla.budget.relative import (
    Contacts,
    camera_closing,
    closing_weights,
    govern_relative,
    min_slack,
    relative_deadline,
)

AXES = camera_axes((0, 60, 120, 180, 240, 300))


def one(r, d, o=0.0, R=0.5):
    """One contact in direction r at surface distance d, moving along r_hat at speed o."""
    r = np.asarray(r, dtype=float)
    r = r / np.linalg.norm(r)
    return Contacts(p=(r * (d + R))[None], R=np.array([R]), w=(r * o)[None])


def test_static_obstacle_ahead_is_the_original_formula():
    v = np.array([8.0, 0.0, 0.0])
    assert relative_deadline(one([1, 0], 5.0), v, 8.0, 0.0) == pytest.approx(max_latency(8, 5, 8, 0)) == 0.125


def test_threat_from_behind_tightens_deadline_of_a_hovering_drone():
    hover = np.zeros(3)
    assert relative_deadline(one([1, 0], 5.0), hover, 8.0, 0.5) == math.inf          # static: no deadline
    # an obstacle 4 m behind, closing at 4 m/s (o = -4: moving toward the drone along r_hat = -x)
    L = relative_deadline(one([-1, 0], 4.0, o=-4.0), hover, 8.0, 0.5)
    assert L == pytest.approx((4.0 - 0.5 - 16 / 16) / 4.0)
    assert min_slack(one([-1, 0], 4.0, o=-4.0), hover, L + 0.1, 8.0, 0.5) < 0


def test_closing_weights_match_original_in_a_static_world():
    v = np.array([5.0, 0.0, 0.0])
    w_old = view_weights(v, AXES, 2, 0.05)
    empty = Contacts(np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2)))
    w_new = closing_weights(camera_closing(empty, v, AXES), 2, 0.05)
    assert w_new == pytest.approx(w_old)


def test_rear_camera_weight_rises_before_threat_is_close():
    v = np.array([3.0, 0.0, 0.0])
    far_threat = one([-1, 0], 12.0, o=-9.0)            # 12 m behind, moving at 9 m/s: closes at 9 - 3 = 6 m/s
    w = closing_weights(camera_closing(far_threat, v, AXES), 2, 0.05)
    assert w[3] == pytest.approx(w.max())             # rear camera (180 deg) now dominates
    assert w[3] > w[0]


def test_governor_slows_for_static_obstacle_ahead():
    out = govern_relative(np.array([6.0, 0.0, 0.0, 0.3]), 0.2, one([1, 0], 2.0), 8.0, 0.5, 10.0)
    assert out[0] == pytest.approx(safe_speed(0.2, 2.0, 8.0, 0.5)) and out[1] == pytest.approx(0) and out[3] == 0.3


def test_governor_escapes_threat_from_behind_instead_of_braking():
    # hovering, threat 2 m behind closing at 5 m/s: braking cannot help, the command must flee forward
    threat = one([-1, 0], 2.0, o=-5.0)
    out = govern_relative(np.zeros(4), 0.2, threat, 8.0, 0.5, 10.0)
    assert out[0] > 0.5
    closing_after = threat.closing(np.array([out[0], out[1], 0.0]))[0]
    assert closing_after <= safe_speed(0.2, 2.0, 8.0, 0.5) + 1e-9


def test_inside_margin_drone_can_slide_but_not_approach():
    wall = one([1, 0], 0.2)  # inside the 0.5 m margin, straight ahead
    out = govern_relative(np.array([3.0, 3.0, 0.0, 0.0]), 0.2, wall, 8.0, 0.5, 10.0)
    assert out[0] <= 1e-9 and out[1] == pytest.approx(3.0)


def test_obstacle_flown_past_is_not_a_threat():
    # obstacle 3 m to the side and 3 m ahead (outside the 30 deg cone): clearance shrinks, but the drone misses it
    side = Contacts(p=np.array([[3.0, 3.0]]), R=np.array([0.5]), w=np.zeros((1, 2)))
    v = np.array([6.0, 0.0, 0.0])
    assert side.closing(v)[0] > 0
    assert relative_deadline(side, v, 8.0, 0.5) == math.inf
    assert govern_relative(np.array([6.0, 0, 0, 0]), 0.3, side, 8.0, 0.5, 10.0)[0] == pytest.approx(6.0)
    # aim at it and it becomes a threat
    v_at = np.array([3.0, 3.0, 0.0]) * 2
    assert relative_deadline(side, v_at, 8.0, 0.5) < math.inf
