import math

import numpy as np
import pytest

from deadlinevla.budget.deadline import max_latency, violates
from deadlinevla.budget.governor import cap_speed, govern, safe_speed
from deadlinevla.config import load_config, set_field


def test_doc_example_deadline():
    # v = 8, a = 8, d = 5, m = 0: braking 4 m, L_max = 1/8 s
    assert max_latency(8, 5, 8, 0) == pytest.approx(0.125)
    # v = 4 in the same scene allows 1 s
    assert max_latency(4, 5, 8, 0) == pytest.approx(1.0)


def test_deadline_edge_cases():
    assert max_latency(0.0, 5, 8, 0.5) == math.inf
    assert max_latency(10, 5, 8, 0) < 0  # braking alone needs 6.25 m > 5 m


def test_doc_example_safe_speed():
    # a = 8, L = 0.125, d - m = 5: -1 + sqrt(1 + 80) = 8
    assert safe_speed(0.125, 5, 8, 0) == pytest.approx(8.0)
    assert safe_speed(0.1, 0.5, 8, 0.5) == 0.0
    assert safe_speed(math.inf, 5, 8, 0) == 0.0


@pytest.mark.parametrize("v,d", [(2.0, 3.0), (6.0, 9.0), (9.0, 15.0)])
def test_governor_inverts_deadline(v, d):
    a, m = 7.5, 0.5
    L = max_latency(v, d, a, m)
    assert L > 0
    assert safe_speed(L, d, a, m) == pytest.approx(v)
    assert not violates(v, L * 0.99, d, a, m)
    assert violates(v, L * 1.01, d, a, m)


def test_govern_keeps_direction_and_yaw():
    cfg = set_field(load_config(None), "drone.a_brake", 8.0)
    cfg = set_field(cfg, "drone.margin", 0.0)
    out = govern(np.array([10.0, 0.0, 0.0, 0.7]), 0.125, 5.0, cfg)
    assert out[0] == pytest.approx(8.0)  # capped at v* = 8 (= v_max too)
    assert out[3] == 0.7
    slow = govern(np.array([3.0, 4.0, 0.0, 0.0]), 0.125, 5.0, cfg)
    assert np.allclose(slow, [3.0, 4.0, 0.0, 0.0])  # below the cap: untouched
    assert np.allclose(cap_speed(np.array([3.0, 4.0, 0.0, 1.0]), 2.5), [1.5, 2.0, 0.0, 1.0])
