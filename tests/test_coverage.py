"""v6 coverage allocation: requirements in metres, per direction."""

import numpy as np
import pytest

from deadlinevla.budget.allocator import CoverageAllocator
from deadlinevla.budget.latency_model import synthetic_prior

TOKENS = (0, 16, 64, 256)
YAWS = (0, 60, 120, 180, 240, 300)
DET = np.array([0.0, 4.0, 8.0, 16.0])          # s = 0.5 m, p_min = 8, W = 128/256/512, FOV 90


@pytest.fixture(scope="module")
def ca():
    return CoverageAllocator(synthetic_prior("synthetic_orin_nx", TOKENS), YAWS, 90.0, DET, TOKENS)


def tokens(a):
    return int(np.asarray(TOKENS)[a].sum())


def test_allocation_climbs_with_speed(ca):
    got = [tokens(ca.allocate(np.array([v, 0.0, 0.0]), 8.4, 0.5, 0.19)[0]) for v in (0, 3, 6, 9, 12)]
    assert got[0] == got[1] == 96                     # slow: all low
    assert got == sorted(got) and got[-1] > got[1]    # monotone in speed
    fast, _ = ca.allocate(np.array([12.0, 0.0, 0.0]), 8.4, 0.5, 0.19)
    assert fast[0] == 3 and fast[3] == 1              # front high, rear stays low


def test_requirement_is_met_in_every_direction(ca):
    v = np.array([7.0, 2.0, 0.0])
    alloc, lat = ca.allocate(v, 8.4, 0.5, 0.19)
    th = np.radians(np.arange(0, 360, 5))
    c = np.maximum(0, np.cos(th) * v[0] + np.sin(th) * v[1])
    req = np.minimum(0.5 + (lat + 0.19) * c + c ** 2 / (2 * 8.4), DET[3])
    off = np.abs((th[:, None] - np.radians(YAWS)[None] + np.pi) % (2 * np.pi) - np.pi)
    reach = np.where(off <= np.radians(45) + 1e-9, DET[alloc][None], 0).max(1)
    assert (reach >= req - 1e-9).all()


def test_detected_threat_from_behind_raises_the_rear(ca):
    calm, _ = ca.allocate(np.array([3.0, 0.0, 0.0]), 8.4, 0.5, 0.19)
    threat, _ = ca.allocate(np.array([3.0, 0.0, 0.0]), 8.4, 0.5, 0.19, [np.pi], [8.0])
    assert calm[3] == 1 and threat[3] >= 2


def test_never_blind_when_nothing_suffices(ca):
    alloc, _ = ca.allocate(np.array([25.0, 0.0, 0.0]), 8.4, 0.5, 0.19)   # needs > 16 m ahead
    assert alloc[0] == 3 and (alloc >= 1).all()
