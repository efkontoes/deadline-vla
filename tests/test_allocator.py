import itertools

import numpy as np
import pytest

from deadlinevla.budget.allocator import allocate, camera_axes, quality_table, view_weights
from deadlinevla.budget.latency_model import synthetic_prior, tokens_of

YAWS = (0, 60, 120, 180, 240, 300)
TOKENS = (0, 16, 64, 256)


def test_doc_example_weights():
    w = view_weights(np.array([5.0, 0.0, 0.0]), camera_axes(YAWS), kappa=2, eps=0.05)
    assert w == pytest.approx([1.05, 0.30, 0.05, 0.05, 0.05, 0.30])


def test_hover_gives_uniform_weights():
    w = view_weights(np.array([0.05, 0.0, 0.0]), camera_axes(YAWS), 2, 0.05, hover_speed=0.3)
    assert np.allclose(w, w[0])


def _brute_force(weights, L_max, model, q):
    best, best_key = None, None
    lat = model.predict_all()
    for floor in (True, False):
        for i, a in enumerate(model.allocs):
            if lat[i] > L_max or (floor and (a < 1).any()):
                continue
            key = (round(float(q[a] @ weights), 9), -lat[i])
            if best_key is None or key > best_key:
                best, best_key = i, key
        if best is not None:
            return best
    return int(np.argmin(lat))


@pytest.mark.parametrize("L_max", [0.03, 0.08, 0.12, 0.2, 0.35, 10.0, -1.0])
@pytest.mark.parametrize("heading", [0.0, 90.0, 200.0])
def test_allocate_matches_brute_force(L_max, heading):
    model = synthetic_prior("synthetic_orin_nx", TOKENS)
    q = quality_table("log1p_tokens", TOKENS)
    h = np.radians(heading)
    w = view_weights(np.array([np.cos(h), np.sin(h), 0.0]) * 4, camera_axes(YAWS), 2, 0.05)
    alloc, lat = allocate(w, L_max, model, q)
    i = _brute_force(w, L_max, model, q)
    assert lat == pytest.approx(model.predict_all()[i])
    assert q[alloc] @ w == pytest.approx(q[model.allocs[i]] @ w)


def test_allocator_spends_budget_forward():
    model = synthetic_prior("synthetic_orin_nx", TOKENS)
    q = quality_table("log1p_tokens", TOKENS)
    w = view_weights(np.array([6.0, 0.0, 0.0]), camera_axes(YAWS), 2, 0.05)
    alloc, lat = allocate(w, 0.2, model, q)
    assert lat <= 0.2
    assert alloc[0] == alloc.max()           # front view gets the most
    assert (alloc >= 1).all()                # floor holds when affordable
    assert alloc[0] >= alloc[3]


def test_infeasible_returns_cheapest():
    model = synthetic_prior("synthetic_orin_nx", TOKENS)
    q = quality_table("log1p_tokens", TOKENS)
    alloc, lat = allocate(np.ones(6), -0.5, model, q)
    assert lat == pytest.approx(model.predict_all().min())
    assert tokens_of(alloc, TOKENS) == 0


def test_allocs_enumeration_order():
    model = synthetic_prior("synthetic_orin_nx", TOKENS)
    assert len(model.allocs) == 4 ** 6
    assert list(model.allocs[1]) == [0, 0, 0, 0, 0, 1]
    assert [tuple(a) for a in model.allocs[:5]] == list(itertools.islice(itertools.product(range(4), repeat=6), 5))
