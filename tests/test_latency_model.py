import numpy as np
import pytest

from deadlinevla.budget.latency_model import (
    LatencyModel,
    LatencyTable,
    SyntheticBoard,
    TableBoard,
    all_allocs,
    alloc_index,
    fit_linear,
    profile_allocations,
)

TOKENS = (0, 16, 64, 256)


class _BoardRuntime:
    def __init__(self, src, seed=0):
        self.src, self.rng, self._t = src, np.random.default_rng(seed), {}

    def infer(self, obs, alloc):
        self._t = self.src.sample(alloc, self.rng)
        return None

    def timings(self):
        return self._t


def test_alloc_index_roundtrip():
    allocs = all_allocs(6)
    for i in (0, 1, 77, 4095):
        assert alloc_index(allocs[i]) == i


def test_exact_linear_board_fits_perfectly():
    allocs = all_allocs(6)
    counts = np.stack([(allocs == r).sum(1) for r in (1, 2, 3)], 1)
    y = 0.02 + counts @ np.array([0.004, 0.01, 0.03])
    table = LatencyTable(allocs, y, y, y)
    fit = fit_linear(table)
    assert fit.worst_residual < 1e-9
    assert fit.per_view == pytest.approx([0.004, 0.01, 0.03])
    model = LatencyModel(table, max_residual=1e-6)
    assert not model.use_table
    assert model.predict([3, 0, 0, 0, 0, 1]) == pytest.approx(0.02 + 0.03 + 0.004)


def test_profile_synthetic_board_and_fallback(tmp_path):
    src = SyntheticBoard("synthetic_orin_nx", TOKENS)
    allocs = all_allocs(6)[::37]
    table = profile_allocations(_BoardRuntime(src), allocs, iters=15, meta={"board": src.name})
    assert (table.p95 >= table.p50).all() and (table.pmax >= table.p95).all()
    fit = fit_linear(table, tier_tokens=TOKENS)
    assert fit.e is not None and fit.c > 0
    # quadratic attention cost: a strict residual threshold forces the table fallback
    model = LatencyModel(table, max_residual=1e-4, tier_tokens=TOKENS)
    assert model.use_table
    assert model.predict(allocs[5]) == pytest.approx(table.p95[5])
    path = table.save(tmp_path / "t.npz")
    back = LatencyTable.load(path)
    assert np.allclose(back.p95, table.p95) and back.meta["board"] == src.name
    assert set(back.stages_p50) == set(table.stages_p50)


def test_table_board_respects_max():
    allocs = all_allocs(6)
    p = np.full(len(allocs), 0.1)
    tb = TableBoard(LatencyTable(allocs, p, p * 1.2, p * 1.3))
    rng = np.random.default_rng(0)
    samples = [tb.sample(allocs[9], rng)["total"] for _ in range(500)]
    assert max(samples) <= 0.13 + 1e-12
    assert np.median(samples) == pytest.approx(0.1, rel=0.05)
