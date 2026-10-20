import numpy as np
import pytest

from deadlinevla.serve.executor import ChunkExecutor


class FixedRuntime:
    """Returns chunk [n, n+1, ...] (first component) for the n-th request, with a fixed latency."""

    def __init__(self, latency, H=5):
        self.latency, self.H, self.n = latency, H, 0

    def infer(self, obs, alloc):
        self.n += 1
        c = np.zeros((self.H, 4))
        c[:, 0] = 100 * self.n + np.arange(self.H)
        return c

    def timings(self):
        return {"total": self.latency}


def _run(ex, ticks):
    out = []
    for _ in range(ticks):
        if ex.can_request():
            ex.request(None, np.zeros(6, int))
        out.append(ex.next_action())
    return out


def test_async_splices_dropping_past_actions():
    ex = ChunkExecutor(FixedRuntime(0.25), dt=0.1, mode="async")
    acts = _run(ex, 8)
    # request at tick 0 arrives at tick 3 (ceil 2.5); actions 0..2 are in the past -> start at index 3
    assert [a[1] for a in acts[:3]] == [True, True, True]
    assert acts[3][0][0] == 103 and acts[4][0][0] == 104
    # second request made at tick 3 arrives at tick 6 -> index 3 of chunk 2
    assert acts[6][0][0] == 203


def test_latency_beyond_horizon_keeps_final_action():
    ex = ChunkExecutor(FixedRuntime(0.9), dt=0.1, mode="async")
    acts = _run(ex, 11)
    assert acts[9][0][0] == 104 and not acts[9][1]


def test_sync_waits_then_plays_from_start():
    ex = ChunkExecutor(FixedRuntime(0.25), dt=0.1, mode="sync")
    acts = _run(ex, 12)
    assert [a[0][0] for a in acts[3:8]] == [100, 101, 102, 103, 104]
    assert acts[8][1]  # stall while the next chunk is computed, holding the last command
    assert acts[8][0][0] == 104


def test_bad_mode():
    with pytest.raises(ValueError):
        ChunkExecutor(FixedRuntime(0.1), 0.1, "sometimes")
