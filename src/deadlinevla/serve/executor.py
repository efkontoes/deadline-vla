"""Chunk execution on the control clock.

An inference requested at tick t with latency L delivers its chunk at tick t + ceil(L / dt). Until
then the drone keeps flying on what it already has: that is the "flies blind for v L metres" of the
deadline.

async  a new inference starts as soon as the previous one returns; the arriving chunk is spliced in
       by dropping the actions that are already in the past (if the latency exceeds the chunk
       horizon, only the chunk's final action is kept).
sync   inference starts only when the current chunk is used up; the drone holds its last command
       while waiting and then plays the new chunk from its first action.

A stall is a tick with no fresh action to play (the last command is held).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class _Pending:
    ready_tick: int
    request_tick: int
    chunk: np.ndarray
    latency: float


class ChunkExecutor:
    def __init__(self, runtime: Any, dt: float, mode: str = "async", action_dim: int = 4):
        if mode not in ("sync", "async"):
            raise ValueError(f"mode must be sync or async, got {mode!r}")
        self.runtime, self.dt, self.mode, self.action_dim = runtime, dt, mode, action_dim
        self.reset()

    def reset(self) -> None:
        self.tick = 0
        self.chunk: Optional[np.ndarray] = None
        self.k = 0
        self.pending: Optional[_Pending] = None
        self.last_action = np.zeros(self.action_dim)
        self.stalls = 0
        self.last_latency = 0.0

    def _poll(self) -> None:
        """Install a chunk whose inference has finished by the current tick."""
        p = self.pending
        if p is not None and p.ready_tick <= self.tick:
            drop = (self.tick - p.request_tick) if self.mode == "async" else 0
            # if the whole chunk is already in the past, keep flying its final action
            self.chunk = p.chunk[min(drop, len(p.chunk) - 1):]
            self.k = 0
            self.pending = None

    def can_request(self) -> bool:
        self._poll()
        if self.pending is not None:
            return False
        if self.mode == "async":
            return True
        return self.chunk is None or self.k >= len(self.chunk)

    def request(self, obs: Any, alloc: np.ndarray) -> float:
        """Start inference on ``obs`` with ``alloc``; returns the sampled/measured latency."""
        chunk = self.runtime.infer(obs, alloc)
        latency = float(self.runtime.timings()["total"])
        delay = max(1, int(math.ceil(latency / self.dt - 1e-9)))
        self.pending = _Pending(self.tick + delay, self.tick, np.asarray(chunk, dtype=np.float64), latency)
        self.last_latency = latency
        return latency

    def next_action(self) -> tuple[np.ndarray, bool]:
        """Action for the current tick and whether it was a stall. Advances the clock by one tick."""
        self._poll()
        stalled = False
        if self.chunk is not None and self.k < len(self.chunk):
            a = self.chunk[self.k]
            self.k += 1
            self.last_action = a
        else:
            a = self.last_action
            stalled = True
            self.stalls += 1
        self.tick += 1
        return np.array(a, dtype=np.float64), stalled
