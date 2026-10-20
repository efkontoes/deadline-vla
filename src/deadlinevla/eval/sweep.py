"""The evaluation sweep: methods x governor x speed caps over a fixed scene set, optionally in parallel.

Every cell sees the same scenes, and cells with the same speed cap share the latency RNG stream
(common random numbers), so differences between methods are not sampling noise in the latency.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from deadlinevla.config import Config
from deadlinevla.eval.rollout import EpisodeLog, run_episodes
from deadlinevla.sim.scenes import Scene

Cell = tuple[str, bool, float]
_STATE: dict[str, Any] = {}


def _setup(cfg: Config, scenes: Sequence[Scene], table_path: str, ckpt_path: Optional[str]) -> None:
    from deadlinevla.budget.latency_model import LatencyModel, LatencyTable, make_latency_source
    from deadlinevla.policy.base import make_policy
    from deadlinevla.sim.drone_env import make_env

    table = LatencyTable.load(table_path)
    adapter = make_policy(cfg)
    if adapter.trainable and ckpt_path is not None:
        adapter.load(ckpt_path, cfg.model.device)
    _STATE.update(
        cfg=cfg, scenes=list(scenes), adapter=adapter, env=make_env(cfg),
        latency=LatencyModel(table, cfg.budget.latency_stat, cfg.budget.max_residual, cfg.tier_tokens),
        # The simulated loop sees the onboard computer's latency, never the desktop's.
        source=make_latency_source(cfg.serve.board, cfg.tier_tokens, table),
    )


def run_cell(cell: Cell) -> tuple[Cell, list[EpisodeLog]]:
    from deadlinevla.serve.pilot import Pilot
    from deadlinevla.serve.runtime import SimulatedRuntime

    method, gov, cap = cell
    s = _STATE
    cfg = s["cfg"]
    rng = np.random.default_rng([cfg.run.seed, int(round(cap * 100))])
    pilot = Pilot(cfg, SimulatedRuntime(s["adapter"], s["source"], rng), s["latency"], method, gov, s["adapter"])
    return cell, run_episodes(pilot, s["env"], s["scenes"], cap)


def run_sweep(cfg: Config, scenes: Sequence[Scene], table_path: str | Path, ckpt_path: Optional[str | Path],
              workers: int = 1, on_cell: Callable[[Cell, list[EpisodeLog]], Any] = lambda c, logs: None
              ) -> list[tuple[Cell, list[EpisodeLog]]]:
    cells = [(m, g, float(c)) for m in cfg.eval.methods for g in cfg.eval.governor for c in cfg.eval.speed_caps]
    args = (cfg, list(scenes), str(table_path), None if ckpt_path is None else str(ckpt_path))
    out = []
    if workers <= 1:
        _setup(*args)
        for cell in cells:
            res = run_cell(cell)
            on_cell(*res)
            out.append(res)
        return out
    with mp.get_context("spawn").Pool(workers, initializer=_setup, initargs=args) as pool:
        for res in pool.imap(run_cell, cells):
            on_cell(*res)
            out.append(res)
    return out
