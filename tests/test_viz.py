"""Replays reproduce evaluation cells exactly, and animations write valid GIFs."""

import numpy as np
import pytest
from PIL import Image

from deadlinevla.budget.latency_model import LatencyTable, synthetic_prior
from deadlinevla.config import set_field
from deadlinevla.eval.rollout import build_eval_scenes
from deadlinevla.eval.sweep import run_sweep
from deadlinevla.viz.replay import animate_side_by_side, check_against_log, replay_cell


@pytest.fixture(scope="module")
def setup(light_cfg, tmp_path_factory):
    cfg = light_cfg
    for k, v in (("model.policy", "surrogate"), ("sim.render", False), ("eval.methods", ("deadline", "front_only")),
                 ("eval.governor", (True,)), ("eval.speed_caps", (6.0,))):
        cfg = set_field(cfg, k, v)
    m = synthetic_prior("synthetic_orin_nx", cfg.tier_tokens)
    table = LatencyTable(m.allocs, m.table.p50, m.table.p95, m.table.pmax, meta={"board": "synthetic_orin_nx"})
    path = table.save(tmp_path_factory.mktemp("lat") / "t.npz")
    scenes = build_eval_scenes(cfg, ["goal_nav", "rear_threat"], per_family=2)
    return cfg, path, scenes


def test_replay_reproduces_the_sweep(setup):
    cfg, path, scenes = setup
    for (method, _gov, cap), logs in run_sweep(cfg, scenes, path, None, workers=1):
        episodes = [lg.summary() for lg in logs]
        traces = replay_cell(cfg, scenes, method, cap, path)
        check_against_log(traces, episodes, cap)               # raises on any difference
        assert all(len(t.pos) == len(lg.ticks["speed"]) for t, lg in zip(traces, logs))
        assert np.array_equal(traces[-1].tokens, logs[-1].ticks["tokens"])


def test_mismatch_is_caught(setup):
    cfg, path, scenes = setup
    traces = replay_cell(cfg, scenes[:1], "deadline", 6.0, path)
    fake = [{"method": "deadline", "governor": True, "family": traces[0].family, "seed": traces[0].seed,
             "speed_cap": 6.0, "success": not traces[0].success, "collision": traces[0].collision,
             "time": traces[0].time}]
    with pytest.raises(AssertionError):
        check_against_log(traces, fake, 6.0)


def test_animation_writes_a_gif(setup, tmp_path):
    cfg, path, scenes = setup
    a = replay_cell(cfg, scenes[2:3], "deadline", 6.0, path)[-1]          # a rear_threat scene: has a mover
    b = replay_cell(cfg, scenes[2:3], "front_only", 6.0, path)[-1]
    out = animate_side_by_side([a, b], ["ours", "front-only"], cfg, tmp_path / "x.gif", "test", "footer",
                               max_frames=12)
    im = Image.open(out)
    assert im.n_frames >= 12 and im.size[0] > 400
