import numpy as np
import pytest

from deadlinevla.budget.latency_model import SyntheticBoard, synthetic_prior
from deadlinevla.config import set_field
from deadlinevla.eval.baselines import baseline_alloc
from deadlinevla.eval.metrics import summarize
from deadlinevla.eval.rollout import build_eval_scenes, run_episodes
from deadlinevla.policy.base import make_policy
from deadlinevla.serve.pilot import MODES, Pilot
from deadlinevla.serve.runtime import SimulatedRuntime
from deadlinevla.sim.drone_env import make_env


@pytest.fixture(scope="module")
def setup(light_cfg):
    cfg = set_field(light_cfg, "model.policy", "surrogate")
    lat = synthetic_prior("synthetic_orin_nx", cfg.tier_tokens)
    board = SyntheticBoard("synthetic_orin_nx", cfg.tier_tokens)
    return cfg, lat, board, make_policy(cfg)


def test_baselines(setup):
    cfg, lat, _, _ = setup
    assert baseline_alloc("all_high", {}, lat, cfg).tolist() == [3] * 6
    assert baseline_alloc("front_only", {}, lat, cfg).tolist() == [3, 0, 0, 0, 0, 0]
    assert baseline_alloc("uniform_fixed", {}, lat, cfg).tolist() == [2] * 6
    imp = baseline_alloc("importance", {"importance": np.array([10, 1, 1, 1, 1, 1.0])}, lat, cfg)
    assert imp[0] == imp.max()
    budget = 6 * cfg.tier_tokens[2]
    assert np.asarray(cfg.tier_tokens)[imp].sum() <= budget


@pytest.mark.parametrize("mode", MODES)
def test_pilot_runs_every_mode(mode, setup):
    cfg, lat, board, adapter = setup
    env = make_env(cfg)
    scenes = build_eval_scenes(cfg, ["goal_nav"], per_family=1)
    pilot = Pilot(cfg, SimulatedRuntime(adapter, board, np.random.default_rng(0)), lat, mode, True, adapter)
    logs = run_episodes(pilot, env, scenes, 4.0)
    m = summarize(logs)
    assert m.episodes == 1 and 0.0 <= m.violation_rate <= 1.0
    t = logs[0].ticks
    assert len(t["speed"]) == len(t["alloc"]) > 0
    assert (t["v_gov"] <= 4.0 + 1e-9).all()  # speed cap holds
    assert (t["v_gov"] <= t["v_cmd"] + 1e-9).all()  # the governor only ever slows down


def test_governor_slows_slow_allocations(setup):
    """With the slowest allocation in a slalom, the governor must cut commanded speed somewhere."""
    cfg, lat, board, adapter = setup
    env = make_env(cfg)
    pilot = Pilot(cfg, SimulatedRuntime(adapter, board, np.random.default_rng(1)), lat, "all_high", True, adapter)
    t = run_episodes(pilot, env, build_eval_scenes(cfg, ["slalom"], per_family=1), 8.0)[0].ticks
    assert (t["v_gov"] < t["v_cmd"] - 1e-6).any()
