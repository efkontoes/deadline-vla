import numpy as np
import pytest

from deadlinevla.budget.clearance import depth_clearance
from deadlinevla.sim.drone_env import make_env, measure_braking_details
from deadlinevla.sim.expert import Expert
from deadlinevla.sim.scenes import FAMILIES, build_scene, cone_clearance, relative_direction


@pytest.mark.parametrize("family", FAMILIES)
def test_scenes_are_deterministic(family, light_cfg):
    a, b = build_scene(family, 3, light_cfg), build_scene(family, 3, light_cfg)
    assert np.array_equal(a.obs_xy, b.obs_xy) and a.instruction == b.instruction
    assert (family == "rear_threat") == (a.mover is not None)


def test_goal_nav_puts_goals_behind_sometimes(light_cfg):
    dirs = {relative_direction(build_scene("goal_nav", s, light_cfg)) for s in range(40)}
    assert "behind you" in dirs and "in front of you" in dirs


def test_measured_braking_is_physical(light_cfg):
    det = measure_braking_details(make_env(light_cfg))
    assert 5.0 < det["a_brake"] <= light_cfg.drone.a_max
    for run in det["runs"]:  # the braking model upper-bounds every measured stop
        assert run["stop_dist"] <= run["v"] ** 2 / (2 * det["a_brake"]) + 1e-9


@pytest.mark.parametrize("family", FAMILIES)
def test_expert_reaches_goal_without_collision(family, light_cfg):
    env, ex = make_env(light_cfg), Expert(light_cfg)
    for seed in range(2):
        scene = build_scene(family, seed, light_cfg)
        assert ex.reset(scene)
        obs = env.reset(scene)
        while not obs["info"]["done"]:
            obs, info = env.step(ex.act(obs))
        assert info["goal_reached"] and not info["collision"], (family, seed)


def test_observation_contract(light_cfg):
    env = make_env(light_cfg)
    obs = env.reset(build_scene("goal_nav", 0, light_cfg))
    R = light_cfg.tier_res[3]
    assert obs["images"].shape == (6, R, R, 3) and obs["images"].dtype == np.uint8
    assert obs["depth"].shape == (6, R, R)
    assert obs["state"].shape == (7,)
    assert isinstance(obs["instruction"], str) and obs["instruction"]


def test_depth_clearance_tracks_oracle(light_cfg):
    """v2 depth clearance should agree with ground truth when an obstacle is straight ahead."""
    env = make_env(light_cfg)
    scene = build_scene("slalom", 0, light_cfg)
    obs = env.reset(scene)
    yaw = scene.start[3]
    v_world = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    oracle = cone_clearance(scene, env.pos, 0.0, v_world, 10.0, light_cfg.drone.radius, light_cfg.sim.max_range)
    depth = depth_clearance(obs["depth"], np.array([1.0, 0.0, 0.0]), env.pos[2], light_cfg.cameras.fov,
                            light_cfg.cameras.cam_yaws, 10.0, light_cfg.drone.radius, light_cfg.sim.max_range)
    assert oracle < 6.0
    assert depth == pytest.approx(oracle, abs=0.6)
