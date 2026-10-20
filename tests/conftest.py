from pathlib import Path

import pytest

from deadlinevla.config import load_config, set_field

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="session")
def light_cfg():
    """Light config with a fixed braking deceleration close to what measure_braking finds."""
    cfg = load_config(CONFIGS / "light.yaml", ["sim.max_episode_s=20"])
    return set_field(set_field(cfg, "drone.a_brake", 8.4), "drone.tau_track", 0.1)
