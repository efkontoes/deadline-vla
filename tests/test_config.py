import dataclasses
from pathlib import Path

import pytest
import yaml

from deadlinevla.config import ConfigError, a_brake, load_config, save_resolved, set_field

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_default_yaml_matches_dataclass_defaults():
    assert load_config(CONFIGS / "default.yaml") == load_config(None, ["sim.backend=isaac"])


def test_light_inherits_and_overrides():
    cfg = load_config(CONFIGS / "light.yaml", ["drone.v_max=6", "eval.speed_caps=[1, 2]"])
    assert cfg.sim.backend == "light"
    assert cfg.drone.v_max == 6.0 and isinstance(cfg.drone.v_max, float)
    assert cfg.eval.speed_caps == (1.0, 2.0)
    assert cfg.tier_res == (0, 16, 32, 64)
    assert cfg.tier_tokens == (0, 16, 64, 256)
    assert cfg.drone.margin == 0.5  # inherited from default.yaml


@pytest.mark.parametrize("bad", ["drone.nope=1", "nosuch.field=1", "drone.v_max=fast", "serve.mode=sometimes"])
def test_rejects_bad_overrides(bad):
    with pytest.raises(ConfigError):
        load_config(CONFIGS / "light.yaml", [bad])


def test_frozen_and_set_field(tmp_path):
    cfg = load_config(CONFIGS / "light.yaml")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.drone.v_max = 3.0
    with pytest.raises(ConfigError):
        a_brake(cfg)
    cfg2 = set_field(cfg, "drone.a_brake", 8.4)
    assert a_brake(cfg2) == 8.4 and cfg.drone.a_brake == "measured"
    path = save_resolved(cfg2, tmp_path)
    raw = yaml.safe_load(path.read_text())
    assert raw["drone"]["a_brake"] == 8.4
    assert raw["cameras"]["tiers"][2] == {"res": 64, "tokens": 256}
