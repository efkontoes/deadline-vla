"""v4: one perception model for policy and governor."""

import numpy as np
import pytest

from deadlinevla.budget.relative import Contacts, govern_relative
from deadlinevla.config import set_field
from deadlinevla.perception import detection_range, focal_px, visible


def test_review_example(light_cfg):
    # FOV 90 deg, s = 0.5 m, p_min = 8 px: low tier (128 px) 4 m, high tier (512 px) 16 m
    assert focal_px(128, 90.0) == pytest.approx(64.0)
    assert detection_range(np.array([0, 1, 3]), np.full(3, 0.5), light_cfg).tolist() == pytest.approx([0, 4, 16])
    assert detection_range(np.array([1]), np.array([0.25]), light_cfg)[0] == pytest.approx(2.0)


def test_visibility_needs_fov_and_range(light_cfg):
    behind = np.array([[-6.0, 0.0]])            # 6 m straight behind, s = 0.5 -> needs tier >= 2 (8 m)
    s, d = np.array([0.5]), np.array([5.75])
    assert not visible(behind, s, d, [3, 3, 3, 0, 3, 3], light_cfg)[0]   # rear camera off
    assert not visible(behind, s, d, [3, 3, 3, 1, 3, 3], light_cfg)[0]   # rear at low tier: 4 m < 5.75 m
    assert visible(behind, s, d, [0, 0, 0, 2, 0, 0], light_cfg)[0]       # rear at mid tier: 8 m


def test_camera_governor_cannot_react_to_what_it_cannot_see(light_cfg):
    from deadlinevla.perception import visible as vis

    threat = Contacts(p=np.array([[-4.0, 0.0]]), R=np.array([0.5]), w=np.array([[6.0, 0.0]]), size=np.array([0.5]))
    for alloc, expect_escape in (([3, 0, 0, 0, 0, 0], False), ([3, 0, 0, 3, 0, 0], True)):
        seen = threat.subset(vis(threat.p, threat.size, threat.d, alloc, light_cfg))
        out = govern_relative(np.zeros(4), 0.3, seen, 8.4, 0.5, 10.0)
        assert (out[0] > 0.5) == expect_escape


def test_oracle_mode_is_still_available(light_cfg):
    cfg = set_field(light_cfg, "perception.governor", "oracle")
    assert cfg.perception.governor == "oracle"
