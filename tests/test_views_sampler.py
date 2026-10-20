import numpy as np

from deadlinevla.config import set_field
from deadlinevla.data.dataset import sample_alloc
from deadlinevla.policy.views import downsample, render_views


def test_render_views_drops_off_views_and_resizes(light_cfg):
    imgs = np.random.default_rng(0).integers(0, 255, (6, 64, 64, 3), dtype=np.uint8)
    views = render_views(imgs, [3, 0, 1, 2, 0, 1], light_cfg.tier_res)
    assert [(v.view, v.tier) for v in views] == [(0, 3), (2, 1), (3, 2), (5, 1)]
    assert [v.image.shape[0] for v in views] == [64, 16, 32, 16]
    assert np.array_equal(views[0].image, imgs[0])


def test_downsample_is_area_average():
    img = np.arange(16, dtype=np.float64).reshape(4, 4, 1)
    assert downsample(img, 2)[..., 0].tolist() == [[2.5, 4.5], [10.5, 12.5]]


def test_sampler_modes(light_cfg):
    rng = np.random.default_rng(0)
    for mode in ("uniform", "allocator", "mixed"):
        cfg = set_field(light_cfg, "train.alloc_sampler", mode)
        draws = np.stack([sample_alloc(cfg, rng) for _ in range(300)])
        assert draws.shape == (300, 6) and draws.min() >= 0 and draws.max() <= 3
        assert (draws == 0).any()  # views get switched off
    # allocator-biased draws favour some view over others (direction weighting), uniform does not
    cfg = set_field(set_field(light_cfg, "train.alloc_sampler", "allocator"), "train.p_view_off", 0.0)
    draws = np.stack([sample_alloc(cfg, rng) for _ in range(300)])
    assert len({tuple(d) for d in draws}) > 5
