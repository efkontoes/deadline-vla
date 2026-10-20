"""Rendering views at their tier, and the learned (view, tier) embedding.

``render_views`` is the single place a recorded six-view frame becomes the model's input for an
allocation. Training and serving both call it, so the two can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:  # torch is optional: the light path never needs it
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None
    nn = None


@dataclass
class RenderedView:
    view: int       # camera index j
    tier: int       # r_j in 1..3
    image: Any      # (res, res, 3) uint8 array or (3, res, res) float tensor


def downsample(img: np.ndarray, res: int) -> np.ndarray:
    """Area-average an (R, R, C) image down to (res, res, C); nearest-neighbour if R % res != 0."""
    R = img.shape[0]
    if res == R:
        return img
    if R % res == 0:
        k = R // res
        out = img.reshape(res, k, res, k, -1).mean(axis=(1, 3))
        return out.astype(img.dtype) if img.dtype != np.uint8 else np.round(out).astype(np.uint8)
    idx = (np.arange(res) * R / res).astype(int)
    return img[idx][:, idx]


def render_views(images: Any, alloc: Sequence[int], tiers: Sequence[int]) -> list[RenderedView]:
    """Downsample each view to its tier resolution and drop views at tier 0.

    ``images``: (V, R, R, 3) uint8 array, or a (V, 3, R, R) float tensor.
    ``tiers``: resolution per tier index 0..3 (``Config.tier_res``).
    """
    out = []
    is_tensor = torch is not None and isinstance(images, torch.Tensor)
    for j, r in enumerate(alloc):
        r = int(r)
        if r == 0:
            continue
        res = int(tiers[r])
        if is_tensor:
            img = images[j]
            if img.shape[-1] != res:
                img = torch.nn.functional.interpolate(img[None].float(), size=(res, res), mode="area")[0]
        else:
            img = downsample(np.asarray(images[j]), res)
        out.append(RenderedView(view=j, tier=r, image=img))
    return out


if nn is not None:

    class ViewTierEmbedding(nn.Module):
        """Learned vector per (view j, tier r_j), added to every visual token of that view.

        Tells the backbone which direction and resolution each token came from. Initialised at zero
        so a pretrained model starts out unchanged.
        """

        def __init__(self, n_views: int, n_tiers: int, dim: int):
            super().__init__()
            self.n_tiers = n_tiers
            self.table = nn.Embedding(n_views * n_tiers, dim)
            nn.init.zeros_(self.table.weight)

        def forward(self, tokens: torch.Tensor, view: int, tier: int) -> torch.Tensor:
            idx = torch.tensor(view * self.n_tiers + tier, device=tokens.device)
            return tokens + self.table(idx).to(tokens.dtype)

else:  # pragma: no cover

    class ViewTierEmbedding:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("ViewTierEmbedding needs torch: pip install -e .[torch]")
