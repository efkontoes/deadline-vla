"""GR00T N1.7 adapter over upstream Isaac-GR00T: the desktop-only, large-model reference.

Qwen3-VL encodes images at native resolution, so a tier is simply the size of the image handed in;
views at tier 0 are left out of the observation. Inference needs a 16 GB+ GPU, fine-tuning 40 GB+.
Fine-tuning runs through Isaac-GR00T's own launcher on the LeRobot dataset this repo records, with
``render_views`` applied by a data transform; ``forward_loss`` is therefore not implemented here.

Status: written against the Isaac-GR00T policy API; not executed in this repository.
Weights are under the NVIDIA Open Model License.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deadlinevla.config import Config
from deadlinevla.policy.base import PolicyAdapter, PolicySpec
from deadlinevla.policy.views import render_views
from deadlinevla.sim.drone_env import ACTION_DIM, STATE_KEYS

EMBODIMENT_TAG = "new_embodiment"


class Gr00tAdapter(PolicyAdapter):  # pragma: no cover - needs Isaac-GR00T + a large GPU
    name = "gr00t_n17"
    trainable = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.policy = None
        self.load(cfg.model.pretrained, cfg.model.device)

    def load(self, path, device="cuda"):
        try:
            from gr00t.policy.gr00t_policy import Gr00tPolicy
        except ImportError:
            try:
                from gr00t.model.policy import Gr00tPolicy
            except ImportError as e:
                raise ImportError("model.policy=gr00t_n17 needs Isaac-GR00T installed from its repo") from e
        self.policy = Gr00tPolicy(model_path=str(path), embodiment_tag=EMBODIMENT_TAG, device=device)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "README.txt").write_text(f"GR00T reference: weights at {self.cfg.model.pretrained}\n")
        return path

    def spec(self) -> PolicySpec:
        H = self.cfg.model.chunk_size or 16
        K = self.cfg.model.num_denoise_steps or 4
        return PolicySpec(H, K, self.cfg.tier_res, self.cfg.tier_tokens, ACTION_DIM, len(STATE_KEYS))

    def forward_loss(self, batch, alloc):
        raise NotImplementedError("fine-tune GR00T with the Isaac-GR00T launcher (see README)")

    def predict_chunk(self, obs, alloc):
        views = render_views(obs["images"], alloc, self.cfg.tier_res)
        o = {f"video.cam{rv.view}": rv.image[None, None] for rv in views}   # (B=1, T=1, H, W, 3)
        state = np.asarray(obs["state"], dtype=np.float32)
        for k, name in enumerate(STATE_KEYS):
            o[f"state.{name}"] = state[None, None, k:k + 1]
        o["annotation.human.task_description"] = [obs["instruction"]]
        out = self.policy.get_action(o)
        if isinstance(out, tuple):
            out = out[0]
        keys = ("vx_b", "vy_b", "vz_b", "yaw_rate")
        chunk = np.concatenate([np.asarray(out[f"action.{k}"]).reshape(-1, 1) for k in keys], axis=1)
        return chunk[: self.spec().chunk_size]
