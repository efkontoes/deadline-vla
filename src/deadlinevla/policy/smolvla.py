"""SmolVLA adapter over upstream LeRobot. No model code is copied.

Two changes to the upstream image loop, both done by wrapping methods on the loaded instance:

1. ``prepare_images`` returns only the active views, each already at its tier resolution, instead of
   every configured camera padded to 512 px. SmolVLM's vision embeddings handle variable patch grids,
   so a 256 px view genuinely costs fewer patches and fewer visual tokens.
2. ``embed_image`` adds the ``ViewTierEmbedding`` for the view being embedded.

Fallback (Risks: "base model plumbing"): ``fixed_slots=True`` keeps all six slots at the native
resolution and masks off views through ``img_masks``. Correct but saves no compute; use it only if
variable views fight the model.

Status: written against lerobot>=0.4 (``lerobot.policies.smolvla``); not executed in this repository
(no GPU here). Validating it is go / no-go check 2 at M0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from deadlinevla.config import Config
from deadlinevla.policy.base import PolicyAdapter, PolicySpec
from deadlinevla.policy.views import ViewTierEmbedding, render_views
from deadlinevla.sim.drone_env import ACTION_DIM, STATE_KEYS


def _import_smolvla():
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as e:  # pragma: no cover
        raise ImportError("model.policy=smolvla needs LeRobot: pip install -e .[lerobot]") from e
    try:
        from lerobot.policies.factory import make_pre_post_processors
    except ImportError:  # older lerobot: tokenisation happens inside the policy
        make_pre_post_processors = None
    return SmolVLAPolicy, make_pre_post_processors


class SmolVLAAdapter(PolicyAdapter):  # pragma: no cover - needs torch + lerobot
    name = "smolvla"

    def __init__(self, cfg: Config, fixed_slots: bool = False):
        super().__init__(cfg)
        import torch

        self.torch = torch
        self.fixed_slots = fixed_slots
        self.device = cfg.model.device if torch.cuda.is_available() else "cpu"
        self.policy = None
        self.optimizer = None
        self._views: list[tuple[int, int]] = []
        self._cursor = 0
        self._precomputed = None   # vision tokens from TrtRuntime's per-tier engines
        self.load(cfg.model.pretrained, self.device)

    # ---------------------------------------------------------------------------------------------
    def load(self, path, device="cpu"):
        torch = self.torch
        SmolVLAPolicy, make_pp = _import_smolvla()
        self.policy = SmolVLAPolicy.from_pretrained(str(path)).to(device)
        pc = self.policy.config
        self.pre, self.post = (make_pp(pc, pretrained_path=str(path)) if make_pp else (None, None))
        hidden = self.policy.model.vlm_with_expert.config.text_config.hidden_size
        self.vt_embed = ViewTierEmbedding(self.cfg.cameras.n_cams, 4, hidden).to(device)
        extra = Path(path) / "view_tier_embedding.pt"
        if extra.exists():
            self.vt_embed.load_state_dict(torch.load(extra, map_location=device))
        self._patch()
        params = list(self.policy.parameters()) + list(self.vt_embed.parameters())
        self.optimizer = torch.optim.AdamW([p for p in params if p.requires_grad], lr=self.cfg.train.lr)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(path)
        self.torch.save(self.vt_embed.state_dict(), path / "view_tier_embedding.pt")
        return path

    def spec(self) -> PolicySpec:
        pc = self.policy.config
        H = self.cfg.model.chunk_size or int(getattr(pc, "n_action_steps", getattr(pc, "chunk_size", 50)))
        K = self.cfg.model.num_denoise_steps or int(getattr(pc, "num_steps", 10))
        return PolicySpec(H, K, self.cfg.tier_res, self.cfg.tier_tokens, ACTION_DIM, len(STATE_KEYS))

    # ---------------------------------------------------------------------------------------------
    def _patch(self) -> None:
        """Wrap the upstream image loop on this instance only."""
        policy, adapter = self.policy, self
        torch = self.torch
        orig_prepare = policy.prepare_images
        vlm = policy.model.vlm_with_expert
        orig_embed = vlm.embed_image

        def prepare_images(batch):
            if adapter.fixed_slots:
                images, masks = orig_prepare(batch)
                on = {j for j, _ in adapter._views}
                masks = [m & (j in on) for j, m in enumerate(masks)]
                adapter._cursor = 0
                return images, masks
            images, masks = [], []
            for j, _ in adapter._views:
                img = batch[f"observation.images.cam{j}"].to(adapter.device, dtype=torch.float32)
                images.append(img * 2.0 - 1.0)  # SigLIP range, as upstream does after resize
                masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=adapter.device))
            adapter._cursor = 0
            return images, masks

        def embed_image(image):
            pre = adapter._precomputed
            tokens = pre[adapter._cursor] if pre is not None else orig_embed(image)
            if adapter._cursor < len(adapter._views):
                j, r = adapter._views[adapter._cursor]
                tokens = adapter.vt_embed(tokens, j, r)
            adapter._cursor += 1
            return tokens

        policy.prepare_images = prepare_images
        vlm.embed_image = embed_image

    def _to_batch(self, views_list, state, instructions, actions=None) -> dict[str, Any]:
        """All samples in a batch must share one allocation (the loader groups by allocation)."""
        torch = self.torch
        first = views_list[0]
        self._views = [(rv.view, rv.tier) for rv in first]
        batch: dict[str, Any] = {}
        for k, rv in enumerate(first):
            imgs = np.stack([vl[k].image for vl in views_list]).astype(np.float32) / 255.0
            batch[f"observation.images.cam{rv.view}"] = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(self.device)
        batch["observation.state"] = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device)
        batch["task"] = list(instructions)
        if actions is not None:
            batch["action"] = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        return self.pre(batch) if self.pre is not None else batch

    def forward_loss(self, batch, alloc):
        allocs = np.asarray(alloc)
        if not (allocs == allocs[0]).all():
            # Split into groups that share an allocation; the loss is the sample-weighted mean.
            total, per = 0.0, np.zeros(len(allocs))
            keys = [tuple(a) for a in allocs]
            for key in dict.fromkeys(keys):
                idx = [i for i, k in enumerate(keys) if k == key]
                sub = {k: [batch[k][i] for i in idx] if isinstance(batch[k], list) else np.asarray(batch[k])[idx]
                       for k in ("views", "state", "instruction", "actions")}
                loss, m = self.forward_loss(sub, np.asarray([key] * len(idx)))
                total = total + loss * len(idx)
                per[idx] = float(loss.detach())
            return total / len(allocs), {"per_sample": per}
        b = self._to_batch(batch["views"], batch["state"], batch["instruction"], batch["actions"])
        loss, _ = self.policy.forward(b)
        return loss, {"per_sample": np.full(len(allocs), float(loss.detach()))}

    def predict_chunk(self, obs, alloc):
        torch = self.torch
        views = render_views(obs["images"], alloc, self.cfg.tier_res)
        if not views:
            # Every view off: run with the front view at the lowest tier masked out by fixed slots.
            views = render_views(obs["images"], [1] + [0] * (len(alloc) - 1), self.cfg.tier_res)
        b = self._to_batch([views], np.asarray(obs["state"])[None], [obs["instruction"]])
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(b)
        if self.post is not None:
            chunk = self.post(chunk)
        return chunk[0].float().cpu().numpy()[:, :ACTION_DIM]

    def export_vision_onnx(self, path: Path, tier: int) -> Path:
        """Vision encoder + connector at one tier resolution, as ONNX (input ``image``, output ``tokens``)."""
        torch = self.torch
        vlm = self.policy.model.vlm_with_expert

        class _Vision(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, image):
                return self.fn(image)

        res = self.cfg.tier_res[tier]
        # the class method is the unpatched upstream encoder (the instance attribute is our wrapper);
        # the (view, tier) embedding is added at run time, not baked into the engine
        fn = lambda img: type(vlm).embed_image(vlm, img)  # noqa: E731
        dummy = torch.zeros(1, 3, res, res, device=self.device)
        torch.onnx.export(_Vision(fn), (dummy,), str(path), input_names=["image"], output_names=["tokens"],
                          opset_version=17)
        return Path(path)

    def view_importance(self) -> Optional[np.ndarray]:
        return None  # B3 uses attention hooks registered by eval.baselines when available
