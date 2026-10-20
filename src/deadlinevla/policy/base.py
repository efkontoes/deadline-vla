"""PolicyAdapter: every model hides behind this interface.

Batches (from ``data.dataset.make_dataloader``) are dicts:
    views        list (B) of list[RenderedView]  - already rendered at each sample's allocation
    alloc        (B, V) int
    state        (B, 7) float
    instruction  list (B) of str
    actions      (B, H, 4) float target chunk
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from deadlinevla.config import Config


@dataclass(frozen=True)
class PolicySpec:
    chunk_size: int                 # H
    num_denoise_steps: int          # K
    tier_res: tuple[int, ...]       # resolution per tier 0..3
    tier_tokens: tuple[int, ...]    # n(r) per tier 0..3
    action_dim: int = 4
    state_dim: int = 7


class PolicyAdapter(ABC):
    name = "abstract"
    trainable = True
    checkpoint_name = "policy"   # file or directory name under <run>/checkpoints/

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def load(self, path: str | Path, device: str = "cpu") -> None:
        """Weights plus the model's own config."""

    @abstractmethod
    def save(self, path: str | Path) -> Path: ...

    @abstractmethod
    def spec(self) -> PolicySpec: ...

    @abstractmethod
    def forward_loss(self, batch: dict[str, Any], alloc: np.ndarray) -> tuple[Any, dict[str, Any]]:
        """Training loss with views rendered at ``alloc``. ``metrics['per_sample']`` if available."""

    @abstractmethod
    def predict_chunk(self, obs: dict[str, Any], alloc: np.ndarray) -> np.ndarray:
        """(H, 4) body-frame velocity chunk. PyTorch (or numpy) reference inference."""

    # Hooks with sensible defaults ----------------------------------------------------------------
    def fit_stats(self, dataset: Any) -> None:
        """Called once before training with the training split (normalisation statistics)."""

    def train_step(self, batch: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        """One optimiser step. Torch adapters use ``self.optimizer``; numpy adapters override."""
        loss, metrics = self.forward_loss(batch, batch["alloc"])
        opt = getattr(self, "optimizer", None)
        if opt is None:
            raise RuntimeError(f"{self.name}: no optimizer configured")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.detach()), metrics

    def view_importance(self) -> Optional[np.ndarray]:
        """Per-view importance from the last inference (baseline B3); None if unsupported."""
        return None

    def set_speed_cap(self, v_max: float) -> None:
        """Policies that choose their own cruise speed (the surrogate) follow the eval speed cap."""


def make_policy(cfg: Config) -> PolicyAdapter:
    name = cfg.model.policy
    if name == "mock":
        from deadlinevla.policy.mock import MockPolicy

        return MockPolicy(cfg)
    if name == "surrogate":
        from deadlinevla.policy.surrogate import SurrogatePolicy

        return SurrogatePolicy(cfg)
    if name == "smolvla":
        from deadlinevla.policy.smolvla import SmolVLAAdapter

        return SmolVLAAdapter(cfg)
    if name == "gr00t_n17":
        from deadlinevla.policy.gr00t import Gr00tAdapter

        return Gr00tAdapter(cfg)
    raise ValueError(f"unknown model.policy {name!r}")
