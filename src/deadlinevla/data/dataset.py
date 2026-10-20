"""Demonstration dataset, allocation sampler p(r) and the budget-randomised data loader."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from deadlinevla.budget.allocator import camera_axes, choose_allocation, quality_table, view_weights
from deadlinevla.budget.deadline import max_latency
from deadlinevla.budget.latency_model import LatencyModel, synthetic_prior
from deadlinevla.config import Config, a_brake
from deadlinevla.policy.views import render_views


@dataclass
class Episode:
    images: np.ndarray       # (T, V, R, R, 3) uint8
    state: np.ndarray        # (T, 7)
    action: np.ndarray       # (T, 4)
    clearance: np.ndarray    # (T,)
    instruction: str
    family: str = ""
    seed: int = -1


class DemoDataset:
    """Episodes in memory; sample i returns frame t and the next H actions (padded with the last)."""

    def __init__(self, episodes: list[Episode], chunk_size: int):
        self.episodes = episodes
        self.H = chunk_size
        self.index = [(e, t) for e, ep in enumerate(episodes) for t in range(len(ep.action))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        e, t = self.index[i]
        ep = self.episodes[e]
        T = len(ep.action)
        idx = np.minimum(np.arange(t, t + self.H), T - 1)
        return {"images": ep.images[t], "state": ep.state[t], "instruction": ep.instruction,
                "actions": ep.action[idx], "action_is_pad": np.arange(t, t + self.H) >= T,
                "clearance": float(ep.clearance[t])}


def _load_npz(root: Path) -> list[Episode]:
    meta = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    eps = []
    for m in meta:
        z = np.load(root / "data" / f"episode_{m['episode_index']:06d}.npz")
        eps.append(Episode(images=z["images"], state=z["state"], action=z["action"], clearance=z["clearance"],
                           instruction=str(z["task"]), family=m.get("family", ""), seed=m.get("seed", -1)))
    return eps


def _load_lerobot(root: Path, repo_id: str, n_views: int) -> list[Episode]:  # pragma: no cover
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=root)
    eps: dict[int, dict[str, list]] = {}
    for i in range(len(ds)):
        f = ds[i]
        e = int(f["episode_index"])
        d = eps.setdefault(e, {"images": [], "state": [], "action": [], "clearance": [], "task": f["task"]})
        imgs = [f[f"observation.images.cam{j}"] for j in range(n_views)]
        d["images"].append(np.stack([(np.asarray(im).transpose(1, 2, 0) * 255).astype(np.uint8)
                                     if np.asarray(im).shape[0] == 3 else np.asarray(im) for im in imgs]))
        d["state"].append(np.asarray(f["observation.state"]))
        d["action"].append(np.asarray(f["action"]))
        d["clearance"].append(float(np.asarray(f["observation.clearance"]).ravel()[0]))
    return [Episode(np.stack(d["images"]), np.stack(d["state"]), np.stack(d["action"]),
                    np.asarray(d["clearance"]), d["task"]) for _, d in sorted(eps.items())]


def split_episodes(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Episode-level split: no frame of a validation episode is ever trained on."""
    perm = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def load_dataset(cfg: Config, root: str | Path, split: str = "train", chunk_size: Optional[int] = None) -> DemoDataset:
    root = Path(root)
    if (root / "meta" / "episodes.jsonl").exists() and (root / "data").exists() and \
            any((root / "data").glob("episode_*.npz")):
        eps = _load_npz(root)
    else:
        eps = _load_lerobot(root, cfg.data.repo_id, cfg.cameras.n_cams)
    tr, va = split_episodes(len(eps), cfg.data.val_frac, cfg.run.seed)
    pick = {"train": tr, "val": va, "all": np.arange(len(eps))}[split]
    return DemoDataset([eps[i] for i in pick], chunk_size or cfg.model.chunk_size or 5)


# --------------------------------------------------------------------------------------------------
# p(r): the allocation sampler


@functools.lru_cache(maxsize=4)
def _prior(board: str, tokens: tuple[int, ...], n_views: int) -> LatencyModel:
    return synthetic_prior(board, tokens, n_views)


def sample_alloc(cfg: Config, rng: np.random.Generator, latency: Optional[LatencyModel] = None) -> np.ndarray:
    """Draw r from p(r).

    uniform    each view's tier uniform over 0..3
    allocator  random speed, direction and clearance pushed through the real allocator
    mixed      half and half
    Then each view is switched off with probability ``p_view_off`` (at least one view stays on unless
    the allocator itself chose all-off, which is a legal run-time outcome).
    """
    V = cfg.cameras.n_cams
    mode = cfg.train.alloc_sampler
    if mode == "mixed":
        mode = "uniform" if rng.random() < 0.5 else "allocator"
    if mode == "uniform":
        alloc = rng.integers(0, 4, size=V)
    else:
        lat = latency or _prior(cfg.serve.board, cfg.tier_tokens, V)
        speed = rng.uniform(0.0, cfg.drone.v_max)
        h = rng.uniform(-np.pi, np.pi)
        v_body = np.array([np.cos(h), np.sin(h), 0.0]) * speed
        d = rng.uniform(0.5, 0.75 * cfg.sim.max_range)
        L_max = max_latency(speed, d, a_brake(cfg), cfg.drone.margin)
        w = view_weights(v_body, camera_axes(cfg.cameras.cam_yaws), cfg.budget.kappa, cfg.budget.epsilon,
                         cfg.budget.hover_speed)
        alloc, _ = choose_allocation(w, L_max, lat, quality_table(cfg.budget.quality_fn, cfg.tier_tokens),
                                     cfg.budget.objective, cfg.budget.qmin_mid_weight)
        alloc = alloc.copy()
    was_on = alloc.any()
    alloc[rng.random(V) < cfg.train.p_view_off] = 0
    if was_on and not alloc.any():
        alloc[rng.integers(V)] = rng.integers(1, 4)
    return alloc.astype(np.int64)


def make_dataloader(ds: DemoDataset, cfg: Config, sampler=sample_alloc, rng: Optional[np.random.Generator] = None,
                    batch_size: Optional[int] = None, shuffle: bool = True, latency: Optional[LatencyModel] = None,
                    fixed_alloc: Optional[np.ndarray] = None) -> Iterator[dict[str, Any]]:
    """Endless iterator of batches; a fresh allocation is drawn for every sample.

    ``fixed_alloc`` overrides the sampler (used by offline evaluation on an allocation grid).
    """
    rng = rng or np.random.default_rng(cfg.run.seed)
    B = batch_size or cfg.train.batch_size
    n = len(ds)
    while True:
        order = rng.permutation(n) if shuffle else np.arange(n)
        for s in range(0, n, B):
            items = [ds[int(i)] for i in order[s:s + B]]
            allocs = np.stack([np.asarray(fixed_alloc) if fixed_alloc is not None else sampler(cfg, rng, latency)
                               for _ in items])
            yield collate(items, allocs, cfg)
        if not shuffle:
            return


def collate(items: list[dict[str, Any]], allocs: np.ndarray, cfg: Config) -> dict[str, Any]:
    """Render every sample at its own allocation and stack the rest."""
    return {
        "views": [render_views(it["images"], a, cfg.tier_res) for it, a in zip(items, allocs)],
        "alloc": np.asarray(allocs),
        "state": np.stack([it["state"] for it in items]),
        "instruction": [it["instruction"] for it in items],
        "actions": np.stack([it["actions"] for it in items]),
        "action_is_pad": np.stack([it["action_is_pad"] for it in items]),
        "clearance": np.array([it["clearance"] for it in items]),
    }
