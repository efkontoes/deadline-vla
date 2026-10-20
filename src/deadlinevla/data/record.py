"""Record expert demonstrations.

All six views are stored at the highest tier only; lower tiers are rendered later by downsampling
(``policy.views.render_views``), so one recording serves every allocation.

Writes a LeRobot dataset when ``lerobot`` is installed (``data.format`` auto|lerobot), otherwise a
LeRobot-shaped local layout (``data.format=npz``) with the same feature names:

    <root>/meta/info.json
    <root>/meta/episodes.jsonl
    <root>/data/episode_000000.npz   images (T,V,R,R,3) uint8, state, action, clearance, task
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from deadlinevla.config import Config
from deadlinevla.sim.drone_env import ACTION_DIM, STATE_KEYS, DroneEnv
from deadlinevla.sim.expert import Expert
from deadlinevla.sim.scenes import build_scene

ACTION_KEYS = ("vx_b", "vy_b", "vz_b", "yaw_rate")


def lerobot_available() -> bool:
    try:
        import lerobot  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_format(fmt: str) -> str:
    if fmt == "auto":
        return "lerobot" if lerobot_available() else "npz"
    if fmt not in ("lerobot", "npz"):
        raise ValueError(f"data.format must be auto, lerobot or npz, got {fmt!r}")
    return fmt


class NpzWriter:
    def __init__(self, root: Path, repo_id: str, cfg: Config):
        self.root, self.repo_id, self.cfg = root, repo_id, cfg
        (root / "data").mkdir(parents=True, exist_ok=True)
        (root / "meta").mkdir(parents=True, exist_ok=True)
        self.episodes: list[dict] = []

    def add_episode(self, frames: dict[str, np.ndarray], task: str, extra: dict) -> None:
        i = len(self.episodes)
        np.savez_compressed(self.root / "data" / f"episode_{i:06d}.npz", task=np.array(task), **frames)
        self.episodes.append({"episode_index": i, "length": int(len(frames["action"])), "task": task, **extra})

    def finalize(self, meta: dict) -> None:
        cfg = self.cfg
        info = {
            "repo_id": self.repo_id, "format": "deadlinevla-npz", "fps": cfg.drone.ctrl_hz,
            "total_episodes": len(self.episodes), "total_frames": sum(e["length"] for e in self.episodes),
            "features": {
                **{f"observation.images.cam{j}": {"dtype": "uint8", "shape": [cfg.tier_res[3]] * 2 + [3]}
                   for j in range(cfg.cameras.n_cams)},
                "observation.state": {"dtype": "float32", "shape": [len(STATE_KEYS)], "names": list(STATE_KEYS)},
                "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": list(ACTION_KEYS)},
                "observation.clearance": {"dtype": "float32", "shape": [1]},
            },
            "cam_yaws": list(cfg.cameras.cam_yaws), **meta,
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        with open(self.root / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
            for e in self.episodes:
                f.write(json.dumps(e) + "\n")


class LeRobotWriter:  # pragma: no cover - needs lerobot
    def __init__(self, root: Path, repo_id: str, cfg: Config):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        R = cfg.tier_res[3]
        features = {
            **{f"observation.images.cam{j}": {"dtype": "image", "shape": (R, R, 3),
                                               "names": ["height", "width", "channels"]}
               for j in range(cfg.cameras.n_cams)},
            "observation.state": {"dtype": "float32", "shape": (len(STATE_KEYS),), "names": list(STATE_KEYS)},
            "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": list(ACTION_KEYS)},
            "observation.clearance": {"dtype": "float32", "shape": (1,), "names": ["d"]},
        }
        self.ds = LeRobotDataset.create(repo_id=repo_id, fps=int(cfg.drone.ctrl_hz), features=features,
                                        root=root, use_videos=False)
        self.cfg = cfg
        self.n = 0

    def add_episode(self, frames, task, extra):
        V = self.cfg.cameras.n_cams
        for t in range(len(frames["action"])):
            frame = {f"observation.images.cam{j}": frames["images"][t, j] for j in range(V)}
            frame["observation.state"] = frames["state"][t].astype(np.float32)
            frame["action"] = frames["action"][t].astype(np.float32)
            frame["observation.clearance"] = np.array([frames["clearance"][t]], dtype=np.float32)
            frame["task"] = task
            self.ds.add_frame(frame)
        self.ds.save_episode()
        self.n += 1

    def finalize(self, meta):
        if hasattr(self.ds, "finalize"):
            self.ds.finalize()
        (Path(self.ds.root) / "meta" / "deadlinevla.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def record_demos(env: DroneEnv, expert: Expert, n_episodes: int, repo_id: str, root: str | Path,
                 families: Optional[Sequence[str]] = None, seed0: int = 0,
                 log: Callable[[str], Any] = print) -> Path:
    """Roll out the expert and keep only episodes that reach the goal without collision."""
    cfg = env.cfg
    root = Path(root)
    fmt = resolve_format(cfg.data.format)
    writer = LeRobotWriter(root, repo_id, cfg) if fmt == "lerobot" else NpzWriter(root, repo_id, cfg)
    families = list(families or cfg.scenes.families)
    seed, kept, rejected = seed0, 0, {"no_plan": 0, "failed": 0}
    max_ticks = int(cfg.sim.max_episode_s * cfg.drone.ctrl_hz) + 1
    while kept < n_episodes:
        if seed - seed0 > 5 * n_episodes + 20:
            raise RuntimeError(f"expert keeps failing: {rejected} after {seed - seed0} scenes")
        fam = families[kept % len(families)]
        scene = build_scene(fam, seed, cfg)
        seed += 1
        if not expert.reset(scene):
            rejected["no_plan"] += 1
            continue
        obs = env.reset(scene)
        buf: dict[str, list] = {"images": [], "state": [], "action": [], "clearance": []}
        info = obs["info"]
        for _ in range(max_ticks):
            a = expert.act(obs)
            buf["images"].append(obs["images"])
            buf["state"].append(obs["state"])
            buf["action"].append(a)
            buf["clearance"].append(obs["info"]["clearance"])
            obs, info = env.step(a)
            if info["done"]:
                break
        if not info["goal_reached"]:
            rejected["failed"] += 1
            continue
        frames = {"images": np.stack(buf["images"]).astype(np.uint8),
                  "state": np.stack(buf["state"]).astype(np.float32),
                  "action": np.stack(buf["action"]).astype(np.float32),
                  "clearance": np.asarray(buf["clearance"], dtype=np.float32)}
        writer.add_episode(frames, scene.instruction, {"family": fam, "seed": scene.seed})
        kept += 1
        log(f"  episode {kept}/{n_episodes} [{fam} seed={scene.seed}] {len(frames['action'])} steps")
    writer.finalize({"rejected": rejected, "format_written": fmt})
    return root
