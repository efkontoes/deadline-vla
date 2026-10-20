"""A tiny numpy stand-in for a VLA, so training, export, profiling and evaluation run without a GPU.

Per active view: pool the tier-rendered image to 4x4x3; slot it into that view's place (zeros for
views that are off); add a one-hot (view, tier) code, the playing role of ``ViewTierEmbedding``;
append state and a hashed bag of instruction words; regress the normalised H-step action chunk with
a one-hidden-layer MLP trained by Adam on MSE.

It is a plumbing test, not a navigation policy: it gets the budget-randomised training loop and the
variable-view input contract exercised end to end. MSE stands in for the flow-matching loss.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any, Optional

import numpy as np

from deadlinevla.config import Config
from deadlinevla.policy.base import PolicyAdapter, PolicySpec
from deadlinevla.policy.views import RenderedView, downsample, render_views

POOL = 4
N_WORDS = 32


def _instruction_features(text: str) -> np.ndarray:
    v = np.zeros(N_WORDS)
    for w in text.lower().replace(",", " ").split():
        v[zlib.crc32(w.encode()) % N_WORDS] = 1.0
    return v


class MockPolicy(PolicyAdapter):
    name = "mock"
    checkpoint_name = "policy.npz"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.V = cfg.cameras.n_cams
        self.H = cfg.model.chunk_size or 5
        self.K = cfg.model.num_denoise_steps or 10
        self.slot = POOL * POOL * 3
        self.in_dim = self.V * self.slot + self.V * 4 + 7 + N_WORDS
        self.hidden = cfg.model.hidden
        self.out_dim = self.H * 4
        rng = np.random.default_rng(cfg.run.seed)
        self.params = {
            "W1": rng.normal(0, 1 / np.sqrt(self.in_dim), (self.in_dim, self.hidden)),
            "b1": np.zeros(self.hidden),
            "W2": rng.normal(0, 1 / np.sqrt(self.hidden), (self.hidden, self.out_dim)) * 0.1,
            "b2": np.zeros(self.out_dim),
        }
        self.stats = {"a_mean": np.zeros(4), "a_std": np.ones(4), "s_mean": np.zeros(7), "s_std": np.ones(7)}
        self._adam = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in self.params.items()}
        self._t = 0
        self._last_importance: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------------------------------
    def spec(self) -> PolicySpec:
        return PolicySpec(self.H, self.K, self.cfg.tier_res, self.cfg.tier_tokens)

    def fit_stats(self, dataset: Any) -> None:
        acts = np.concatenate([ep.action for ep in dataset.episodes])
        states = np.concatenate([ep.state for ep in dataset.episodes])
        self.stats = {"a_mean": acts.mean(0), "a_std": acts.std(0) + 1e-3,
                      "s_mean": states.mean(0), "s_std": states.std(0) + 1e-3}

    def features(self, views: list[RenderedView], state: np.ndarray, instruction: str) -> np.ndarray:
        x = np.zeros(self.in_dim)
        for rv in views:
            img = np.asarray(rv.image, dtype=np.float64)
            pooled = downsample(img, POOL)
            x[rv.view * self.slot:(rv.view + 1) * self.slot] = pooled.ravel() / 127.5 - 1.0
            x[self.V * self.slot + rv.view * 4 + rv.tier] = 1.0
        for j in range(self.V):
            if not any(rv.view == j for rv in views):
                x[self.V * self.slot + j * 4] = 1.0  # tier 0 code
        o = self.V * self.slot + self.V * 4
        x[o:o + 7] = (np.asarray(state) - self.stats["s_mean"]) / self.stats["s_std"]
        x[o + 7:] = _instruction_features(instruction)
        return x

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.params
        h = np.tanh(X @ p["W1"] + p["b1"])
        return h, h @ p["W2"] + p["b2"]

    # ---------------------------------------------------------------------------------------------
    def _batch_xy(self, batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        X = np.stack([self.features(v, s, i) for v, s, i in
                      zip(batch["views"], batch["state"], batch["instruction"])])
        Y = ((np.asarray(batch["actions"]) - self.stats["a_mean"]) / self.stats["a_std"]).reshape(len(X), -1)
        return X, Y

    def forward_loss(self, batch, alloc):
        X, Y = self._batch_xy(batch)
        _, out = self._forward(X)
        per = ((out - Y) ** 2).mean(axis=1)
        return float(per.mean()), {"per_sample": per}

    def train_step(self, batch):
        X, Y = self._batch_xy(batch)
        p = self.params
        h, out = self._forward(X)
        diff = out - Y
        per = (diff ** 2).mean(axis=1)
        g_out = 2 * diff / diff.size
        grads = {"W2": h.T @ g_out, "b2": g_out.sum(0)}
        g_h = (g_out @ p["W2"].T) * (1 - h ** 2)
        grads.update(W1=X.T @ g_h, b1=g_h.sum(0))
        self._t += 1
        lr, b1, b2 = self.cfg.train.lr, 0.9, 0.999
        for k, g in grads.items():
            m, v = self._adam[k]
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g * g
            mh, vh = m / (1 - b1 ** self._t), v / (1 - b2 ** self._t)
            p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        return float(per.mean()), {"per_sample": per}

    def predict_chunk(self, obs, alloc):
        views = render_views(obs["images"], alloc, self.cfg.tier_res)
        x = self.features(views, obs["state"], obs["instruction"])
        W1 = self.params["W1"]
        contrib = np.array([np.linalg.norm(x[j * self.slot:(j + 1) * self.slot] @ W1[j * self.slot:(j + 1) * self.slot])
                            for j in range(self.V)])
        self._last_importance = contrib + 1e-6
        _, out = self._forward(x[None])
        return out.reshape(self.H, 4) * self.stats["a_std"] + self.stats["a_mean"]

    def view_importance(self):
        return self._last_importance

    # ---------------------------------------------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{f"p_{k}": v for k, v in self.params.items()},
                 **{f"s_{k}": v for k, v in self.stats.items()},
                 meta=np.array(json.dumps({"H": self.H, "K": self.K, "hidden": self.hidden})))
        return path

    def load(self, path, device="cpu"):
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        if meta["H"] != self.H or meta["hidden"] != self.hidden:
            raise ValueError(f"checkpoint {path} has H={meta['H']} hidden={meta['hidden']}, config differs")
        self.params = {k[2:]: z[k] for k in z.files if k.startswith("p_")}
        self.stats = {k[2:]: z[k] for k in z.files if k.startswith("s_")}
