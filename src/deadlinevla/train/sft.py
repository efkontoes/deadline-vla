"""Method step 6: budget-randomised fine-tuning.

    L(theta) = E_{(o, l, A) ~ D} E_{r ~ p(r)} [ L_FM(theta; Phi_r(o), l, A) ]

Not robustness augmentation: this is what makes a per-tick allocation legal at test time. Without
it, every change of allocation is a distribution shift.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from deadlinevla.budget.allocator import camera_axes, choose_allocation, quality_table, view_weights
from deadlinevla.budget.deadline import max_latency
from deadlinevla.budget.latency_model import LatencyModel, synthetic_prior
from deadlinevla.config import Config, a_brake
from deadlinevla.data.dataset import DemoDataset, collate, load_dataset, make_dataloader
from deadlinevla.policy.base import PolicyAdapter, make_policy


def _per_group(per: np.ndarray, allocs: np.ndarray, acc: dict[str, list[float]]) -> None:
    """Accumulate loss per number of active views and (view-weighted) per tier."""
    active = (allocs > 0).sum(1)
    for k in np.unique(active):
        acc.setdefault(f"loss_active{k}", []).extend(per[active == k].tolist())
    for r in (0, 1, 2, 3):
        frac = (allocs == r).mean(1)
        if frac.sum() > 0:
            acc.setdefault(f"loss_tier{r}", []).append(float((per * frac).sum() / frac.sum()))


def train_sft(cfg: Config, run_dir: str | Path, dataset_root: str | Path,
              latency: Optional[LatencyModel] = None, log: Callable[[str], Any] = print) -> Path:
    """Fine-tune with a fresh allocation per sample; returns the checkpoint path."""
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    adapter = make_policy(cfg)
    if not adapter.trainable:
        log(f"policy {adapter.name} is not trained here; writing a placeholder checkpoint")
        return adapter.save(ckpt_dir / adapter.checkpoint_name)

    train = load_dataset(cfg, dataset_root, "train", adapter.spec().chunk_size)
    val = load_dataset(cfg, dataset_root, "val", adapter.spec().chunk_size)
    log(f"train: {len(train.episodes)} episodes / {len(train)} samples; val: {len(val.episodes)} episodes")
    adapter.fit_stats(train)
    rng = np.random.default_rng(cfg.run.seed)
    loader = make_dataloader(train, cfg, rng=rng, latency=latency)

    rows: list[dict[str, Any]] = []
    acc: dict[str, list[float]] = {}
    losses: list[float] = []
    for step in range(1, cfg.train.max_steps + 1):
        batch = next(loader)
        loss, m = adapter.train_step(batch)
        losses.append(loss)
        if "per_sample" in m:
            _per_group(np.asarray(m["per_sample"]), batch["alloc"], acc)
        if step % cfg.train.log_every == 0 or step == cfg.train.max_steps:
            row = {"step": step, "loss": float(np.mean(losses))}
            row.update({k: float(np.mean(v)) for k, v in sorted(acc.items())})
            rows.append(row)
            log(f"  step {step:6d}  loss {row['loss']:.4f}")
            losses, acc = [], {}

    run_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "step", k != "loss", k))
    with open(run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    path = adapter.save(ckpt_dir / adapter.checkpoint_name)

    if len(val):
        grid = offline_eval(adapter, val, default_alloc_grid(cfg), latency)
        (run_dir / "offline_eval.json").write_text(json.dumps(grid, indent=2), encoding="utf-8")
        log("  offline eval: " + ", ".join(f"{k}={v['loss']:.4f}" for k, v in grid.items()))
    return path


def default_alloc_grid(cfg: Config) -> dict[str, Any]:
    V = cfg.cameras.n_cams
    return {"all_high": [3] * V, "forward_only": [3] + [0] * (V - 1), "uniform_low": [1] * V,
            "allocator": "allocator"}


def offline_eval(adapter: PolicyAdapter, val: DemoDataset, allocs: dict[str, Any],
                 latency: Optional[LatencyModel] = None, batch_size: int = 64,
                 max_samples: int = 2048) -> dict[str, dict[str, float]]:
    """Action error against the expert at a fixed grid of allocations. A cheap check before rollouts.

    ``"allocator"`` means: per sample, the allocation the deadline allocator would pick from that
    sample's speed and recorded clearance.
    """
    cfg = adapter.cfg
    lat = latency or synthetic_prior(cfg.serve.board, cfg.tier_tokens, cfg.cameras.n_cams)
    q = quality_table(cfg.budget.quality_fn, cfg.tier_tokens)
    axes = camera_axes(cfg.cameras.cam_yaws)
    idx = np.arange(len(val))
    if len(idx) > max_samples:
        idx = np.random.default_rng(0).choice(idx, max_samples, replace=False)
    out = {}
    for name, alloc in allocs.items():
        per_all, tokens = [], []
        for s in range(0, len(idx), batch_size):
            items = [val[int(i)] for i in idx[s:s + batch_size]]
            if isinstance(alloc, str):
                rows = []
                for it in items:
                    v_body = np.asarray(it["state"][:3], dtype=np.float64)
                    speed = float(np.linalg.norm(v_body))
                    L_max = max_latency(speed, it["clearance"], a_brake(cfg), cfg.drone.margin)
                    w = view_weights(v_body, axes, cfg.budget.kappa, cfg.budget.epsilon, cfg.budget.hover_speed)
                    rows.append(choose_allocation(w, L_max, lat, q, cfg.budget.objective,
                                                  cfg.budget.qmin_mid_weight)[0])
                A = np.stack(rows)
            else:
                A = np.tile(np.asarray(alloc), (len(items), 1))
            _, m = adapter.forward_loss(collate(items, A, cfg), A)
            per_all.extend(np.asarray(m["per_sample"]).tolist())
            tokens.extend(np.asarray(cfg.tier_tokens)[A].sum(1).tolist())
        out[name] = {"loss": float(np.mean(per_all)), "tokens_per_sample": float(np.mean(tokens))}
    return out
