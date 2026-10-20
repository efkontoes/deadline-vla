"""Runs every CLI stage on a tiny light config: the pipeline contract end to end."""

import json
from pathlib import Path

from deadlinevla.cli import main

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "light.yaml"
TINY = [
    "data.episodes=3", "train.max_steps=6", "train.log_every=3", "train.batch_size=16",
    "serve.profile_iters=3", "eval.episodes_per_cell=1", "eval.speed_caps=[5.0]",
    "eval.methods=[deadline, front_only]", "eval.governor=[true]", "scenes.families=[goal_nav, rear_threat]",
    "sim.max_episode_s=8", "cameras.tiers=[{res: 8, tokens: 16}, {res: 16, tokens: 64}, {res: 32, tokens: 256}]",
]


def test_all_stages(tmp_path):
    run = tmp_path / "run"
    assert main(["all", "--config", str(CONFIG), "--run-dir", str(run), *TINY]) == 0
    man = json.loads((run / "manifest.json").read_text())
    assert set(man["stages"]) == {"env-check", "record", "train", "export", "profile", "eval", "report"}
    assert (run / "config.resolved.yaml").exists()
    assert (run / "checkpoints" / "policy.npz").exists()
    assert (run / "latency_table.npz").exists()
    assert (run / "report" / "speed_vs_compute.png").exists()
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    assert {r["method"] for r in rows} == {"deadline", "front_only"}
    offline = json.loads((run / "offline_eval.json").read_text())
    assert set(offline) == {"all_high", "forward_only", "uniform_low", "allocator"}
    assert man["stages"]["eval"]["synthetic_latency"] is True
