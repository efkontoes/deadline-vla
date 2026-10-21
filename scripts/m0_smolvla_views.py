"""M0 check 2: does SmolVLA accept six views with mixed tiers and dropped views, changing only the image loop?

Run on a GPU machine with LeRobot installed (pip install -e .[lerobot]):

    python scripts/m0_smolvla_views.py --config configs/default.yaml [--fixed-slots] [--out m0_smolvla.json]

For each test allocation it runs predict_chunk (inference) and forward_loss (training path) through
policy/smolvla.py, records the visual tokens each view produced and the latency, and checks:
  - the chunk has shape (H, 4) and is finite
  - each active view yields tokens, and lower tiers yield fewer tokens than higher ones
  - views at tier 0 contribute no tokens (unless --fixed-slots, the six-slot fallback)
Exit code 0 only if every check passes. Nothing here has been run yet: this script is the check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ALLOCS = {
    "all_high": [3, 3, 3, 3, 3, 3],
    "mixed": [3, 2, 1, 1, 2, 3],
    "uniform_low": [1, 1, 1, 1, 1, 1],
    "front_only": [3, 0, 0, 0, 0, 0],
    "two_off": [2, 2, 0, 2, 0, 2],
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--fixed-slots", action="store_true", help="test the six-slot fallback instead")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="m0_smolvla.json")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    import torch

    from deadlinevla.config import load_config, set_field
    from deadlinevla.policy.smolvla import SmolVLAAdapter
    from deadlinevla.policy.views import render_views

    cfg = set_field(load_config(args.config, args.overrides), "model.policy", "smolvla")
    adapter = SmolVLAAdapter(cfg, fixed_slots=args.fixed_slots)
    spec = adapter.spec()
    R = cfg.tier_res[3]
    rng = np.random.default_rng(0)
    obs = {"images": rng.integers(0, 255, (cfg.cameras.n_cams, R, R, 3), dtype=np.uint8),
           "state": np.zeros(7), "instruction": "fly to the red backpack behind you"}

    # Record the token count of every embedded view.
    vlm = adapter.policy.model.vlm_with_expert
    inner = vlm.embed_image
    seen: list[int] = []

    def counting(image):
        out = inner(image)
        seen.append(int(out.shape[-2]))
        return out

    vlm.embed_image = counting

    results, ok_all = {}, True
    for name, alloc in ALLOCS.items():
        alloc = np.array(alloc)
        rec: dict = {"alloc": alloc.tolist()}
        try:
            seen.clear()
            chunk = adapter.predict_chunk(obs, alloc)
            rec["tokens_per_view"] = list(seen)
            times = []
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                adapter.predict_chunk(obs, alloc)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            rec["latency_ms_median"] = 1000 * float(np.median(times))
            rec["chunk_shape"] = list(np.asarray(chunk).shape)
            checks = {
                "shape": tuple(np.asarray(chunk).shape) == (spec.chunk_size, 4),
                "finite": bool(np.isfinite(chunk).all()),
                "views": len(seen) == (cfg.cameras.n_cams if args.fixed_slots else int((alloc > 0).sum())),
            }
            batch = {"views": [render_views(obs["images"], alloc, cfg.tier_res)] * 2,
                     "state": np.zeros((2, 7)), "instruction": [obs["instruction"]] * 2,
                     "actions": np.zeros((2, spec.chunk_size, 4)), "alloc": np.stack([alloc, alloc])}
            loss, _ = adapter.forward_loss(batch, batch["alloc"])
            checks["loss_finite"] = bool(torch.isfinite(loss).item())
            rec["loss"] = float(loss.detach())
        except Exception as e:  # report and continue: the point is to see what breaks
            checks = {"ran": False}
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["checks"] = checks
        rec["pass"] = all(checks.values())
        ok_all &= rec["pass"]
        results[name] = rec
        print(f"{name:12s} pass={rec['pass']}  tokens/view={rec.get('tokens_per_view')}  "
              f"latency={rec.get('latency_ms_median', float('nan')):.1f} ms  {rec.get('error', '')}", flush=True)

    # Lower tiers must give fewer tokens than higher tiers (the whole premise of the budget).
    hi = results["all_high"].get("tokens_per_view") or []
    lo = results["uniform_low"].get("tokens_per_view") or []
    monotone = bool(hi and lo and max(lo) < min(hi))
    print(f"tier monotonicity (low tokens < high tokens): {monotone}")
    results["tier_monotone"] = monotone
    ok_all &= monotone or args.fixed_slots

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"M0 check 2: {'PASS' if ok_all else 'FAIL'} -> {args.out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
