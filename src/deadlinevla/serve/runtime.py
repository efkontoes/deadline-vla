"""Inference runtimes. All expose ``infer(obs, alloc) -> float[H, 4]`` and ``timings() -> dict``.

SimulatedRuntime  runs the policy in-process for the *action* and samples the *latency* from a board
                  model (a synthetic board, or the table measured on the real Jetson). This is how
                  the simulated loop sees the onboard computer's latency instead of the desktop's
                  (Risks: hardware-in-the-loop latency; decision: injected delay in sim).
TorchRuntime      PyTorch reference inference, wall-clock timed.
TrtRuntime        TensorRT vision engines per resolution tier on the board; buffers allocated once.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from deadlinevla.budget.latency_model import LatencySource
from deadlinevla.config import Config
from deadlinevla.policy.base import PolicyAdapter


class SimulatedRuntime:
    def __init__(self, adapter: Optional[PolicyAdapter], source: LatencySource, rng: np.random.Generator,
                 latency_only: bool = False):
        self.adapter, self.source, self.rng = adapter, source, rng
        self.latency_only = latency_only or adapter is None
        self._t: dict[str, float] = {}

    def infer(self, obs: Any, alloc: np.ndarray) -> Optional[np.ndarray]:
        self._t = self.source.sample(alloc, self.rng)
        if self.latency_only:
            return None
        return self.adapter.predict_chunk(obs, alloc)

    def timings(self) -> dict[str, float]:
        return dict(self._t)


class TorchRuntime:
    def __init__(self, adapter: PolicyAdapter):
        self.adapter = adapter
        self._t: dict[str, float] = {}

    def infer(self, obs, alloc):
        sync = _cuda_sync()
        sync()
        t0 = time.perf_counter()
        chunk = self.adapter.predict_chunk(obs, alloc)
        sync()
        self._t = {"total": time.perf_counter() - t0}
        return chunk

    def timings(self):
        return dict(self._t)


def _cuda_sync():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.synchronize
    except ImportError:
        pass
    return lambda: None


class TrtRuntime:  # pragma: no cover - needs TensorRT on a Jetson / NVIDIA GPU
    """One TensorRT vision engine per tier; backbone prefill and action head run in fp16 PyTorch.

    The vision encoder is the part whose cost depends on the allocation, so it is the part compiled
    per resolution tier. Exporting the backbone and the K-step head to TensorRT as well is an M3 task
    once the upstream graph is pinned; ``timings`` already separates ``encode`` from ``backbone``.
    """

    def __init__(self, adapter: PolicyAdapter, engine_dir: str | Path):
        import tensorrt as trt
        import torch

        self.trt, self.torch = trt, torch
        self.adapter = adapter
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engines: dict[int, Any] = {}
        self.contexts: dict[int, Any] = {}
        self.buffers: dict[int, dict[str, Any]] = {}
        self.stream = torch.cuda.Stream()
        self._t: dict[str, float] = {}
        self.load(engine_dir, "cuda")

    def load(self, engines: str | Path, device: str = "cuda") -> None:
        trt, torch = self.trt, self.torch
        meta = json.loads((Path(engines) / "export.json").read_text())
        runtime = trt.Runtime(self.logger)
        for tier_s, path in meta["vision_engines"].items():
            tier = int(tier_s)
            eng = runtime.deserialize_cuda_engine(Path(path).read_bytes())
            ctx = eng.create_execution_context()
            bufs = {}
            for i in range(eng.num_io_tensors):
                name = eng.get_tensor_name(i)
                shape = tuple(eng.get_tensor_shape(name))
                dtypes = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32}
                dtype = dtypes[eng.get_tensor_dtype(name)]
                bufs[name] = torch.empty(shape, dtype=dtype, device=device)
                ctx.set_tensor_address(name, bufs[name].data_ptr())
            self.engines[tier], self.contexts[tier], self.buffers[tier] = eng, ctx, bufs

    def _encode(self, image: np.ndarray, tier: int):
        torch = self.torch
        bufs = self.buffers[tier]
        x = torch.from_numpy(image).to("cuda").permute(2, 0, 1)[None].to(bufs["image"].dtype) / 255.0
        bufs["image"].copy_(x * 2.0 - 1.0)
        self.contexts[tier].execute_async_v3(self.stream.cuda_stream)
        return bufs["tokens"].clone()

    def infer(self, obs, alloc):
        from deadlinevla.policy.views import render_views

        torch = self.torch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        views = render_views(obs["images"], alloc, self.adapter.cfg.tier_res)
        t1 = time.perf_counter()
        tokens = [self._encode(rv.image, rv.tier) for rv in views]
        self.stream.synchronize()
        t2 = time.perf_counter()
        self.adapter._precomputed = tokens
        try:
            chunk = self.adapter.predict_chunk(obs, alloc)
        finally:
            self.adapter._precomputed = None
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        self._t = {"fixed": t1 - t0, "encode": t2 - t1, "backbone": t3 - t2, "total": t3 - t0}
        return chunk

    def timings(self):
        return dict(self._t)


def export_engines(adapter: PolicyAdapter, cfg: Config, out_dir: str | Path) -> Path:
    """Export per-tier vision engines (TensorRT) or record that the runtime is simulated."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"policy": adapter.name, "precision": cfg.serve.precision, "runtime": cfg.serve.runtime}
    if cfg.serve.runtime == "simulated" or not hasattr(adapter, "export_vision_onnx"):
        meta["note"] = "no engines: inference runs in-process and latency comes from the board model"
    else:  # pragma: no cover - needs torch + onnx (+ tensorrt to build)
        onnx_paths = {r: adapter.export_vision_onnx(out / f"vision_t{r}.onnx", r) for r in (1, 2, 3)}
        meta["vision_onnx"] = {str(r): str(p) for r, p in onnx_paths.items()}
        try:
            meta["vision_engines"] = {str(r): str(build_engine(p, out / f"vision_t{r}.engine", cfg.serve.precision))
                                      for r, p in onnx_paths.items()}
        except ImportError:
            meta["build_on_board"] = [f"trtexec --onnx={p} --saveEngine={out / f'vision_t{r}.engine'} --fp16"
                                      for r, p in onnx_paths.items()]
    path = out / "export.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def build_engine(onnx_path: Path, engine_path: Path, precision: str = "fp16") -> Path:  # pragma: no cover
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    engine_path.write_bytes(builder.build_serialized_network(network, config))
    return engine_path


def make_runtime(cfg: Config, adapter: Optional[PolicyAdapter], source: Optional[LatencySource],
                 rng: np.random.Generator, engine_dir: Optional[str | Path] = None, latency_only: bool = False):
    kind = cfg.serve.runtime
    if kind == "simulated":
        if source is None:
            raise ValueError("simulated runtime needs a latency source")
        return SimulatedRuntime(adapter, source, rng, latency_only)
    if kind == "torch":
        return TorchRuntime(adapter)
    if kind == "trt":
        if engine_dir is None:
            raise ValueError("trt runtime needs the export directory")
        return TrtRuntime(adapter, engine_dir)
    raise ValueError(f"unknown serve.runtime {kind!r}")
