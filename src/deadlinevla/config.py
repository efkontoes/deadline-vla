"""Typed, frozen configuration.

One YAML file is parsed into :class:`Config` with one sub-config per section. Unknown keys are
rejected, ``section.field=value`` overrides are applied on top, and the resolved config is written
next to every output as ``config.resolved.yaml``.

A YAML file may start with ``base: other.yaml`` to inherit from another file (resolved relative to
itself); its own keys are merged over the base section by section.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RunConfig:
    out_dir: str = "runs/default"
    seed: int = 0


@dataclass(frozen=True)
class SimConfig:
    backend: str = "isaac"
    num_envs: int = 1
    physics_hz: int = 50
    max_episode_s: float = 40.0
    arena: float = 30.0
    altitude: float = 1.5
    goal_radius: float = 1.0
    max_range: float = 20.0
    render: bool = True       # False: skip cameras (only for policies that need no images)


@dataclass(frozen=True)
class DroneConfig:
    a_brake: Union[float, str] = "measured"
    tau_track: Union[float, str] = "measured"   # velocity-tracking time constant of the controller
    v_max: float = 8.0
    margin: float = 0.5
    ctrl_hz: float = 10.0
    radius: float = 0.25
    tau_vel: float = 0.1
    a_max: float = 9.0
    yaw_rate_max: float = 2.0


@dataclass(frozen=True)
class TierSpec:
    res: int
    tokens: int


@dataclass(frozen=True)
class CamerasConfig:
    n_cams: int = 6
    cam_yaws: tuple[float, ...] = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
    model: str = "pinhole"
    fov: float = 90.0
    tiers: tuple[TierSpec, ...] = (TierSpec(128, 16), TierSpec(256, 64), TierSpec(512, 256))


@dataclass(frozen=True)
class ScenesConfig:
    families: tuple[str, ...] = ("goal_nav", "sparse_nav", "rear_threat", "slalom")
    clutter_density: float = 0.03
    seed: int = 0
    goal_dist: tuple[float, ...] = (8.0, 18.0)
    obstacle_radius: tuple[float, ...] = (0.3, 0.8)
    threat_speed: float = 3.0


@dataclass(frozen=True)
class ExpertConfig:
    planner: str = "astar"
    speed_profile: bool = True
    grid_res: float = 0.25
    lookahead: float = 1.5
    assumed_latency: float = 0.1
    cruise_speed: Optional[float] = None


@dataclass(frozen=True)
class DataConfig:
    episodes: int = 200
    repo_id: str = "local/deadlinevla_demos"
    format: str = "auto"
    val_frac: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    policy: str = "smolvla"
    pretrained: str = "lerobot/smolvla_base"
    chunk_size: Optional[int] = None
    num_denoise_steps: Optional[int] = None
    device: str = "cuda"
    hidden: int = 256
    surrogate_ranges: tuple[float, ...] = (0.0, 4.0, 8.0, 15.0)


@dataclass(frozen=True)
class TrainConfig:
    alloc_sampler: str = "mixed"
    p_view_off: float = 0.15
    lr: float = 1e-4
    batch_size: int = 32
    max_steps: int = 20000
    log_every: int = 50


@dataclass(frozen=True)
class BudgetConfig:
    kappa: float = 2.0
    epsilon: float = 0.05
    quality_fn: str = "log1p_tokens"
    cone_deg: float = 30.0
    clearance_source: str = "oracle"
    latency_table: str = "measured"
    latency_stat: str = "p95"
    hover_speed: float = 0.3
    max_residual: float = 0.005
    floor_ignore_z: float = 0.2
    objective: str = "speed"          # speed (v2: cheapest allocation that sees enough) | quality (v1)
    qmin_mid_weight: float = 0.25     # views with w_j >= this need mid tier in Q_min, the rest low
    governor_delays: bool = True      # governor blind time = L + control period + tracking time constant


@dataclass(frozen=True)
class ServeConfig:
    mode: str = "async"
    precision: str = "fp16"
    runtime: str = "trt"
    board: str = "jetson_orin_nx_16gb"
    profile_iters: int = 20


@dataclass(frozen=True)
class EvalConfig:
    methods: tuple[str, ...] = ("deadline", "deadline_v1", "all_high", "uniform_fixed", "importance", "front_only")
    governor: tuple[bool, ...] = (True, False)
    episodes_per_cell: int = 10
    speed_caps: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 10.0)
    uniform_tier: int = 2
    collision_threshold: float = 1.0
    seed_offset: int = 10000
    workers: int = 1                  # parallel processes for the eval sweep


@dataclass(frozen=True)
class GatesConfig:
    max_violation_rate: Optional[float] = None
    max_success_drop: float = 0.05
    baseline: Optional[str] = None


@dataclass(frozen=True)
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    cameras: CamerasConfig = field(default_factory=CamerasConfig)
    scenes: ScenesConfig = field(default_factory=ScenesConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    gates: GatesConfig = field(default_factory=GatesConfig)

    # Convenience views used all over the code.
    @property
    def tier_res(self) -> tuple[int, ...]:
        """Resolution per tier index 0..3 (0 for off)."""
        return (0,) + tuple(t.res for t in self.cameras.tiers)

    @property
    def tier_tokens(self) -> tuple[int, ...]:
        """Visual tokens per tier index 0..3 (n(0) = 0)."""
        return (0,) + tuple(t.tokens for t in self.cameras.tiers)

    @property
    def dt(self) -> float:
        return 1.0 / self.drone.ctrl_hz


# --------------------------------------------------------------------------------------------------
# Building dataclasses from plain dicts


def _coerce(value: Any, hint: Any, where: str) -> Any:
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if hint is Any:
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(value, arg, where)
            except ConfigError as e:
                errors.append(str(e))
        raise ConfigError(f"{where}: {value!r} matches none of {hint}: {errors}")
    if is_dataclass(hint):
        if isinstance(value, hint):
            return value
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        return _build(hint, value, where)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        item = args[0] if args else Any
        return tuple(_coerce(v, item, f"{where}[{i}]") for i, v in enumerate(value))
    if hint is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected bool, got {value!r}")
    if hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected int, got {value!r}")
    if hint is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise ConfigError(f"{where}: expected float, got {value!r}")
    if hint is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{where}: expected str, got {value!r}")
    return value


def _build(cls: type, raw: dict, where: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown key(s) in {where or 'config'}: {sorted(unknown)}")
    kwargs = {k: _coerce(v, hints[k], f"{where}.{k}" if where else k) for k, v in raw.items()}
    return cls(**kwargs)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def to_dict(cfg: Config) -> dict:
    return _to_plain(cfg)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    base = raw.pop("base", None)
    if base is not None:
        raw = _deep_merge(_read_yaml((path.parent / base).resolve()), raw)
    return raw


def _apply_override(raw: dict, override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"override must look like section.field=value, got {override!r}")
    key, text = override.split("=", 1)
    parts = key.strip().split(".")
    if len(parts) < 2:
        raise ConfigError(f"override key must be section.field, got {key!r}")
    node = raw
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {key!r} goes through a non-mapping")
    node[parts[-1]] = yaml.safe_load(text)


def load_config(path: Optional[str | Path] = None, overrides: Sequence[str] = ()) -> Config:
    """Parse YAML (or defaults), apply ``key.sub=value`` overrides, validate and freeze."""
    raw = _read_yaml(Path(path)) if path is not None else {}
    for ov in overrides:
        _apply_override(raw, ov)
    cfg = _build(Config, raw)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if len(cfg.cameras.cam_yaws) != cfg.cameras.n_cams:
        raise ConfigError("cameras.cam_yaws must have n_cams entries")
    if len(cfg.cameras.tiers) != 3:
        raise ConfigError("cameras.tiers must list exactly tiers 1, 2 and 3")
    toks = cfg.tier_tokens
    if any(b <= a for a, b in zip(toks, toks[1:])):
        raise ConfigError("tier token counts must strictly increase")
    if cfg.sim.backend not in ("isaac", "light"):
        raise ConfigError(f"sim.backend must be isaac or light, got {cfg.sim.backend!r}")
    if cfg.budget.objective not in ("speed", "quality"):
        raise ConfigError("budget.objective must be speed or quality")
    if not cfg.sim.render and cfg.budget.clearance_source == "depth":
        raise ConfigError("budget.clearance_source=depth needs sim.render=true")
    if not cfg.sim.render and cfg.model.policy not in ("surrogate",):
        raise ConfigError(f"model.policy={cfg.model.policy} needs images: set sim.render=true")
    if cfg.budget.clearance_source not in ("oracle", "depth"):
        raise ConfigError("budget.clearance_source must be oracle or depth")
    if cfg.serve.mode not in ("sync", "async"):
        raise ConfigError("serve.mode must be sync or async")
    if cfg.train.alloc_sampler not in ("uniform", "allocator", "mixed"):
        raise ConfigError("train.alloc_sampler must be uniform, allocator or mixed")
    for name in ("a_brake", "tau_track"):
        v = getattr(cfg.drone, name)
        if isinstance(v, str) and v != "measured":
            raise ConfigError(f"drone.{name} must be a number or 'measured'")
    if len(cfg.model.surrogate_ranges) != 4:
        raise ConfigError("model.surrogate_ranges needs one range per tier 0..3")


def set_field(cfg: Any, dotted: str, value: Any) -> Any:
    """Return a copy of a frozen config with one nested field replaced."""
    head, _, rest = dotted.partition(".")
    if not rest:
        return dataclasses.replace(cfg, **{head: value})
    return dataclasses.replace(cfg, **{head: set_field(getattr(cfg, head), rest, value)})


def a_brake(cfg: Config) -> float:
    """The measured braking deceleration; fails loudly if it was never resolved."""
    a = cfg.drone.a_brake
    if isinstance(a, str):
        raise ConfigError("drone.a_brake is still 'measured': run env-check first or resolve it")
    return float(a)


def tau_track(cfg: Config) -> float:
    """The measured velocity-tracking time constant; fails loudly if it was never resolved."""
    t = cfg.drone.tau_track
    if isinstance(t, str):
        raise ConfigError("drone.tau_track is still 'measured': run env-check first or resolve it")
    return float(t)


def save_resolved(cfg: Config, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "config.resolved.yaml"
    path.write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False), encoding="utf-8")
    return path
