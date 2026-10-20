"""CI gates: fail on too many deadline violations or a success drop against a stored baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from deadlinevla.config import Config


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    checked: dict[str, Any] = field(default_factory=dict)


def check_gates(metrics: dict[str, Any], baseline: Optional[dict[str, Any]], cfg: Config) -> GateResult:
    """``metrics`` / ``baseline``: summary dicts of the deadline pilot (with governor)."""
    g = cfg.gates
    reasons, checked = [], {}
    vr = metrics.get("violation_rate")
    if g.max_violation_rate is not None:
        checked["violation_rate"] = (vr, g.max_violation_rate)
        if vr is not None and vr > g.max_violation_rate:
            reasons.append(f"violation rate {vr:.4f} > max {g.max_violation_rate:.4f}")
    if baseline is not None:
        s, s0 = metrics.get("success_rate"), baseline.get("success_rate")
        checked["success_rate"] = (s, s0, g.max_success_drop)
        if s is not None and s0 is not None and s < s0 - g.max_success_drop:
            reasons.append(f"success {s:.3f} dropped more than {g.max_success_drop} below baseline {s0:.3f}")
    return GateResult(passed=not reasons, reasons=reasons, checked=checked)
