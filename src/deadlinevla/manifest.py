"""manifest.json: how pipeline stages hand outputs to each other.

Every CLI stage records what it produced; the next stage reads it from here instead of guessing paths.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_MISSING = object()


class Manifest:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "manifest.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"stages": {}}

    def record(self, stage: str, **outputs: Any) -> None:
        self.data["stages"][stage] = {**outputs, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.save()

    def has(self, stage: str, key: str | None = None) -> bool:
        st = self.data["stages"].get(stage)
        return st is not None and (key is None or key in st)

    def get(self, stage: str, key: str, default: Any = _MISSING) -> Any:
        st = self.data["stages"].get(stage, {})
        if key in st:
            return st[key]
        if default is not _MISSING:
            return default
        raise KeyError(f"manifest has no {stage}.{key}; run the '{stage}' stage first ({self.path})")

    def path_of(self, stage: str, key: str) -> Path:
        """Paths are stored relative to the run dir so a run folder can be moved."""
        return self.run_dir / self.get(stage, key)

    def rel(self, path: str | Path) -> str:
        p = Path(path)
        try:
            return p.resolve().relative_to(self.run_dir.resolve()).as_posix()
        except ValueError:
            return str(p)

    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
