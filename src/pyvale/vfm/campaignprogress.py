"""Durable progress helpers for long-running VFM campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np


@dataclass(slots=True)
class ProgressEstimate:
    completed: int
    total: int
    elapsed_seconds: float
    eta_seconds: float | None

    @classmethod
    def from_counts(cls, completed: int, total: int, started: float):
        elapsed = time.monotonic() - started
        eta = None
        if completed > 0 and total > completed:
            eta = elapsed / completed * (total - completed)
        return cls(completed, total, elapsed, eta)

    def line(self, *, prefix: str = "progress") -> str:
        eta = "unknown" if self.eta_seconds is None else _duration(self.eta_seconds)
        return (
            f"{prefix} complete={self.completed}/{self.total} "
            f"elapsed={_duration(self.elapsed_seconds)} eta={eta}"
        )


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {"updated_at": datetime.now().astimezone().isoformat(), **payload}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(enriched, stream, indent=2, default=_json_default)
        stream.write("\n")
        stream.flush()
    temporary.replace(path)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def _duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 120.0:
        return f"{seconds:.0f}s"
    if seconds < 7200.0:
        return f"{seconds / 60.0:.1f}min"
    return f"{seconds / 3600.0:.1f}h"
