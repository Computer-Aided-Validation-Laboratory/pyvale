"""Compact filesystem sink for identification diagnostic arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class DiagnosticArtifactWriter:
    """Write callback events as JSON metadata plus compressed NumPy arrays."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._counts: dict[str, int] = {}

    def __deepcopy__(self, memo):
        # Runtime phase components are deep-copied together. All copies must
        # still write to the one run-scoped artefact directory.
        return self

    def __call__(self, kind: str, payload: dict[str, object]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        index = self._counts.get(kind, 0)
        self._counts[kind] = index + 1
        stem = f"{kind}_{index:03d}"
        arrays: dict[str, np.ndarray] = {}
        metadata = _extract_arrays(payload, arrays, prefix="payload")
        (self.root / f"{stem}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if arrays:
            np.savez_compressed(self.root / f"{stem}.npz", **arrays)


def _extract_arrays(value: Any, arrays: dict[str, np.ndarray], *, prefix: str):
    if isinstance(value, np.ndarray):
        key = prefix.replace(".", "__")
        arrays[key] = value
        return {"array_key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {
            str(key): _extract_arrays(item, arrays, prefix=f"{prefix}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _extract_arrays(item, arrays, prefix=f"{prefix}.{index}")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)
