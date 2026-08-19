# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Renderer result containers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class RenderResult:
    """3D render output using ``(frame, camera, height, width, channel)``."""

    images: np.ndarray | None
    output_paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class ImageWarpResult:
    """Planar image-warp output."""

    images: np.ndarray
    masks: np.ndarray | None = None


__all__ = ["ImageWarpResult", "RenderResult"]
