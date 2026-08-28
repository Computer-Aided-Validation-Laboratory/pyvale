# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Renderer result containers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class RenderResult:
    """Output from a three-dimensional renderer.

    Parameters
    ----------
    images : numpy.ndarray or None
        Image stack with shape ``(frame, camera, height, width, channel)``.
        ``None`` is used by file-only backends.
    output_paths : tuple[pathlib.Path, ...], optional
        Paths written by a backend that persists rendered images.
    """

    images: np.ndarray | None
    output_paths: tuple[Path, ...] = ()


__all__ = ["RenderResult"]
