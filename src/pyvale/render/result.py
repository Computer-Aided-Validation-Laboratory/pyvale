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
    """Output from a three dimensional renderer.

    Parameters
    ----------
    images : np.ndarray or None
        Image stack with shape
        ``(num_frames, num_cameras, height, width, num_channels)``
        with a backend-specific dtype. Riley returns ``float64`` images; when
        camera image sizes or channel counts differ, Riley pads to the largest
        height, width and channel count with each camera's background value.
        Use ``images[:, camera_index, :height, :width, :channels]`` to select
        that camera's rendered pixels. ``None`` indicates no in-memory output.
    output_paths : tuple[Path, ...], optional
        Paths written by a backend that persists rendered images.
    """

    images: np.ndarray | None
    output_paths: tuple[Path, ...] = ()


__all__ = ["RenderResult"]
