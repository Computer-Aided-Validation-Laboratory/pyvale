# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Concrete scene input for three dimensional and planar renderers."""

from dataclasses import dataclass, field

import riley

from .camera import Camera
from .light import Light
from .mesh import Mesh3D


@dataclass(slots=True)
class Scene3D:
    """A complete three dimensional rendering request.

    Parameters
    ----------
    meshes : list[object]
        Meshes accepted by the selected backend.
    cameras : list[Camera | riley.Camera]
        One or more common perspective cameras. Native ``riley.Camera``
        instances are passed through to the Riley renderer unchanged.
    lights : list[Light] or None, optional
        Explicit scene lights. ``None`` leaves lighting to the backend.
    """

    meshes: list[object]
    cameras: list[Camera | riley.Camera]
    lights: list[Light] | None = None


__all__ = ["Scene3D"]
