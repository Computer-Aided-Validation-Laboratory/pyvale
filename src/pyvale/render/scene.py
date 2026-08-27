# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Concrete scene input for three-dimensional and planar renderers."""

from dataclasses import dataclass

import numpy as np
import riley

from .camera import Camera, Camera2D
from .light import Light
from .mesh import Mesh2D


@dataclass(slots=True)
class Scene3D:
    """A complete three-dimensional rendering request.

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


@dataclass(slots=True)
class Scene2D:
    """A complete planar image-warp rendering request.

    Parameters
    ----------
    mesh : Mesh2D
        Planar finite-element mesh with displacement data.
    camera : Camera2D
        Orthographic camera defining the image plane.
    source_image : numpy.ndarray or None, optional
        Reference image to deform. Required by ``ImageDef2D``, unused by
        analytic renderers (``PixIntGrid2D``, ``PixIntSpeck2D``).
    """

    mesh: Mesh2D
    camera: Camera2D
    source_image: np.ndarray | None = None


__all__ = ["Scene2D", "Scene3D"]
