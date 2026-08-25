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
from .mesh import Mesh2D, Mesh3D


RenderMesh = Mesh3D | riley.Mesh
"""A mesh accepted by a 3D render scene.

``render.Mesh3D`` is the common mesh used by Blender and future common
backends.
``riley.Mesh`` is accepted so Riley can expose its complete native shader model
without a pyvale wrapper.
"""


@dataclass(slots=True)
class Scene3D:
    """A complete three-dimensional rendering request.

    Parameters
    ----------
    meshes : list[RenderMesh]
        Backend-compatible meshes. Blender requires common :class:`Mesh3D`
        data;
        Riley requires native :class:`riley.Mesh` data.
    cameras : list[Camera]
        One or more common perspective cameras.
    lights : list[Light] or None, optional
        Explicit scene lights. ``None`` leaves lighting to the backend.
    """

    meshes: list[RenderMesh]
    cameras: list[Camera]
    lights: list[Light] | None = None


@dataclass(frozen=True, slots=True)
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


__all__ = ["RenderMesh", "Scene3D", "Scene2D"]
