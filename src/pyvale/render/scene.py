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


@dataclass(frozen=True, slots=True)
class RenderScene:
    """A complete three-dimensional rendering request.

    Parameters
    ----------
    meshes : tuple[RenderMesh, ...]
        Backend-compatible meshes. Blender requires common :class:`Mesh3D`
        data;
        Riley requires native :class:`riley.Mesh` data.
    cameras : tuple[Camera, ...]
        One or more common perspective cameras.
    lights : tuple[Light, ...] or None, optional
        Explicit scene lights. ``None`` leaves lighting to the backend.
    """

    meshes: tuple[RenderMesh, ...]
    cameras: tuple[Camera, ...]
    lights: tuple[Light, ...] | None = None

    def __post_init__(self) -> None:
        """Store immutable scene sequences for validation and rendering."""
        object.__setattr__(self, "meshes", tuple(self.meshes))
        object.__setattr__(self, "cameras", tuple(self.cameras))
        if self.lights is not None:
            object.__setattr__(self, "lights", tuple(self.lights))


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

    def __post_init__(self) -> None:
        """Validate scene components."""
        if not isinstance(self.mesh, Mesh2D):
            raise TypeError("Scene2D.mesh must be a render.Mesh2D.")
        if not isinstance(self.camera, Camera2D):
            raise TypeError("Scene2D.camera must be a render.Camera2D.")


__all__ = ["RenderMesh", "RenderScene", "Scene2D"]
