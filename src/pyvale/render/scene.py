# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Concrete scene input for three-dimensional renderers."""

from dataclasses import dataclass

import riley

from .camera import Camera
from .light import Light
from .mesh import Mesh


RenderMesh = Mesh | riley.Mesh
"""A mesh accepted by a render scene.

``render.Mesh`` is the common mesh used by Blender and future common backends.
``riley.Mesh`` is accepted so Riley can expose its complete native shader model
without a pyvale wrapper.
"""


@dataclass(frozen=True, slots=True)
class RenderScene:
    """A complete three-dimensional rendering request.

    Parameters
    ----------
    meshes : tuple[RenderMesh, ...]
        Backend-compatible meshes. Blender requires common :class:`Mesh` data;
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


__all__ = ["RenderMesh", "RenderScene"]
