# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Reserved public entry point for the future Refract backend."""

from collections.abc import Sequence

from .camera import Camera
from .light import Light
from .mesh import Mesh
from .renderer3d import IRenderer3D
from .result import RenderResult


class Refract(IRenderer3D):
    """Placeholder until Refract exposes a stable Python rendering API.

    Notes
    -----
    Refract will implement :class:`IRenderer3D` once its backend and shader
    contracts are available. This public stub cannot yet render a scene.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reserve construction arguments for the future Refract backend.

        Parameters
        ----------
        *args : object
            Future positional configuration arguments.
        **kwargs : object
            Future keyword configuration arguments.
        """
        pass

    def verify_input(self, meshes: Sequence[Mesh], cameras: Sequence[Camera],
                     lights: Sequence[Light] | None = None) -> object:
        """Raise until Refract's input contract is implemented.

        Parameters
        ----------
        meshes : Sequence[Mesh]
            Future scene meshes.
        cameras : Sequence[Camera]
            Future scene cameras.
        lights : Sequence[Light] or None, optional
            Future scene lights.

        Raises
        ------
        NotImplementedError
            Always, because Refract is a public stub.
        """
        raise NotImplementedError("Refract integration is not available yet.")

    def _render(self, render_plan: object) -> RenderResult:
        """Raise until the Refract renderer is implemented.

        Raises
        ------
        NotImplementedError
            Always, because Refract is a public stub.
        """
        raise NotImplementedError("Refract integration is not available yet.")


__all__ = ["Refract"]
