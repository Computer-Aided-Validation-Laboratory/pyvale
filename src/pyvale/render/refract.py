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
    """Placeholder until Refract exposes a stable Python rendering API."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def verify_input(self, meshes: Sequence[Mesh], cameras: Sequence[Camera],
                     lights: Sequence[Light] | None = None) -> object:
        raise NotImplementedError("Refract integration is not available yet.")

    def _render(self, render_plan: object) -> RenderResult:
        raise NotImplementedError("Refract integration is not available yet.")


__all__ = ["Refract"]
