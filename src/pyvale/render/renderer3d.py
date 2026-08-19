# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""The mandatory validated lifecycle for 3D renderers."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .camera import Camera
from .light import Light
from .mesh import Mesh
from .result import RenderResult


class IRenderer3D(ABC):
    """Abstract 3D renderer with validation before all expensive work."""

    def render(
        self,
        meshes: Sequence[Mesh],
        cameras: Sequence[Camera],
        lights: Sequence[Light] | None = None,
    ) -> RenderResult:
        """Validate a scene then render it."""
        render_plan = self.verify_input(meshes, cameras, lights)
        return self._render(render_plan)

    @abstractmethod
    def verify_input(
        self,
        meshes: Sequence[Mesh],
        cameras: Sequence[Camera],
        lights: Sequence[Light] | None = None,
    ) -> object:
        """Validate inputs without expensive scene or image preparation."""

    @abstractmethod
    def _render(self, render_plan: object) -> RenderResult:
        """Render a plan that has already passed validation."""


__all__ = ["IRenderer3D"]
