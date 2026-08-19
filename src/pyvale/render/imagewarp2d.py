# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Validated lifecycle for planar image-warp renderers."""

from abc import ABC, abstractmethod

from .result import ImageWarpResult


class IImageWarp2D(ABC):
    """Abstract planar image-warp interface."""

    def render(self, *args: object, **kwargs: object) -> ImageWarpResult:
        """Validate a request before preparation and image deformation."""
        render_plan = self.verify_input(*args, **kwargs)
        return self._render(render_plan)

    @abstractmethod
    def verify_input(self, *args: object, **kwargs: object) -> object:
        """Check inputs without image allocation or interpolation."""

    @abstractmethod
    def _render(self, render_plan: object) -> ImageWarpResult:
        """Perform the validated warp."""


__all__ = ["IImageWarp2D"]
