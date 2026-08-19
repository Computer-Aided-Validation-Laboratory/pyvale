# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Validated lifecycle for planar image-warp renderers."""

from abc import ABC, abstractmethod

from .result import ImageWarpResult


class IImageWarp2D(ABC):
    """Abstract interface for validated planar image-warp renderers.

    Implementations define their own request arguments, then perform cheap
    validation in :meth:`verify_input` before image preparation or warping.
    """

    def render(self, *args: object, **kwargs: object) -> ImageWarpResult:
        """Validate and execute a planar image-warp request.

        Parameters
        ----------
        *args : object
            Positional arguments accepted by the concrete image-warp backend.
        **kwargs : object
            Keyword arguments accepted by the concrete image-warp backend.

        Returns
        -------
        ImageWarpResult
            Warped images and optional masks.
        """
        render_plan = self.verify_input(*args, **kwargs)
        return self._render(render_plan)

    @abstractmethod
    def verify_input(self, *args: object, **kwargs: object) -> object:
        """Validate a request without allocating or interpolating images.

        Returns
        -------
        object
            An opaque, backend-specific plan consumed by :meth:`_render`.
        """

    @abstractmethod
    def _render(self, render_plan: object) -> ImageWarpResult:
        """Perform a previously validated image-warp request.

        Parameters
        ----------
        render_plan : object
            Backend-specific plan returned by :meth:`verify_input`.

        Returns
        -------
        ImageWarpResult
            Warped images and optional masks.
        """


__all__ = ["IImageWarp2D"]
