# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Validated lifecycle for planar image-warp renderers."""

from abc import ABC, abstractmethod

from .result import ImageWarpResult
from .scene import Scene2D


class IImageWarp2D(ABC):
    """Abstract interface for planar 2D image-warp renderers.

    Implementations receive a complete :class:`Scene2D` and perform input
    verification in :meth:`verify_input` before image preparation or warping.
    """

    def render(self, scene: Scene2D) -> ImageWarpResult:
        """Validate and execute a planar image-warp request.

        Parameters
        ----------
        scene : Scene2D
            Complete planar rendering request.

        Returns
        -------
        ImageWarpResult
            Warped images, optional masks, and optional output paths.
        """
        self.verify_input(scene)
        return self._render(scene)

    @abstractmethod
    def verify_input(self, scene: Scene2D) -> None:
        """Validate a request without allocating or interpolating images.

        Parameters
        ----------
        scene : Scene2D
            Complete planar rendering request to validate.

        Raises
        ------
        ValueError
            If the scene is invalid or unsupported.
        """

    @abstractmethod
    def _render(self, scene: Scene2D) -> ImageWarpResult:
        """Perform a previously validated image-warp request.

        Parameters
        ----------
        scene : Scene2D
            Previously validated planar rendering request.

        Returns
        -------
        ImageWarpResult
            Warped images, optional masks, and optional output paths.
        """


__all__ = ["IImageWarp2D"]
