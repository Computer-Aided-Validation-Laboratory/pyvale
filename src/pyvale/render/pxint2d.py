# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Reserved public entry point for future planar pixel integration."""

from .imagewarp2d import IImageWarp2D
from .result import ImageWarpResult


class PxInt2D(IImageWarp2D):
    """Placeholder for the future Grid2D and Speck2D image-warp workflow."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def verify_input(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError("PxInt2D integration is not available yet.")

    def _render(self, render_plan: object) -> ImageWarpResult:
        raise NotImplementedError("PxInt2D integration is not available yet.")


__all__ = ["PxInt2D"]
