# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Reserved public entry point for future planar pixel integration."""

from .imagewarp2d import IImageWarp2D
from .result import ImageWarpResult


class PxInt2D(IImageWarp2D):
    """Placeholder for the future Grid2D and Speck2D image-warp workflow.

    Notes
    -----
    The name reserves the public API while its constituent renderers are being
    designed. It cannot yet validate or render a request.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reserve construction arguments for the future PxInt2D backend.

        Parameters
        ----------
        *args : object
            Future positional configuration arguments.
        **kwargs : object
            Future keyword configuration arguments.
        """
        pass

    def verify_input(self, *args: object, **kwargs: object) -> object:
        """Raise until the PxInt2D input contract is implemented.

        Raises
        ------
        NotImplementedError
            Always, because PxInt2D is a public stub.
        """
        raise NotImplementedError("PxInt2D integration is not available yet.")

    def _render(self, render_plan: object) -> ImageWarpResult:
        """Raise until the PxInt2D renderer is implemented.

        Raises
        ------
        NotImplementedError
            Always, because PxInt2D is a public stub.
        """
        raise NotImplementedError("PxInt2D integration is not available yet.")


__all__ = ["PxInt2D"]
