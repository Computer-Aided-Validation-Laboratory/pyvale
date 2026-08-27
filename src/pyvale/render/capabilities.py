# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Immutable renderer capability declarations."""

from dataclasses import dataclass

from .mesh import EElementType


@dataclass(frozen=True, slots=True)
class RenderCapabilities:
    """Features advertised by a renderer before a render request.

    Parameters
    ----------
    element_types : frozenset[EElementType]
        Surface element topologies accepted by the renderer.
    supports_lights : bool
        Whether explicit :class:`~pyvale.render.Light` objects are supported.
    supports_camera_distortion : bool
        Whether camera distortion parameters are supported.
    supports_psf : bool
        Whether point-spread-function parameters are supported.
    """

    element_types: frozenset[EElementType]
    supports_lights: bool
    supports_camera_distortion: bool
    supports_psf: bool


__all__ = ["RenderCapabilities"]
