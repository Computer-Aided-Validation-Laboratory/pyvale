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
    """Features that a renderer can advertise before a render request."""

    element_types: frozenset[EElementType]
    supports_lights: bool
    supports_camera_distortion: bool
    supports_psf: bool


__all__ = ["RenderCapabilities"]
