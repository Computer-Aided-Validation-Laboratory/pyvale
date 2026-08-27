# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Analytic two-dimensional finite-element renderers."""

from .grid import PixIntGrid2D
from .model import (
    AnalyticRule,
    Eggbox,
    EPxIntMapping,
    GaussianPSF,
    GaussRule,
    PxInt2DOpts,
    RectRule,
    quantise_image,
)
from .speck import AdditiveSpeckles, PixIntSpeck2D

__all__ = [
    "AdditiveSpeckles",
    "AnalyticRule",
    "EPxIntMapping",
    "Eggbox",
    "GaussRule",
    "GaussianPSF",
    "PixIntGrid2D",
    "PixIntSpeck2D",
    "PxInt2DOpts",
    "RectRule",
    "quantise_image",
]
