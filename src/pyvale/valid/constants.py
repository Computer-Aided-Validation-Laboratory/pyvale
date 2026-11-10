# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
"""Numerical constants used by validation metrics."""

from typing import Final

MAVM_PROBABILITY_TOLERANCE: Final[float] = 1.0e-12
"""Tolerance for floating-point probability comparisons."""

MAVM_DUPLICATE_TOLERANCE: Final[float] = 1e-12
"""Default absolute tolerance for merging duplicate observations."""

MAVM_AREA_TOLERANCE: Final[float] = 1e-12
"""Default minimum retained rectangle area."""

METRIC_ZERO_TOLERANCE: Final[float] = 1e-12
"""Default threshold for exact-zero denominator detection."""
