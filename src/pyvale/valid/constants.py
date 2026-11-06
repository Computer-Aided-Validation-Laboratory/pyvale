# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Numerical constants used by validation metrics."""

from typing import Final

MAVM_AREA_TOLERANCE: Final[float] = 1.0e-12
"""Smallest signed MAVM rectangle area treated as nonzero."""

MAVM_DUPLICATE_TOLERANCE: Final[float] = 1.0e-12
"""Maximum separation treated as a duplicate observation in default MAVM."""

MAVM_PROBABILITY_TOLERANCE: Final[float] = 1.0e-12
"""Comparison tolerance for merged empirical-CDF probability steps."""

METRIC_ZERO_TOLERANCE: Final[float] = 1.0e-12
"""Threshold below which a metric denominator is treated as zero."""
