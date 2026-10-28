# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Cython-accelerated validation metric implementations.

The modules in this package are valid pure Python source files.  A wheel build
compiles them into extension modules, while source checkouts retain the same
public API through their Python fallback.
"""

from pyvale.valid.cython.mavm import (
    cyth_calc_mavm_1d,
    cyth_calc_mavm_pbox_1d,
)

__all__ = [
    "cyth_calc_mavm_1d",
    "cyth_calc_mavm_pbox_1d",
]
