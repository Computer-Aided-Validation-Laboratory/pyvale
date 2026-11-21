# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""NumPy array checks shared by validation records and calculations."""

import numpy as np


def _validate_real_array(
    values: np.ndarray | float, name: str, *, allow_nan: bool = True
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real integer or floating observations.")

    if np.any(np.isinf(array)):
        raise ValueError(f"{name} cannot contain infinite values (infinity).")

    if not allow_nan and np.any(np.isnan(array)):
        raise ValueError(f"{name} must contain finite values.")

    return array
