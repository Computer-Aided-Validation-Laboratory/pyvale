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
    """Check that a NumPy array contains real integer or floating-point values.

    Parameters
    ----------
    values : ndarray or float
        Scalar or array of any rank. Infinities are always rejected.
    name : str
        Input label used in exception messages.
    allow_nan : bool, optional
        Allow NaN entries to represent missing values.

    Returns
    -------
    ndarray
        Array with the same shape and dtype inferred from the input.
    """
    array = np.asarray(values)

    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real integer or floating observations.")

    if np.any(np.isinf(array)):
        raise ValueError(f"{name} cannot contain infinite values (infinity).")

    if not allow_nan and np.any(np.isnan(array)):
        raise ValueError(f"{name} must contain finite values.")

    return array
