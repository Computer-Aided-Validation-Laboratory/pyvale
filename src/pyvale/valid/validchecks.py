# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Input checks at public validation boundaries."""

from __future__ import annotations

import numpy as np

from pyvale.valid.constants import MAVM_DUPLICATE_TOLERANCE


def validate_real_array(
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


def validate_finite_sample(values: np.ndarray, name: str) -> np.ndarray:
    array = validate_real_array(values, name)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; select axes explicitly.")

    finite = np.asarray(array[~np.isnan(array)], dtype=np.float64)

    if not len(finite):
        raise ValueError(f"{name} needs at least one finite observation.")

    return finite


def validate_unique_sample(values: np.ndarray, data_label: str) -> None:
    sorted_values = np.sort(values)

    if np.any(np.diff(sorted_values) <= MAVM_DUPLICATE_TOLERANCE):
        raise ValueError(
            f"REJECT MAVM duplicates requires unique {data_label} observations within "
            f"MAVM_DUPLICATE_TOLERANCE={MAVM_DUPLICATE_TOLERANCE}. "
            "Use duplicates=EMAVMDuplicates.INCLUDE for repeated observations."
        )
