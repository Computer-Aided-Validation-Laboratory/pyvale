"""Input checks at public validation boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyvale.valid.constants import MAVM_DUPLICATE_TOLERANCE

if TYPE_CHECKING:
    from pyvale.valid.valtypes import EMAVMDuplicates


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


def validate_mavm_config(alpha: float, duplicates: EMAVMDuplicates, tol: float) -> None:
    from pyvale.valid.valtypes import EMAVMDuplicates

    if not isinstance(duplicates, EMAVMDuplicates):
        raise TypeError("MAVM duplicates must be EMAVMDuplicates.")
    if (
        not isinstance(alpha, (int, float, np.integer, np.floating))
        or isinstance(alpha, (bool, np.bool_))
        or not np.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError("MAVM alpha must be between zero and one.")
    if (
        not isinstance(tol, (int, float, np.integer, np.floating))
        or isinstance(tol, (bool, np.bool_))
        or not np.isfinite(tol)
        or tol < 0
    ):
        raise ValueError("MAVM tolerance must be finite and nonnegative.")


def validate_unique_sample(values: np.ndarray, data_label: str) -> None:
    sorted_values = np.sort(values)
    if np.any(np.diff(sorted_values) <= MAVM_DUPLICATE_TOLERANCE):
        raise ValueError(
            f"REJECT MAVM duplicates requires unique {data_label} observations within "
            f"MAVM_DUPLICATE_TOLERANCE={MAVM_DUPLICATE_TOLERANCE}. "
            "Use duplicates=EMAVMDuplicates.INCLUDE for repeated observations."
        )
