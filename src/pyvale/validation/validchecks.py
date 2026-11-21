# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Input checks at public validation boundaries."""

import numpy as np

from pyvale.validation.arraychecks import _validate_real_array
from pyvale.validation.constants import MAVM_DUPLICATE_TOLERANCE
from pyvale.validation.ecdftypes import ValidECDF
from pyvale.validation.validtypes import EMAVMDuplicates


def _validate_finite_sample(values: np.ndarray, name: str) -> np.ndarray:
    array = _validate_real_array(values, name)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; select axes explicitly.")

    finite = np.asarray(array[~np.isnan(array)], dtype=np.float64)

    if not len(finite):
        raise ValueError(f"{name} needs at least one finite observation.")

    return finite


def _validate_unique_sample(values: np.ndarray, data_label: str) -> None:
    sorted_values = np.sort(values)

    if np.any(np.diff(sorted_values) <= MAVM_DUPLICATE_TOLERANCE):
        raise ValueError(
            f"REJECT MAVM duplicates requires unique {data_label} observations within "
            f"MAVM_DUPLICATE_TOLERANCE={MAVM_DUPLICATE_TOLERANCE}. "
            "Use duplicates=EMAVMDuplicates.INCLUDE for repeated observations."
        )


def _validate_mavm_config(alpha: float, duplicates: EMAVMDuplicates, tol: float) -> None:
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


def _validate_ecdf(cdf: ValidECDF) -> None:
    if not isinstance(cdf, ValidECDF):
        raise TypeError("Model CDFs must be ValidECDF records.")

    _validate_real_array(cdf.quantiles, "quantiles", allow_nan=False)
    _validate_real_array(cdf.probabilities, "probabilities", allow_nan=False)

    if (
        cdf.quantiles.ndim != 1
        or not len(cdf.quantiles)
        or cdf.quantiles.shape != cdf.probabilities.shape
    ):
        raise ValueError(
            "ECDF support and probabilities need matching nonempty 1D shapes."
        )

    if np.any(cdf.quantiles[1:] <= cdf.quantiles[:-1]):
        raise ValueError("ECDF support must increase strictly.")

    if (
        np.any(cdf.probabilities <= 0)
        or np.any(cdf.probabilities > 1)
        or np.any(cdf.probabilities[1:] <= cdf.probabilities[:-1])
        or cdf.probabilities[-1] != 1
    ):
        raise ValueError(
            "ECDF probabilities must increase strictly in (0, 1] and end at one."
        )
