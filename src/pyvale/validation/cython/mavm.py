# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Cython-accelerated Modified Area Validation Metric calculations.

This is Cython pure-Python syntax: it is executable without compilation and is
compiled into an extension module in packaged builds. Public metric functions
in :mod:`pyvale.validation.metrics` forward to this implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from pyvale.validation.constants import (
    MAVM_AREA_TOLERANCE,
    MAVM_DUPLICATE_TOLERANCE,
    MAVM_PROBABILITY_TOLERANCE,
)
from pyvale.validation.imetric import MAVMResult
from pyvale.validation.validchecks import (
    _validate_finite_sample,
    _validate_mavm_config,
)
from pyvale.validation.validtypes import EMAVMDuplicates

try:
    import cython
except ModuleNotFoundError:

    class _CythonFallback:
        """Make Cython annotations no-ops when running uncompiled source."""

        compiled: bool = False
        double = float
        Py_ssize_t = int

        @staticmethod
        def cfunc(function: object) -> object:
            """Return a pure-Python function unchanged."""
            return function

        @staticmethod
        def ccall(function: object) -> object:
            """Return a pure-Python function unchanged."""
            return function

        @staticmethod
        def boundscheck(enabled: bool) -> object:
            """Return a decorator that preserves the supplied function."""
            return lambda function: function

        @staticmethod
        def wraparound(enabled: bool) -> object:
            """Return a decorator that preserves the supplied function."""
            return lambda function: function

    cython = _CythonFallback()


@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
def _integrate_mavm_bound_default(
    sim_quantiles: cython.double[::1],
    exp_bound_quantiles: cython.double[::1],
    sim_probability_step: cython.double,
    exp_probability_step: cython.double,
    tolerance: cython.double,
) -> tuple[float, float]:
    """Integrate one MAVM bound with the fullfield-compatible algorithm."""
    exp_count: cython.Py_ssize_t = exp_bound_quantiles.shape[0]
    sim_count: cython.Py_ssize_t = sim_quantiles.shape[0]
    d_plus: cython.double = 0.0
    d_minus: cython.double = 0.0
    d_remainder: cython.double = 0.0
    d_area: cython.double = 0.0
    exp_idx: cython.Py_ssize_t = 0
    sim_idx: cython.Py_ssize_t = 0

    if exp_count > sim_count:
        for sim_idx in range(sim_count):
            if abs(d_remainder) > tolerance:
                d_area = (exp_bound_quantiles[exp_idx] - sim_quantiles[sim_idx]) * (
                    exp_probability_step * (exp_idx + 1)
                    - sim_probability_step * sim_idx
                )
                if d_area > 0.0:
                    d_plus += d_area
                else:
                    d_minus += d_area
                exp_idx += 1

            while (sim_idx + 1) * sim_probability_step > (
                exp_idx + 1
            ) * exp_probability_step:
                d_area = (
                    exp_bound_quantiles[exp_idx] - sim_quantiles[sim_idx]
                ) * exp_probability_step
                if d_area > tolerance:
                    d_plus += d_area
                else:
                    d_minus += d_area
                exp_idx += 1

            if exp_idx < exp_count:
                d_remainder = (
                    exp_bound_quantiles[exp_idx] - sim_quantiles[sim_idx]
                ) * (
                    sim_probability_step * (sim_idx + 1)
                    - exp_probability_step * exp_idx
                )
                if d_remainder > 0.0:
                    d_plus += d_remainder
                else:
                    d_minus += d_remainder
    else:
        for exp_idx in range(exp_count):
            if abs(d_remainder) > tolerance:
                d_area = (exp_bound_quantiles[exp_idx] - sim_quantiles[sim_idx]) * (
                    sim_probability_step * (sim_idx + 1)
                    - exp_probability_step * exp_idx
                )
                if d_area > tolerance:
                    d_plus += d_area
                else:
                    d_minus += d_area
                sim_idx += 1

            while (sim_idx + 1) * sim_probability_step < (
                exp_idx + 1
            ) * exp_probability_step:
                d_area = (
                    exp_bound_quantiles[exp_idx] - sim_quantiles[sim_idx]
                ) * sim_probability_step
                if d_area > tolerance:
                    d_plus += d_area
                else:
                    d_minus += d_area
                sim_idx += 1

            d_remainder = (
                exp_bound_quantiles[exp_idx] - sim_quantiles[sim_idx]
            ) * (
                exp_probability_step * (exp_idx + 1)
                - sim_probability_step * sim_idx
            )
            if d_remainder > tolerance:
                d_plus += d_remainder
            else:
                d_minus += d_remainder

    return d_plus, d_minus


@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
def _integrate_mavm_bound_robust(
    sim_quantiles: cython.double[::1],
    exp_bound_quantiles: cython.double[::1],
    sim_probabilities: cython.double[::1],
    exp_probabilities: cython.double[::1],
    tolerance: cython.double,
) -> tuple[float, float]:
    """Integrate one MAVM bound over a merged empirical-CDF probability grid."""
    sim_count: cython.Py_ssize_t = sim_quantiles.shape[0]
    exp_count: cython.Py_ssize_t = exp_bound_quantiles.shape[0]
    sim_idx: cython.Py_ssize_t = 0
    exp_idx: cython.Py_ssize_t = 0
    probability: cython.double = 0.0
    next_probability: cython.double = 0.0
    d_area: cython.double = 0.0
    d_plus: cython.double = 0.0
    d_minus: cython.double = 0.0
    sim_at_next: cython.bint = False
    exp_at_next: cython.bint = False
    probability_tolerance: cython.double = MAVM_PROBABILITY_TOLERANCE

    while sim_idx < sim_count and exp_idx < exp_count:
        next_probability = min(
            sim_probabilities[sim_idx],
            exp_probabilities[exp_idx],
        )
        d_area = (exp_bound_quantiles[exp_idx] - sim_quantiles[sim_idx]) * (
            next_probability - probability
        )
        if d_area > tolerance:
            d_plus += d_area
        elif d_area < -tolerance:
            d_minus -= d_area

        probability = next_probability
        sim_at_next = (
            abs(probability - sim_probabilities[sim_idx]) <= probability_tolerance
        )
        exp_at_next = (
            abs(probability - exp_probabilities[exp_idx]) <= probability_tolerance
        )
        if sim_at_next:
            sim_idx += 1
        if exp_at_next:
            exp_idx += 1

    return d_plus, d_minus


def _convert_contiguous_float64(values: np.ndarray) -> np.ndarray:
    """Return C-contiguous ``float64`` values for typed Cython memoryviews."""
    return np.ascontiguousarray(values, dtype=np.float64)


@cython.ccall
def _cyth_calc_mavm_1d(
    sim_data: np.ndarray,
    exp_data: np.ndarray,
    alpha: float = 0.05,
    duplicates: EMAVMDuplicates = EMAVMDuplicates.REJECT,
    tol: float = MAVM_AREA_TOLERANCE,
) -> MAVMResult:
    """Calculate MAVM using compiled loops when this module is compiled.

    This has the same input contract and return type as
    :func:`pyvale.validation.metrics.calc_mavm_1d`.
    """
    _validate_mavm_config(alpha, duplicates, tol)
    sim_clean = _validate_finite_sample(sim_data, "sim_data")
    exp_clean = _validate_finite_sample(exp_data, "exp_data")

    sim = _prepare_mavm_distribution(
        sim_clean,
        unique_label="sim" if duplicates is EMAVMDuplicates.REJECT else None,
    )
    exp = _prepare_mavm_exp(
        exp_clean,
        alpha,
        unique_label="experimental" if duplicates is EMAVMDuplicates.REJECT else None,
    )
    return _calculate_prepared_mavm(sim, exp, alpha, duplicates, tol)


def _calculate_mavm(
    sim_clean: np.ndarray,
    exp_clean: np.ndarray,
    alpha: float,
    duplicates: EMAVMDuplicates,
    tol: float,
) -> MAVMResult:
    return _calculate_prepared_mavm(
        _prepare_mavm_distribution(sim_clean),
        _prepare_mavm_exp(exp_clean, alpha),
        alpha,
        duplicates,
        tol,
    )


@dataclass(slots=True)
class _PreparedECDF:
    quantiles: np.ndarray
    probabilities: np.ndarray


@dataclass(slots=True)
class _PreparedMAVMExp:
    distribution: _PreparedECDF
    lower: np.ndarray
    upper: np.ndarray


def _prepare_mavm_distribution(
    samples: np.ndarray, *, unique_label: str | None = None
) -> _PreparedECDF:
    distribution = stats.ecdf(samples).cdf
    if unique_label is not None:
        has_duplicates = len(distribution.quantiles) != len(samples)
        has_close_support = np.any(
            np.diff(distribution.quantiles) <= MAVM_DUPLICATE_TOLERANCE
        )
        if has_duplicates or has_close_support:
            raise ValueError(
                f"REJECT MAVM duplicates requires unique {unique_label} observations "
                f"within MAVM_DUPLICATE_TOLERANCE={MAVM_DUPLICATE_TOLERANCE}. "
                "Use duplicates=EMAVMDuplicates.INCLUDE for repeated observations."
            )
    return _PreparedECDF(
        _convert_contiguous_float64(distribution.quantiles),
        _convert_contiguous_float64(distribution.probabilities),
    )


def _prepare_mavm_exp(
    samples: np.ndarray, alpha: float, *, unique_label: str | None = None
) -> _PreparedMAVMExp:
    distribution = _prepare_mavm_distribution(samples, unique_label=unique_label)
    quantiles = distribution.quantiles
    count = len(quantiles)
    degrees_freedom = count - 1
    t_value = stats.t.ppf(1.0 - alpha, degrees_freedom) if degrees_freedom >= 1 else 0.0
    standard_error = np.nanstd(quantiles) / np.sqrt(count)
    return _PreparedMAVMExp(
        distribution,
        _convert_contiguous_float64(quantiles - t_value * standard_error),
        _convert_contiguous_float64(quantiles + t_value * standard_error),
    )


def _calculate_prepared_mavm(
    sim: _PreparedECDF,
    exp: _PreparedMAVMExp,
    alpha: float,
    duplicates: EMAVMDuplicates,
    tol: float,
) -> MAVMResult:
    sim_quantiles = sim.quantiles
    sim_probabilities = sim.probabilities
    exp_quantiles = exp.distribution.quantiles
    exp_probabilities = exp.distribution.probabilities
    exp_conf_lower, exp_conf_upper = exp.lower, exp.upper
    exp_count = len(exp_quantiles)

    if duplicates is EMAVMDuplicates.REJECT:
        sim_probability_step = 1.0 / len(sim_quantiles)
        exp_probability_step = 1.0 / exp_count
        lower_plus, lower_minus = _integrate_mavm_bound_default(
            sim_quantiles,
            exp_conf_lower,
            sim_probability_step,
            exp_probability_step,
            tol,
        )
        upper_plus, upper_minus = _integrate_mavm_bound_default(
            sim_quantiles,
            exp_conf_upper,
            sim_probability_step,
            exp_probability_step,
            tol,
        )
    elif duplicates is EMAVMDuplicates.INCLUDE:
        lower_plus, lower_minus = _integrate_mavm_bound_robust(
            sim_quantiles,
            exp_conf_lower,
            sim_probabilities,
            exp_probabilities,
            tol,
        )
        upper_plus, upper_minus = _integrate_mavm_bound_robust(
            sim_quantiles,
            exp_conf_upper,
            sim_probabilities,
            exp_probabilities,
            tol,
        )
    else:
        raise ValueError(f"Unsupported MAVM duplicates: {duplicates!r}.")

    d_plus = float(max(abs(lower_plus), abs(upper_plus)))
    d_minus = float(max(abs(lower_minus), abs(upper_minus)))

    return MAVMResult(
        d_plus=d_plus,
        d_minus=d_minus,
        sim_quantiles=sim_quantiles,
        sim_probs=sim_probabilities,
        exp_quantiles=exp_quantiles,
        exp_probs=exp_probabilities,
        exp_conf_lower=exp_conf_lower,
        exp_conf_upper=exp_conf_upper,
        alpha=alpha,
    )


@cython.ccall
def _cyth_calc_mavm_pbox_1d(
    sim_pbox_min: np.ndarray,
    sim_pbox_max: np.ndarray,
    exp_data: np.ndarray,
    alpha: float = 0.05,
    duplicates: EMAVMDuplicates = EMAVMDuplicates.REJECT,
    tol: float = MAVM_AREA_TOLERANCE,
) -> MAVMResult:
    """Compare both simulation endpoints against one experimental sample."""
    _validate_mavm_config(alpha, duplicates, tol)
    lower_samples = _validate_finite_sample(sim_pbox_min, "sim_pbox_min")
    upper_samples = _validate_finite_sample(sim_pbox_max, "sim_pbox_max")
    exp_samples = _validate_finite_sample(exp_data, "exp_data")
    lower_sim = _prepare_mavm_distribution(
        lower_samples,
        unique_label="lower sim" if duplicates is EMAVMDuplicates.REJECT else None,
    )
    upper_sim = _prepare_mavm_distribution(
        upper_samples,
        unique_label="upper sim" if duplicates is EMAVMDuplicates.REJECT else None,
    )
    exp = _prepare_mavm_exp(
        exp_samples,
        alpha,
        unique_label="experimental" if duplicates is EMAVMDuplicates.REJECT else None,
    )
    result_min = _calculate_prepared_mavm(lower_sim, exp, alpha, duplicates, tol)
    result_max = _calculate_prepared_mavm(upper_sim, exp, alpha, duplicates, tol)

    return MAVMResult(
        d_plus=result_min.d_plus,
        d_minus=result_max.d_minus,
        sim_quantiles=result_min.sim_quantiles,
        sim_probs=result_min.sim_probs,
        exp_quantiles=result_min.exp_quantiles,
        exp_probs=result_min.exp_probs,
        exp_conf_lower=result_min.exp_conf_lower,
        exp_conf_upper=result_min.exp_conf_upper,
        alpha=alpha,
    )
