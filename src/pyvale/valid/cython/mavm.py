# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Cython-accelerated Modified Area Validation Metric calculations.

This is Cython pure-Python syntax: it is executable without compilation and is
compiled into an extension module in packaged builds. Public metric functions
in :mod:`pyvale.valid.metrics` forward to this implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from pyvale.valid.constants import (
    MAVM_AREA_TOLERANCE,
    MAVM_PROBABILITY_TOLERANCE,
)
from pyvale.valid.inputvalidation import (
    validate_finite_sample,
    validate_mavm_config,
    validate_unique_sample,
)
from pyvale.valid.valtypes import (
    EMAVMDuplicates,
    MAVMResult,
)

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
    model_quantiles: cython.double[::1],
    exp_bound_quantiles: cython.double[::1],
    model_probability_step: cython.double,
    exp_probability_step: cython.double,
    tolerance: cython.double,
) -> tuple[float, float]:
    """Integrate one MAVM bound with the fullfield-compatible algorithm."""
    exp_count: cython.Py_ssize_t = exp_bound_quantiles.shape[0]
    model_count: cython.Py_ssize_t = model_quantiles.shape[0]
    d_plus: cython.double = 0.0
    d_minus: cython.double = 0.0
    d_remainder: cython.double = 0.0
    d_area: cython.double = 0.0
    exp_idx: cython.Py_ssize_t = 0
    model_idx: cython.Py_ssize_t = 0

    if exp_count > model_count:
        for model_idx in range(model_count):
            if abs(d_remainder) > tolerance:
                d_area = (exp_bound_quantiles[exp_idx] - model_quantiles[model_idx]) * (
                    exp_probability_step * (exp_idx + 1)
                    - model_probability_step * model_idx
                )
                if d_area > 0.0:
                    d_plus += d_area
                else:
                    d_minus += d_area
                exp_idx += 1

            while (model_idx + 1) * model_probability_step > (
                exp_idx + 1
            ) * exp_probability_step:
                d_area = (
                    exp_bound_quantiles[exp_idx] - model_quantiles[model_idx]
                ) * exp_probability_step
                if d_area > tolerance:
                    d_plus += d_area
                else:
                    d_minus += d_area
                exp_idx += 1

            if exp_idx < exp_count:
                d_remainder = (
                    exp_bound_quantiles[exp_idx] - model_quantiles[model_idx]
                ) * (
                    model_probability_step * (model_idx + 1)
                    - exp_probability_step * exp_idx
                )
                if d_remainder > 0.0:
                    d_plus += d_remainder
                else:
                    d_minus += d_remainder
    else:
        for exp_idx in range(exp_count):
            if abs(d_remainder) > tolerance:
                d_area = (exp_bound_quantiles[exp_idx] - model_quantiles[model_idx]) * (
                    model_probability_step * (model_idx + 1)
                    - exp_probability_step * exp_idx
                )
                if d_area > tolerance:
                    d_plus += d_area
                else:
                    d_minus += d_area
                model_idx += 1

            while (model_idx + 1) * model_probability_step < (
                exp_idx + 1
            ) * exp_probability_step:
                d_area = (
                    exp_bound_quantiles[exp_idx] - model_quantiles[model_idx]
                ) * model_probability_step
                if d_area > tolerance:
                    d_plus += d_area
                else:
                    d_minus += d_area
                model_idx += 1

            d_remainder = (
                exp_bound_quantiles[exp_idx] - model_quantiles[model_idx]
            ) * (
                exp_probability_step * (exp_idx + 1)
                - model_probability_step * model_idx
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
    model_quantiles: cython.double[::1],
    exp_bound_quantiles: cython.double[::1],
    model_probabilities: cython.double[::1],
    exp_probabilities: cython.double[::1],
    tolerance: cython.double,
) -> tuple[float, float]:
    """Integrate one MAVM bound over a merged empirical-CDF probability grid."""
    model_count: cython.Py_ssize_t = model_quantiles.shape[0]
    exp_count: cython.Py_ssize_t = exp_bound_quantiles.shape[0]
    model_idx: cython.Py_ssize_t = 0
    exp_idx: cython.Py_ssize_t = 0
    probability: cython.double = 0.0
    next_probability: cython.double = 0.0
    d_area: cython.double = 0.0
    d_plus: cython.double = 0.0
    d_minus: cython.double = 0.0
    model_at_next: cython.bint = False
    exp_at_next: cython.bint = False
    probability_tolerance: cython.double = MAVM_PROBABILITY_TOLERANCE

    while model_idx < model_count and exp_idx < exp_count:
        next_probability = min(
            model_probabilities[model_idx],
            exp_probabilities[exp_idx],
        )
        d_area = (exp_bound_quantiles[exp_idx] - model_quantiles[model_idx]) * (
            next_probability - probability
        )
        if d_area > tolerance:
            d_plus += d_area
        elif d_area < -tolerance:
            d_minus -= d_area

        probability = next_probability
        model_at_next = (
            abs(probability - model_probabilities[model_idx]) <= probability_tolerance
        )
        exp_at_next = (
            abs(probability - exp_probabilities[exp_idx]) <= probability_tolerance
        )
        if model_at_next:
            model_idx += 1
        if exp_at_next:
            exp_idx += 1

    return d_plus, d_minus


def convert_contiguous_float64(values: np.ndarray) -> np.ndarray:
    """Return C-contiguous ``float64`` values for typed Cython memoryviews."""
    return np.ascontiguousarray(values, dtype=np.float64)


@cython.ccall
def cyth_calc_mavm_1d(
    model_data: np.ndarray,
    exp_data: np.ndarray,
    alpha: float = 0.05,
    duplicates: EMAVMDuplicates = EMAVMDuplicates.REJECT,
    tol: float = MAVM_AREA_TOLERANCE,
) -> MAVMResult:
    """Calculate MAVM using compiled loops when this module is compiled.

    This has the same input contract and return type as
    :func:`pyvale.valid.metrics.calc_mavm_1d`.
    """
    validate_mavm_config(alpha, duplicates, tol)
    model_clean = validate_finite_sample(model_data, "model_data")
    exp_clean = validate_finite_sample(exp_data, "exp_data")

    if duplicates is EMAVMDuplicates.REJECT:
        validate_unique_sample(model_clean, "model")
        validate_unique_sample(exp_clean, "experimental")

    return _calculate_mavm(model_clean, exp_clean, alpha, duplicates, tol)


def _calculate_mavm(
    model_clean: np.ndarray,
    exp_clean: np.ndarray,
    alpha: float,
    duplicates: EMAVMDuplicates,
    tol: float,
) -> MAVMResult:
    return _calculate_prepared_mavm(
        _prepare_mavm_distribution(model_clean),
        _prepare_mavm_experiment(exp_clean, alpha),
        alpha, duplicates, tol,
    )


@dataclass(slots=True)
class _PreparedECDF:
    quantiles: np.ndarray
    probabilities: np.ndarray


@dataclass(slots=True)
class _PreparedMAVMExperiment:
    distribution: _PreparedECDF
    lower: np.ndarray
    upper: np.ndarray


def _prepare_mavm_distribution(samples: np.ndarray) -> _PreparedECDF:
    distribution = stats.ecdf(samples).cdf
    return _PreparedECDF(
        convert_contiguous_float64(distribution.quantiles),
        convert_contiguous_float64(distribution.probabilities),
    )


def _prepare_mavm_experiment(
    samples: np.ndarray, alpha: float
) -> _PreparedMAVMExperiment:
    distribution = _prepare_mavm_distribution(samples)
    quantiles = distribution.quantiles
    count = len(quantiles)
    degrees_freedom = count - 1
    t_value = stats.t.ppf(1.0 - alpha, degrees_freedom) if degrees_freedom >= 1 else 0.0
    standard_error = np.nanstd(quantiles) / np.sqrt(count)
    return _PreparedMAVMExperiment(
        distribution,
        convert_contiguous_float64(quantiles - t_value * standard_error),
        convert_contiguous_float64(quantiles + t_value * standard_error),
    )


def _calculate_prepared_mavm(
    model: _PreparedECDF,
    experiment: _PreparedMAVMExperiment,
    alpha: float,
    duplicates: EMAVMDuplicates,
    tol: float,
) -> MAVMResult:
    model_quantiles = model.quantiles
    model_probabilities = model.probabilities
    exp_quantiles = experiment.distribution.quantiles
    exp_probabilities = experiment.distribution.probabilities
    exp_conf_lower, exp_conf_upper = experiment.lower, experiment.upper
    exp_count = len(exp_quantiles)

    if duplicates is EMAVMDuplicates.REJECT:
        model_probability_step = 1.0 / len(model_quantiles)
        exp_probability_step = 1.0 / exp_count
        lower_plus, lower_minus = _integrate_mavm_bound_default(
            model_quantiles,
            exp_conf_lower,
            model_probability_step,
            exp_probability_step,
            tol,
        )
        upper_plus, upper_minus = _integrate_mavm_bound_default(
            model_quantiles,
            exp_conf_upper,
            model_probability_step,
            exp_probability_step,
            tol,
        )
    elif duplicates is EMAVMDuplicates.INCLUDE:
        lower_plus, lower_minus = _integrate_mavm_bound_robust(
            model_quantiles,
            exp_conf_lower,
            model_probabilities,
            exp_probabilities,
            tol,
        )
        upper_plus, upper_minus = _integrate_mavm_bound_robust(
            model_quantiles,
            exp_conf_upper,
            model_probabilities,
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
        d_total=d_plus + d_minus,
        model_quantiles=model_quantiles,
        model_probs=model_probabilities,
        exp_quantiles=exp_quantiles,
        exp_probs=exp_probabilities,
        exp_conf_lower=exp_conf_lower,
        exp_conf_upper=exp_conf_upper,
        alpha=alpha,
    )


@cython.ccall
def cyth_calc_mavm_pbox_1d(
    model_pbox_min: np.ndarray,
    model_pbox_max: np.ndarray,
    exp_data: np.ndarray,
    alpha: float = 0.05,
    duplicates: EMAVMDuplicates = EMAVMDuplicates.REJECT,
    tol: float = MAVM_AREA_TOLERANCE,
) -> MAVMResult:
    """Calculate p-box MAVM using :func:`cyth_calc_mavm_1d` for each bound."""
    validate_mavm_config(alpha, duplicates, tol)
    lower_samples = validate_finite_sample(model_pbox_min, "model_pbox_min")
    upper_samples = validate_finite_sample(model_pbox_max, "model_pbox_max")
    experimental_samples = validate_finite_sample(exp_data, "exp_data")
    if duplicates is EMAVMDuplicates.REJECT:
        validate_unique_sample(lower_samples, "lower model")
        validate_unique_sample(upper_samples, "upper model")
        validate_unique_sample(experimental_samples, "experimental")
    result_min = _calculate_mavm(
        lower_samples, experimental_samples, alpha, duplicates, tol
    )
    result_max = _calculate_mavm(
        upper_samples, experimental_samples, alpha, duplicates, tol
    )

    return MAVMResult(
        d_plus=result_min.d_plus,
        d_minus=result_max.d_minus,
        d_total=result_min.d_plus + result_max.d_minus,
        model_quantiles=result_min.model_quantiles,
        model_probs=result_min.model_probs,
        exp_quantiles=result_min.exp_quantiles,
        exp_probs=result_min.exp_probs,
        exp_conf_lower=result_min.exp_conf_lower,
        exp_conf_upper=result_min.exp_conf_upper,
        alpha=alpha,
    )
