# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Cython-accelerated Modified Area Validation Metric calculations.

This is Cython pure-Python syntax: it is executable without compilation and is
compiled into an extension module in packaged builds.  The reference Python
implementation remains in :mod:`pyvale.valid.metrics`.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

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

from pyvale.valid.constants import (
    MAVM_AREA_TOLERANCE,
    MAVM_PROBABILITY_TOLERANCE,
)
from pyvale.valid.metrics import (
    EMAVMMode,
    MAVMResult,
    _raise_for_near_duplicates,
)


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
    exp_index: cython.Py_ssize_t = 0
    model_index: cython.Py_ssize_t = 0

    if exp_count > model_count:
        for model_index in range(model_count):
            if abs(d_remainder) > tolerance:
                d_area = (
                    exp_bound_quantiles[exp_index] - model_quantiles[model_index]
                ) * (
                    exp_probability_step * (exp_index + 1)
                    - model_probability_step * model_index
                )
                if d_area > 0.0:
                    d_plus += d_area
                else:
                    d_minus += d_area
                exp_index += 1

            while (model_index + 1) * model_probability_step > (
                exp_index + 1
            ) * exp_probability_step:
                d_area = (
                    exp_bound_quantiles[exp_index] - model_quantiles[model_index]
                ) * exp_probability_step
                if d_area > tolerance:
                    d_plus += d_area
                else:
                    d_minus += d_area
                exp_index += 1

            if exp_index < exp_count:
                d_remainder = (
                    exp_bound_quantiles[exp_index] - model_quantiles[model_index]
                ) * (
                    model_probability_step * (model_index + 1)
                    - exp_probability_step * exp_index
                )
                if d_remainder > 0.0:
                    d_plus += d_remainder
                else:
                    d_minus += d_remainder
    else:
        for exp_index in range(exp_count):
            if abs(d_remainder) > tolerance:
                d_area = (
                    exp_bound_quantiles[exp_index] - model_quantiles[model_index]
                ) * (
                    model_probability_step * (model_index + 1)
                    - exp_probability_step * exp_index
                )
                if d_area > tolerance:
                    d_plus += d_area
                else:
                    d_minus += d_area
                model_index += 1

            while (model_index + 1) * model_probability_step < (
                exp_index + 1
            ) * exp_probability_step:
                d_area = (
                    exp_bound_quantiles[exp_index] - model_quantiles[model_index]
                ) * model_probability_step
                if d_area > tolerance:
                    d_plus += d_area
                else:
                    d_minus += d_area
                model_index += 1

            d_remainder = (
                exp_bound_quantiles[exp_index] - model_quantiles[model_index]
            ) * (
                exp_probability_step * (exp_index + 1)
                - model_probability_step * model_index
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
    model_index: cython.Py_ssize_t = 0
    exp_index: cython.Py_ssize_t = 0
    probability: cython.double = 0.0
    next_probability: cython.double = 0.0
    d_area: cython.double = 0.0
    d_plus: cython.double = 0.0
    d_minus: cython.double = 0.0
    model_at_next: cython.bint = False
    exp_at_next: cython.bint = False
    probability_tolerance: cython.double = MAVM_PROBABILITY_TOLERANCE

    while model_index < model_count and exp_index < exp_count:
        next_probability = min(
            model_probabilities[model_index],
            exp_probabilities[exp_index],
        )
        d_area = (exp_bound_quantiles[exp_index] - model_quantiles[model_index]) * (
            next_probability - probability
        )
        if d_area > tolerance:
            d_plus += d_area
        elif d_area < -tolerance:
            d_minus -= d_area

        probability = next_probability
        model_at_next = (
            abs(probability - model_probabilities[model_index]) <= probability_tolerance
        )
        exp_at_next = (
            abs(probability - exp_probabilities[exp_index]) <= probability_tolerance
        )
        if model_at_next:
            model_index += 1
        if exp_at_next:
            exp_index += 1

    return d_plus, d_minus


def _as_contiguous_float64(values: np.ndarray) -> np.ndarray:
    """Return C-contiguous ``float64`` values for typed Cython memoryviews."""
    return np.ascontiguousarray(values, dtype=np.float64)


@cython.ccall
def cyth_calc_mavm_1d(
    model_data: np.ndarray,
    exp_data: np.ndarray,
    alpha: float = 0.05,
    mode: EMAVMMode = EMAVMMode.DEFAULT,
    tol: float = MAVM_AREA_TOLERANCE,
) -> MAVMResult:
    """Calculate MAVM using compiled loops when this module is compiled.

    This has the same input contract and return type as
    :func:`pyvale.valid.metrics.calc_mavm_1d`.
    """
    model_clean = np.asarray(model_data, dtype=np.float64).ravel()
    model_clean = model_clean[~np.isnan(model_clean)]
    exp_clean = np.asarray(exp_data, dtype=np.float64).ravel()
    exp_clean = exp_clean[~np.isnan(exp_clean)]

    if len(model_clean) == 0 or len(exp_clean) == 0:
        raise ValueError("Cannot calculate MAVM on empty or all-NaN data.")

    if mode is EMAVMMode.DEFAULT:
        _raise_for_near_duplicates(model_clean, "model")
        _raise_for_near_duplicates(exp_clean, "experimental")

    model_cdf = stats.ecdf(model_clean).cdf
    exp_cdf = stats.ecdf(exp_clean).cdf
    model_quantiles = _as_contiguous_float64(model_cdf.quantiles)
    exp_quantiles = _as_contiguous_float64(exp_cdf.quantiles)
    model_probabilities = _as_contiguous_float64(model_cdf.probabilities)
    exp_probabilities = _as_contiguous_float64(exp_cdf.probabilities)

    exp_count = len(exp_quantiles)
    degrees_freedom = exp_count - 1
    t_value = stats.t.ppf(1.0 - alpha, degrees_freedom) if degrees_freedom >= 1 else 0.0
    standard_error = np.nanstd(exp_quantiles) / np.sqrt(exp_count)
    exp_conf_lower = _as_contiguous_float64(exp_quantiles - t_value * standard_error)
    exp_conf_upper = _as_contiguous_float64(exp_quantiles + t_value * standard_error)

    if mode is EMAVMMode.DEFAULT:
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
    elif mode is EMAVMMode.ROBUST:
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
        raise ValueError(f"Unsupported MAVM mode: {mode!r}.")

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
    mode: EMAVMMode = EMAVMMode.DEFAULT,
    tol: float = MAVM_AREA_TOLERANCE,
) -> MAVMResult:
    """Calculate p-box MAVM using :func:`cyth_calc_mavm_1d` for each bound."""
    result_min = cyth_calc_mavm_1d(
        model_pbox_min,
        exp_data,
        alpha=alpha,
        mode=mode,
        tol=tol,
    )
    result_max = cyth_calc_mavm_1d(
        model_pbox_max,
        exp_data,
        alpha=alpha,
        mode=mode,
        tol=tol,
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
