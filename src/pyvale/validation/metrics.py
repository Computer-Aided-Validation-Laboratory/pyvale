# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Core 1D validation metrics calculations.

Includes Modified Area Validation Metric (MAVM), classical Area Validation
Metric (AVM / 1-Wasserstein), Kolmogorov-Smirnov distance, Cramér-von Mises,
and deterministic error metrics.
"""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from scipy import stats

from pyvale.validation.arraychecks import _validate_real_array
from pyvale.validation.constants import (
    MAVM_AREA_TOLERANCE,
    METRIC_ZERO_TOLERANCE,
)
from pyvale.validation.cython.mavm import (
    _cyth_calc_mavm_1d,
    _cyth_calc_mavm_pbox_1d,
)
from pyvale.validation.validchecks import _validate_finite_sample
from pyvale.validation.validtypes import (
    EMAVMDuplicates,
    MAVMResult,
)


def calc_mavm_1d(
    model_data: np.ndarray,
    exp_data: np.ndarray,
    *,
    alpha: float = 0.05,
    duplicates: EMAVMDuplicates = EMAVMDuplicates.REJECT,
    tol: float = MAVM_AREA_TOLERANCE,
) -> MAVMResult:
    """Calculates the Modified Area Validation Metric (MAVM) between 1D arrays.

    Implements the corrected fullfieldvalmetrics MAVM algorithm, including
    its fixed unequal-sample probability step. The result values match the
    established fullfield validation workflow.

    Parameters
    ----------
    model_data : np.ndarray
        1D array of simulated / model realization values.
    exp_data : np.ndarray
        1D array of experimental observation values.
    alpha : float, optional
        Legacy Student's t quantile parameter: t.ppf(1-alpha, n_unique-1).
        Confidence shifts use the population standard deviation of unique
        experimental support divided by sqrt(n_unique), including with INCLUDE.
    duplicates : EMAVMDuplicates, optional
        REJECT preserves fullfieldvalmetrics behaviour and requires unique
        observations. INCLUDE supports repeated observations using weighted ECDF
        probability steps.
    tol : float, optional
        Floating-point comparison tolerance for interval boundaries.

    Returns
    -------
    MAVMResult
        Complete result dataclass with d+, d-, and CDF data.
    """
    return _cyth_calc_mavm_1d(model_data, exp_data, alpha, duplicates, tol)


def calc_mavm_pbox_1d(
    model_pbox_min: np.ndarray,
    model_pbox_max: np.ndarray,
    exp_data: np.ndarray,
    *,
    alpha: float = 0.05,
    duplicates: EMAVMDuplicates = EMAVMDuplicates.REJECT,
    tol: float = MAVM_AREA_TOLERANCE,
) -> MAVMResult:
    """Calculates MAVM between an epistemic simulation p-box and experimental
    data.

    Parameters
    ----------
    model_pbox_min : np.ndarray
        Lower envelope model samples (min CDF across epistemic parameter space).
    model_pbox_max : np.ndarray
        Upper envelope model samples (max CDF across epistemic parameter space).
    exp_data : np.ndarray
        1D experimental observation samples.
    alpha : float, optional
        Legacy Student's t quantile parameter; see calc_mavm_1d.
    duplicates : EMAVMDuplicates, optional
        MAVM empirical-CDF integration algorithm.
    tol : float, optional
        Tolerance.

    Returns
    -------
    MAVMResult
        MAVM result comparing the p-box against experimental confidence bounds.
    """
    return _cyth_calc_mavm_pbox_1d(
        model_pbox_min, model_pbox_max, exp_data, alpha, duplicates, tol
    )


def calc_avm_1d(
    model_data: np.ndarray,
    exp_data: np.ndarray,
) -> float:
    """Calculates classical Area Validation Metric (1-Wasserstein distance).

    Parameters
    ----------
    model_data : np.ndarray
        1D array of model realizations.
    exp_data : np.ndarray
        1D array of experimental observations.

    Returns
    -------
    float
        1-Wasserstein / AVM distance between model and experiment.
    """
    model_samples = _validate_finite_sample(model_data, "model_data")
    exp_samples = _validate_finite_sample(exp_data, "exp_data")
    return float(stats.wasserstein_distance(model_samples, exp_samples))


def calc_ks_1d(
    model_data: np.ndarray,
    exp_data: np.ndarray,
) -> float:
    """Calculates Kolmogorov-Smirnov validation distance (L-infinity on CDFs).

    Parameters
    ----------
    model_data : np.ndarray
        1D array of model realizations.
    exp_data : np.ndarray
        1D array of experimental observations.

    Returns
    -------
    float
        Two-sample Kolmogorov-Smirnov test statistic.
    """
    model_samples = _validate_finite_sample(model_data, "model_data")
    exp_samples = _validate_finite_sample(exp_data, "exp_data")
    return _calculate_ks_statistic(model_samples, exp_samples)


def calc_cvm_1d(
    model_data: np.ndarray,
    exp_data: np.ndarray,
) -> float:
    """Calculates Cramér-von Mises validation distance (L2 norm on CDFs).

    Parameters
    ----------
    model_data : np.ndarray
        1D array of model realizations.
    exp_data : np.ndarray
        1D array of experimental observations.

    Returns
    -------
    float
        Two-sample Cramér-von Mises test statistic.
    """
    model_samples = _validate_finite_sample(model_data, "model_data")
    exp_samples = _validate_finite_sample(exp_data, "exp_data")
    return _calculate_cvm_statistic(model_samples, exp_samples)


def _calculate_ks_statistic(sim: np.ndarray, exp: np.ndarray) -> float:
    sim_sorted = np.sort(sim)
    exp_sorted = np.sort(exp)
    support = np.concatenate((sim_sorted, exp_sorted))
    sim_cdf = np.searchsorted(sim_sorted, support, side="right") / len(sim)
    exp_cdf = np.searchsorted(exp_sorted, support, side="right") / len(exp)
    return float(np.max(np.abs(sim_cdf - exp_cdf)))


def _calculate_cvm_statistic(sim: np.ndarray, exp: np.ndarray) -> float:
    sim_count, exp_count = len(sim), len(exp)
    if min(sim_count, exp_count) < 2:
        return float(stats.cramervonmises_2samp(sim, exp).statistic)
    # Anderson's two-sample statistic with pooled average ranks for ties.
    pooled = np.concatenate((np.sort(sim), np.sort(exp)))
    ranks = stats.rankdata(pooled, method="average")
    sim_rank_error = ranks[:sim_count] - np.arange(1, sim_count + 1)
    exp_rank_error = ranks[sim_count:] - np.arange(1, exp_count + 1)
    rank_sum = sim_count * np.sum(sim_rank_error * sim_rank_error) + exp_count * np.sum(
        exp_rank_error * exp_rank_error
    )
    pair_count = sim_count * exp_count
    total_count = sim_count + exp_count
    return float(
        rank_sum / (pair_count * total_count) - (4 * pair_count - 1) / (6 * total_count)
    )


class ERelErrNorm(Enum):
    EXP_MAX = auto()
    SIM_MAX = auto()
    CONSTANT = auto()


@dataclass(slots=True)
class DeterministicMetricResult:
    absolute_error: float
    relative_error: float
    rmse: float
    nmse: float


def _validate_relative_error_config(
    normalisation: ERelErrNorm,
    constant: float | None,
) -> None:
    if not isinstance(normalisation, ERelErrNorm):
        raise TypeError("Normalisation must be ERelErrNorm.")
    if normalisation is ERelErrNorm.CONSTANT:
        if (
            constant is None
            or not isinstance(constant, (int, float, np.integer, np.floating))
            or isinstance(constant, (bool, np.bool_))
            or not np.isfinite(constant)
            or constant <= 0
        ):
            raise ValueError(
                "Constant normalisation requires a positive finite constant."
            )
    elif constant is not None:
        raise ValueError("A normalisation constant requires CONSTANT normalisation.")


def _calculate_relative_error(
    sim: np.ndarray,
    exp: np.ndarray,
    normalisation: ERelErrNorm,
    constant: float | None,
) -> float:
    if normalisation is ERelErrNorm.EXP_MAX:
        denominator = float(np.max(np.abs(exp)))
    elif normalisation is ERelErrNorm.SIM_MAX:
        denominator = float(np.max(np.abs(sim)))
    else:
        assert constant is not None
        denominator = constant
    if denominator == 0:
        return float("nan")
    return float(np.mean(np.abs(sim - exp)) / denominator)


def _validate_paired_samples(
    sim: float | np.ndarray,
    exp: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sim_array = np.atleast_1d(_validate_real_array(sim, "simulation"))
    exp_array = np.atleast_1d(_validate_real_array(exp, "experiment"))
    if sim_array.ndim != 1 or sim_array.shape != exp_array.shape:
        raise ValueError("Deterministic metrics require paired 1D observations.")
    paired = ~np.isnan(sim_array) & ~np.isnan(exp_array)
    if not np.any(paired):
        raise ValueError("Deterministic metrics need a finite observation pair.")
    return sim_array[paired].astype(np.float64), exp_array[paired].astype(np.float64)


def calc_deterministic_metrics_1d(
    sim_valid: float | np.ndarray,
    exp_valid: float | np.ndarray,
    *,
    normalisation: ERelErrNorm = ERelErrNorm.EXP_MAX,
    constant: float | None = None,
) -> DeterministicMetricResult:
    """Return paired errors; NMSE is NaN for variance at or below the zero tolerance."""
    _validate_relative_error_config(normalisation, constant)
    sim, exp = _validate_paired_samples(sim_valid, exp_valid)
    residual = sim - exp
    mean_square_error = float(np.mean(residual * residual))
    exp_variance = float(np.var(exp))
    return DeterministicMetricResult(
        absolute_error=float(np.mean(np.abs(residual))),
        relative_error=_calculate_relative_error(sim, exp, normalisation, constant),
        rmse=float(np.sqrt(mean_square_error)),
        nmse=mean_square_error / exp_variance
        if exp_variance > METRIC_ZERO_TOLERANCE
        else float("nan"),
    )
