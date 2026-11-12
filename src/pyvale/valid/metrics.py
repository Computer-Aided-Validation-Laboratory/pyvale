# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Core 1D validation metrics calculations.

Includes Modified Area Validation Metric (MAVM), classical Area Validation
Metric (AVM / 1-Wasserstein), Kolmogorov-Smirnov distance, Cramér-von Mises,
U-pooling, and deterministic error metrics.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import cast

import numpy as np
from scipy import stats

from pyvale.valid.constants import (
    MAVM_AREA_TOLERANCE,
    METRIC_ZERO_TOLERANCE,
)
from pyvale.valid.cython.mavm import (
    cyth_calc_mavm_1d,
    cyth_calc_mavm_pbox_1d,
)
from pyvale.valid.inputvalidation import validate_finite_sample, validate_real_array
from pyvale.valid.valtypes import EMAVMDuplicates, MAVMResult, ValECDF


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
        Complete result dataclass with d+, d-, d_total, and CDF data.
    """
    return cyth_calc_mavm_1d(model_data, exp_data, alpha, duplicates, tol)


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
    return cyth_calc_mavm_pbox_1d(
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
    model_samples = validate_finite_sample(model_data, "model_data")
    experimental_samples = validate_finite_sample(exp_data, "exp_data")
    return float(stats.wasserstein_distance(model_samples, experimental_samples))


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
    model_samples = validate_finite_sample(model_data, "model_data")
    experimental_samples = validate_finite_sample(exp_data, "exp_data")
    statistic_result = stats.ks_2samp(model_samples, experimental_samples)
    return float(cast(float, statistic_result[0]))


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
    model_samples = validate_finite_sample(model_data, "model_data")
    experimental_samples = validate_finite_sample(exp_data, "exp_data")
    statistic_result = stats.cramervonmises_2samp(model_samples, experimental_samples)
    return float(statistic_result.statistic)


class ERelativeErrorNormalization(Enum):
    EXPERIMENT_MAX = auto()
    SIMULATION_MAX = auto()
    CONSTANT = auto()


@dataclass(slots=True)
class DeterministicMetricResult:
    absolute_error: float
    relative_error: float
    rmse: float
    nmse: float


def validate_relative_error_config(
    normalization: ERelativeErrorNormalization,
    constant: float | None,
) -> None:
    if not isinstance(normalization, ERelativeErrorNormalization):
        raise TypeError("Normalization must be ERelativeErrorNormalization.")
    if normalization is ERelativeErrorNormalization.CONSTANT:
        if (
            constant is None
            or not isinstance(constant, (int, float, np.integer, np.floating))
            or isinstance(constant, (bool, np.bool_))
            or not np.isfinite(constant)
            or constant <= 0
        ):
            raise ValueError(
                "Constant normalization requires a positive finite constant."
            )
    elif constant is not None:
        raise ValueError("A normalization constant requires CONSTANT normalization.")


def calculate_relative_error(
    simulation: np.ndarray,
    experiment: np.ndarray,
    normalization: ERelativeErrorNormalization,
    constant: float | None,
) -> float:
    if normalization is ERelativeErrorNormalization.EXPERIMENT_MAX:
        denominator = float(np.max(np.abs(experiment)))
    elif normalization is ERelativeErrorNormalization.SIMULATION_MAX:
        denominator = float(np.max(np.abs(simulation)))
    else:
        assert constant is not None
        denominator = constant
    if denominator == 0:
        return float("nan")
    return float(np.mean(np.abs(simulation - experiment)) / denominator)


def calc_u_pooling_1d(model_cdfs: list[ValECDF], exp_obs: np.ndarray) -> float:
    """Evaluate right-continuous ECDFs and integrate distance to continuous U(0,1)."""
    observations = validate_real_array(exp_obs, "exp_obs", allow_nan=False)
    if observations.ndim != 1 or not len(observations):
        raise ValueError("U-pooling needs nonempty paired 1D observations.")
    if len(model_cdfs) != len(observations):
        raise ValueError("One model CDF is required per observation.")
    if any(not isinstance(cdf, ValECDF) for cdf in model_cdfs):
        raise TypeError("Model CDFs must be ValECDF records.")
    transformed = np.empty(len(observations), dtype=np.float64)
    for index, (cdf, observation) in enumerate(
        zip(model_cdfs, observations, strict=True)
    ):
        support_index = np.searchsorted(cdf.quantiles, observation, side="right") - 1
        transformed[index] = (
            0.0 if support_index < 0 else cdf.probabilities[support_index]
        )
    transformed.sort()
    count = len(transformed)
    lower_probabilities = np.arange(count) / count
    upper_probabilities = np.arange(1, count + 1) / count
    # Antiderivative of |p - u| is (p - u)|p - u| / 2.
    lower_distance = lower_probabilities - transformed
    upper_distance = upper_probabilities - transformed
    return float(
        np.sum(
            (
                upper_distance * np.abs(upper_distance)
                - lower_distance * np.abs(lower_distance)
            )
            / 2
        )
    )


def validate_paired_samples(
    simulation: float | np.ndarray,
    experiment: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    simulation_array = np.atleast_1d(validate_real_array(simulation, "simulation"))
    experiment_array = np.atleast_1d(validate_real_array(experiment, "experiment"))
    if simulation_array.ndim != 1 or simulation_array.shape != experiment_array.shape:
        raise ValueError("Deterministic metrics require paired 1D observations.")
    paired = ~np.isnan(simulation_array) & ~np.isnan(experiment_array)
    if not np.any(paired):
        raise ValueError("Deterministic metrics need a finite observation pair.")
    return simulation_array[paired].astype(np.float64), experiment_array[paired].astype(
        np.float64
    )


def calc_deterministic_metrics_1d(
    sim_val: float | np.ndarray,
    exp_val: float | np.ndarray,
    *,
    normalization: ERelativeErrorNormalization = ERelativeErrorNormalization.EXPERIMENT_MAX,
    constant: float | None = None,
) -> DeterministicMetricResult:
    """Return paired errors; NMSE is NaN for variance at or below the zero tolerance."""
    validate_relative_error_config(normalization, constant)
    simulation, experiment = validate_paired_samples(sim_val, exp_val)
    residual = simulation - experiment
    mean_square_error = float(np.mean(residual * residual))
    experimental_variance = float(np.var(experiment))
    return DeterministicMetricResult(
        absolute_error=float(np.mean(np.abs(residual))),
        relative_error=calculate_relative_error(
            simulation, experiment, normalization, constant
        ),
        rmse=float(np.sqrt(mean_square_error)),
        nmse=mean_square_error / experimental_variance
        if experimental_variance > METRIC_ZERO_TOLERANCE
        else float("nan"),
    )
