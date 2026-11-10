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

import numpy as np
from scipy import stats

from pyvale.valid.constants import (
    MAVM_AREA_TOLERANCE,
    METRIC_ZERO_TOLERANCE,
)
from pyvale.valid.cython.mavm import (
    _finite_1d,
    cyth_calc_mavm_1d,
    cyth_calc_mavm_pbox_1d,
)
from pyvale.valid.valtypes import EMAVMDuplicates, MAVMResult


def calc_mavm_1d(
    model_data: np.ndarray,
    exp_data: np.ndarray,
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
        Significance level for the Student's t confidence interval (default 0.05
        corresponding to a 95% confidence band).
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
        Confidence level significance (default 0.05 for 95% CI).
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
    u_vals = _finite_1d(model_data, "model_data")
    v_vals = _finite_1d(exp_data, "exp_data")
    return float(stats.wasserstein_distance(u_vals, v_vals))


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
    u_vals = _finite_1d(model_data, "model_data")
    v_vals = _finite_1d(exp_data, "exp_data")
    res = stats.ks_2samp(u_vals, v_vals)
    return float(res.statistic)


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
    u_vals = _finite_1d(model_data, "model_data")
    v_vals = _finite_1d(exp_data, "exp_data")
    res = stats.cramervonmises_2samp(u_vals, v_vals)
    return float(res.statistic)


def calc_u_pooling_1d(
    model_cdfs: list[tuple[np.ndarray, np.ndarray]],
    exp_obs: np.ndarray,
) -> float:
    """Calculates Ferson U-pooling validation metric.

    Transforms experimental observations through corresponding model CDFs and
    measures the distance of the transformed sample distribution from U(0, 1).

    Parameters
    ----------
    model_cdfs : list[tuple[np.ndarray, np.ndarray]]
        List of (probabilities, quantiles) pairs for each test condition.
    exp_obs : np.ndarray
        1D array of experimental observations corresponding to each model CDF.

    Returns
    -------
    float
        Wasserstein distance between the pooled u-values and Uniform(0, 1).
    """
    observations = np.asarray(exp_obs, dtype=np.float64)
    if observations.ndim != 1 or not np.all(np.isfinite(observations)):
        raise ValueError("U-pooling needs finite paired 1D observations.")
    if len(model_cdfs) != len(observations):
        raise ValueError("One model CDF is required per observation.")
    u_samples = []
    for (probs, quants), obs in zip(model_cdfs, observations, strict=True):
        # Interpolate empirical CDF probability at observed value
        u_val = float(np.interp(obs, quants, probs, left=0.0, right=1.0))
        u_samples.append(u_val)

    u_arr = np.array(u_samples)
    uniform_ref = np.linspace(0.0, 1.0, len(u_arr))
    return float(stats.wasserstein_distance(u_arr, uniform_ref))


def calc_deterministic_metrics_1d(
    sim_val: float | np.ndarray,
    exp_val: float | np.ndarray,
) -> dict[str, float]:
    """Calculates classical deterministic error metrics between arrays or
    scalars.

    Returns
    -------
    dict[str, float]
        Dictionary with absolute_error, relative_error, rmse, and nmse.
    """
    sim_arr = np.atleast_1d(np.asarray(sim_val, dtype=np.float64))
    exp_arr = np.atleast_1d(np.asarray(exp_val, dtype=np.float64))
    if sim_arr.ndim != 1 or exp_arr.ndim != 1 or sim_arr.shape != exp_arr.shape:
        raise ValueError("Deterministic metrics require paired 1D observations.")
    if np.any(np.isinf(sim_arr)) or np.any(np.isinf(exp_arr)):
        raise ValueError("Deterministic observations cannot contain infinity.")
    valid = np.isfinite(sim_arr) & np.isfinite(exp_arr)
    if not np.any(valid):
        raise ValueError("Deterministic metrics need a finite observation pair.")
    sim_arr = sim_arr[valid]
    exp_arr = exp_arr[valid]

    abs_err = np.abs(sim_arr - exp_arr)
    denom = np.where(
        np.abs(exp_arr) > METRIC_ZERO_TOLERANCE,
        np.abs(exp_arr),
        1.0,
    )
    rel_err = abs_err / denom

    mse = float(np.mean((sim_arr - exp_arr) ** 2))
    rmse = float(np.sqrt(mse))
    var_exp = float(np.var(exp_arr))
    nmse = float(mse / var_exp) if var_exp > METRIC_ZERO_TOLERANCE else float(mse)

    return {
        "absolute_error": float(np.mean(abs_err)),
        "relative_error": float(np.mean(rel_err)),
        "rmse": rmse,
        "nmse": nmse,
    }
