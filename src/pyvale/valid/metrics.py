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
import numpy as np
from scipy import stats


@dataclass(slots=True)
class MAVMResult:
    """Container for Modified Area Validation Metric (MAVM) calculation
    results.
    """

    d_plus: float
    """Positive mismatch area (model exceeds upper experimental bound)."""

    d_minus: float
    """Negative mismatch area (model falls below lower experimental bound)."""

    d_total: float
    """Total mismatch distance (d_plus + d_minus)."""

    model_quantiles: np.ndarray
    """Quantiles (support values) of the model empirical CDF."""

    model_probs: np.ndarray
    """Cumulative probabilities corresponding to model quantiles."""

    exp_quantiles: np.ndarray
    """Quantiles (support values) of the experimental empirical CDF."""

    exp_probs: np.ndarray
    """Cumulative probabilities corresponding to experimental quantiles."""

    exp_conf_lower: np.ndarray
    """Lower confidence bound quantiles for the experimental CDF."""

    exp_conf_upper: np.ndarray
    """Upper confidence bound quantiles for the experimental CDF."""

    alpha: float
    """Significance level used for the confidence interval (e.g. 0.05)."""


def _integrate_mavm_bound(
    f_mod: np.ndarray,
    sn_exp: np.ndarray,
    p_f: float,
    p_sn: float,
    tol: float = 1e-12,
) -> tuple[float, float]:
    """Integrates positive and negative areas between model CDF and exp bound.
    """
    n_exp = len(sn_exp)
    s_mod = len(f_mod)

    d_plus = 0.0
    d_minus = 0.0
    d_rem = 0.0
    ii = 0

    if n_exp > s_mod:
        for jj in range(s_mod):
            if abs(d_rem) > tol:
                d_ii = (sn_exp[ii] - f_mod[jj]) * (
                    p_sn * (ii + 1) - p_f * jj
                )
                if d_ii > 0.0:
                    d_plus += d_ii
                else:
                    d_minus += d_ii
                ii += 1

            while (jj + 1) * p_f > (ii + 1) * p_sn:
                # Bug fix from paper: step weight is experimental probability
                d_ii = (sn_exp[ii] - f_mod[jj]) * p_sn
                if d_ii > 0.0:
                    d_plus += d_ii
                else:
                    d_minus += d_ii
                ii += 1

            if ii < n_exp and jj < s_mod:
                d_rem = (sn_exp[ii] - f_mod[jj]) * (
                    p_f * (jj + 1) - p_sn * ii
                )
                if d_rem > 0.0:
                    d_plus += d_rem
                else:
                    d_minus += d_rem
    else:
        for jj in range(n_exp):
            if abs(d_rem) > tol:
                d_ii = (sn_exp[jj] - f_mod[ii]) * (
                    p_f * (ii + 1) - p_sn * jj
                )
                if d_ii > 0.0:
                    d_plus += d_ii
                else:
                    d_minus += d_ii
                ii += 1

            while (ii + 1) * p_f < (jj + 1) * p_sn:
                d_ii = (sn_exp[jj] - f_mod[ii]) * p_f
                if d_ii > 0.0:
                    d_plus += d_ii
                else:
                    d_minus += d_ii
                ii += 1

            if ii < s_mod and jj < n_exp:
                d_rem = (sn_exp[jj] - f_mod[ii]) * (
                    p_sn * (jj + 1) - p_f * ii
                )
                if d_rem > 0.0:
                    d_plus += d_rem
                else:
                    d_minus += d_rem

    return d_plus, d_minus


def calc_mavm_1d(
    model_data: np.ndarray,
    exp_data: np.ndarray,
    alpha: float = 0.05,
    tol: float = 1e-12,
) -> MAVMResult:
    """Calculates the Modified Area Validation Metric (MAVM) between 1D arrays.

    Implements the area metric with Student's t confidence bounds on the
    experimental empirical CDF (Whiting et al., 2023).

    Parameters
    ----------
    model_data : np.ndarray
        1D array of simulated / model realization values.
    exp_data : np.ndarray
        1D array of experimental observation values.
    alpha : float, optional
        Significance level for the Student's t confidence interval (default 0.05
        corresponding to a 95% confidence band).
    tol : float, optional
        Floating-point comparison tolerance for interval boundaries.

    Returns
    -------
    MAVMResult
        Complete result dataclass with d+, d-, d_total, and CDF data.
    """
    m_clean = np.asarray(model_data, dtype=np.float64).ravel()
    m_clean = m_clean[~np.isnan(m_clean)]
    e_clean = np.asarray(exp_data, dtype=np.float64).ravel()
    e_clean = e_clean[~np.isnan(e_clean)]

    if len(m_clean) == 0 or len(e_clean) == 0:
        raise ValueError("Cannot calculate MAVM on empty or all-NaN data.")

    model_cdf = stats.ecdf(m_clean).cdf
    exp_cdf = stats.ecdf(e_clean).cdf

    f_mod_vec = np.array(model_cdf.quantiles, dtype=np.float64)
    sn_exp_vec = np.array(exp_cdf.quantiles, dtype=np.float64)

    s_num_mod = len(f_mod_vec)
    n_num_exp = len(sn_exp_vec)

    df = n_num_exp - 1
    t_val = stats.t.ppf(1.0 - alpha, df) if df >= 1 else 0.0
    se = (
        np.nanstd(sn_exp_vec, ddof=1) / np.sqrt(n_num_exp)
        if n_num_exp > 1
        else 0.0
    )

    sn_conf_lower = sn_exp_vec - t_val * se
    sn_conf_upper = sn_exp_vec + t_val * se

    p_f_mod = 1.0 / s_num_mod
    p_sn_exp = 1.0 / n_num_exp

    # d+ is area where model exceeds upper exp bound
    _, dm_upper = _integrate_mavm_bound(
        f_mod_vec, sn_conf_upper, p_f_mod, p_sn_exp, tol
    )
    # d- is area where model falls below lower exp bound
    dp_lower, _ = _integrate_mavm_bound(
        f_mod_vec, sn_conf_lower, p_f_mod, p_sn_exp, tol
    )

    d_plus = float(max(0.0, abs(dm_upper)))
    d_minus = float(max(0.0, dp_lower))
    d_total = d_plus + d_minus

    return MAVMResult(
        d_plus=d_plus,
        d_minus=d_minus,
        d_total=d_total,
        model_quantiles=f_mod_vec,
        model_probs=np.array(model_cdf.probabilities, dtype=np.float64),
        exp_quantiles=sn_exp_vec,
        exp_probs=np.array(exp_cdf.probabilities, dtype=np.float64),
        exp_conf_lower=sn_conf_lower,
        exp_conf_upper=sn_conf_upper,
        alpha=alpha,
    )


def calc_mavm_pbox_1d(
    model_pbox_min: np.ndarray,
    model_pbox_max: np.ndarray,
    exp_data: np.ndarray,
    alpha: float = 0.05,
    tol: float = 1e-12,
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
    tol : float, optional
        Tolerance.

    Returns
    -------
    MAVMResult
        MAVM result comparing the p-box against experimental confidence bounds.
    """
    res_min = calc_mavm_1d(model_pbox_min, exp_data, alpha=alpha, tol=tol)
    res_max = calc_mavm_1d(model_pbox_max, exp_data, alpha=alpha, tol=tol)

    d_plus = res_max.d_plus
    d_minus = res_min.d_minus

    return MAVMResult(
        d_plus=d_plus,
        d_minus=d_minus,
        d_total=d_plus + d_minus,
        model_quantiles=res_min.model_quantiles,
        model_probs=res_min.model_probs,
        exp_quantiles=res_min.exp_quantiles,
        exp_probs=res_min.exp_probs,
        exp_conf_lower=res_min.exp_conf_lower,
        exp_conf_upper=res_min.exp_conf_upper,
        alpha=alpha,
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
    u_vals = np.asarray(model_data, dtype=np.float64).ravel()
    u_vals = u_vals[~np.isnan(u_vals)]
    v_vals = np.asarray(exp_data, dtype=np.float64).ravel()
    v_vals = v_vals[~np.isnan(v_vals)]
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
    u_vals = np.asarray(model_data, dtype=np.float64).ravel()
    u_vals = u_vals[~np.isnan(u_vals)]
    v_vals = np.asarray(exp_data, dtype=np.float64).ravel()
    v_vals = v_vals[~np.isnan(v_vals)]
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
    u_vals = np.asarray(model_data, dtype=np.float64).ravel()
    u_vals = u_vals[~np.isnan(u_vals)]
    v_vals = np.asarray(exp_data, dtype=np.float64).ravel()
    v_vals = v_vals[~np.isnan(v_vals)]
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
    u_samples = []
    for (probs, quants), obs in zip(model_cdfs, exp_obs, strict=False):
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
    sim_arr = np.asarray(sim_val, dtype=np.float64).ravel()
    sim_arr = sim_arr[~np.isnan(sim_arr)]
    exp_arr = np.asarray(exp_val, dtype=np.float64).ravel()
    exp_arr = exp_arr[~np.isnan(exp_arr)]

    if len(sim_arr) == 0 or len(exp_arr) == 0:
        raise ValueError("Cannot calculate metrics on empty data.")

    if sim_arr.shape != exp_arr.shape:
        sim_mean = float(np.mean(sim_arr))
        exp_mean = float(np.mean(exp_arr))
        abs_err = abs(sim_mean - exp_mean)
        denom = abs(exp_mean) if abs(exp_mean) > 1e-12 else 1.0
        rel_err = abs_err / denom
        var_exp = float(np.var(exp_arr))
        nmse = (
            float((abs_err**2) / var_exp)
            if var_exp > 1e-12
            else float(abs_err**2)
        )
        return {
            "absolute_error": abs_err,
            "relative_error": rel_err,
            "rmse": abs_err,
            "nmse": nmse,
        }

    abs_err = np.abs(sim_arr - exp_arr)
    denom = np.where(np.abs(exp_arr) > 1e-12, np.abs(exp_arr), 1.0)
    rel_err = abs_err / denom

    mse = float(np.mean((sim_arr - exp_arr) ** 2))
    rmse = float(np.sqrt(mse))
    var_exp = float(np.var(exp_arr))
    nmse = float(mse / var_exp) if var_exp > 1e-12 else float(mse)

    return {
        "absolute_error": float(np.mean(abs_err)),
        "relative_error": float(np.mean(rel_err)),
        "rmse": rmse,
        "nmse": nmse,
    }
