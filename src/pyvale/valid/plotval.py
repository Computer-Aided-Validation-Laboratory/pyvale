# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Visualization functions for validation metrics, CDFs, and p-boxes."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pyvale.valid.metrics import MAVMResult


def plot_mavm_cdf_1d(
    mavm_res: MAVMResult,
    title: str = "",
    unit: str = "",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots empirical CDFs, experimental confidence band, and MAVM intervals.

    Parameters
    ----------
    mavm_res : MAVMResult
        Result dataclass from calc_mavm_1d.
    title : str, optional
        Plot title.
    unit : str, optional
        Physical unit string for x-axis label.
    ax : plt.Axes | None, optional
        Matplotlib Axes to plot into. If None, creates new figure and axes.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    else:
        fig = ax.get_figure()

    # Step plots for empirical CDFs
    ax.step(
        mavm_res.model_quantiles,
        mavm_res.model_probs,
        where="post",
        color="crimson",
        linewidth=2,
        label="Model CDF",
    )
    ax.step(
        mavm_res.exp_quantiles,
        mavm_res.exp_probs,
        where="post",
        color="navy",
        linewidth=2,
        label="Exp CDF",
    )

    # Confidence interval band
    ax.step(
        mavm_res.exp_conf_lower,
        mavm_res.exp_probs,
        where="post",
        color="royalblue",
        linestyle="--",
        alpha=0.7,
        label=f"Exp {(1 - mavm_res.alpha) * 100:.0f}% CI Lower",
    )
    ax.step(
        mavm_res.exp_conf_upper,
        mavm_res.exp_probs,
        where="post",
        color="royalblue",
        linestyle="--",
        alpha=0.7,
        label=f"Exp {(1 - mavm_res.alpha) * 100:.0f}% CI Upper",
    )

    xlabel = f"Value [{unit}]" if unit else "Value"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)

    title_str = (
        f"{title}\n"
        f"$d^+={mavm_res.d_plus:.3f}$, $d^-={mavm_res.d_minus:.3f}$, "
        f"$d_{{\\text{{total}}}}={mavm_res.d_total:.3f}$"
    )
    ax.set_title(title_str.strip())
    ax.legend(loc="best", fontsize="small")

    return fig, ax


def plot_mavm_summary_bars(
    mavm_results: dict[str, MAVMResult],
    title: str = "MAVM Point Sensor Summary",
    unit: str = "",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a stacked/grouped bar chart of d+ and d- mismatch across sensors.

    Parameters
    ----------
    mavm_results : dict[str, MAVMResult]
        Dictionary mapping sensor labels to MAVMResult objects.
    title : str, optional
        Plot title.
    unit : str, optional
        Unit string.
    ax : plt.Axes | None, optional
        Matplotlib Axes.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    else:
        fig = ax.get_figure()

    labels = list(mavm_results.keys())
    dp_vals = [mavm_results[k].d_plus for k in labels]
    dm_vals = [mavm_results[k].d_minus for k in labels]

    x_indices = np.arange(len(labels))
    width = 0.35

    ax.bar(
        x_indices - width / 2,
        dp_vals,
        width,
        label="$d^+$ (Over-prediction)",
        color="crimson",
        alpha=0.85,
    )
    ax.bar(
        x_indices + width / 2,
        dm_vals,
        width,
        label="$d^-$ (Under-prediction)",
        color="royalblue",
        alpha=0.85,
    )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ylabel = f"MAVM Metric [{unit}]" if unit else "MAVM Metric"
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="best")

    return fig, ax
