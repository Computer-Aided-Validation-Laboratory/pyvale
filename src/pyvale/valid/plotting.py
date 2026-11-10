"""Diagnostic plots for validation samples and labelled metric fields."""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pyvale.valid.core import EValAxis, EValSource, ValCase
from pyvale.valid.engine import (
    EOffsetScenario,
    EResultAxis,
    ValMetricField,
    ValResult,
    select_aleatory,
)
from pyvale.valid.metrics import calc_mavm_1d
from pyvale.valid.strategy import EMetricType, MetricMAVM


def _metric_units(result: ValResult, observable: str, metric: EMetricType) -> str:
    if metric in (EMetricType.KS, EMetricType.CVM, EMetricType.RELATIVE_ERROR):
        return "1"
    return result.domain.observables[observable].units


def plot_val_cdfs(
    case: ValCase,
    result: ValResult,
    observable: str,
    *,
    point: int = 0,
    component: int = 0,
    time: int = 0,
    sim_trans: int = 0,
    sim_nuis: int = 0,
    exp_trans: int = 0,
    exp_nuis: int = 0,
    sim_offset: EOffsetScenario = EOffsetScenario.NOMINAL,
    exp_offset: EOffsetScenario = EOffsetScenario.NOMINAL,
    metric: MetricMAVM | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Show the exact conditional samples compared at one physical cell."""
    if observable not in result.fields:
        raise KeyError(observable)
    sim = select_aleatory(
        case.sim.quantities[observable],
        point,
        component,
        time,
        sim_trans,
        sim_nuis,
        sim_offset,
    )
    exp = select_aleatory(
        case.exp.quantities[observable],
        point,
        component,
        time,
        exp_trans,
        exp_nuis,
        exp_offset,
    )
    sim = sim[np.isfinite(sim)]
    exp = exp[np.isfinite(exp)]
    if len(sim) == 0 or len(exp) == 0:
        raise ValueError("Selected cell has no finite observations.")
    raw = None
    support_min = min(np.min(sim), np.min(exp))
    support_max = max(np.max(sim), np.max(exp))
    if metric is not None:
        raw = calc_mavm_1d(sim, exp, metric.alpha, metric.mode, metric.tol)
        support_min = min(support_min, np.min(raw.exp_conf_lower))
        support_max = max(support_max, np.max(raw.exp_conf_upper))
    margin = max((support_max - support_min) * 0.02, 1.0e-12)
    left = support_min - margin
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    else:
        fig = ax.figure
    for samples, label, color in (
        (sim, "Simulation", "crimson"),
        (exp, "Experiment", "navy"),
    ):
        cdf = stats.ecdf(samples).cdf
        ax.step(
            np.r_[left, cdf.quantiles],
            np.r_[0.0, cdf.probabilities],
            where="post",
            label=label,
            color=color,
        )
    if raw is not None:
        ax.step(
            np.r_[left, raw.exp_conf_lower],
            np.r_[0.0, raw.exp_probs],
            where="post",
            ls="--",
            color="royalblue",
            label="Exp lower confidence",
        )
        ax.step(
            np.r_[left, raw.exp_conf_upper],
            np.r_[0.0, raw.exp_probs],
            where="post",
            ls="--",
            color="steelblue",
            label="Exp upper confidence",
        )
        ax.set_title(f"MAVM: d+={raw.d_plus:.4g}, d-={raw.d_minus:.4g}")
    units = case.domain.observables[observable].units
    ax.set_xlabel(f"{observable} [{units}]")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    return fig, ax


def calc_ecdf_envelope(
    samples: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise ECDF bounds over coherent, supplied sample scenarios."""
    if not samples:
        raise ValueError("At least one sample scenario is required.")
    if any(sample.ndim != 1 or np.any(np.isinf(sample)) for sample in samples):
        raise ValueError("ECDF scenarios need finite or NaN 1D samples.")
    clean = [np.sort(s[np.isfinite(s)]) for s in samples]
    if any(len(s) == 0 for s in clean):
        raise ValueError("Each ECDF scenario needs a finite observation.")
    support = np.unique(np.concatenate(clean))
    cdfs = np.stack([np.searchsorted(s, support, side="right") / len(s) for s in clean])
    return support, np.min(cdfs, axis=0), np.max(cdfs, axis=0)


def plot_val_cdf_envelope(
    case: ValCase,
    observable: str,
    *,
    source: EValSource = EValSource.SIM,
    point: int = 0,
    component: int = 0,
    time: int = 0,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot pointwise envelope of complete conditional CDF scenarios."""
    if not isinstance(source, EValSource):
        raise TypeError("Source must be EValSource.SIM or EValSource.EXP.")
    quantity = getattr(case, source).quantities[observable]
    trans = (
        quantity.data.shape[quantity.axes.index(EValAxis.EPISTEMIC_TRANS)]
        if EValAxis.EPISTEMIC_TRANS in quantity.axes
        else 1
    )
    nuis = (
        quantity.data.shape[quantity.axes.index(EValAxis.EPISTEMIC_NUIS)]
        if EValAxis.EPISTEMIC_NUIS in quantity.axes
        else 1
    )
    offset_count = 3 if quantity.offsets is not None else 1
    samples = [
        select_aleatory(quantity, point, component, time, t, n, o)
        for t, n, o in product(range(trans), range(nuis), range(offset_count))
    ]
    support, lower, upper = calc_ecdf_envelope(samples)
    margin = max((support[-1] - support[0]) * 0.02, 1.0e-12)
    support = np.r_[support[0] - margin, support]
    lower = np.r_[0.0, lower]
    upper = np.r_[0.0, upper]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    else:
        fig = ax.figure
    ax.step(support, lower, where="post", label="Pointwise lower ECDF")
    ax.step(support, upper, where="post", label="Pointwise upper ECDF")
    ax.fill_between(support, lower, upper, step="post", alpha=0.2)
    units = case.domain.observables[observable].units
    ax.set_xlabel(f"{observable} [{units}]")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    return fig, ax


def _select_field(
    field: ValMetricField,
    selectors: dict[EResultAxis, int],
    keep: EResultAxis,
    values: np.ndarray | None = None,
) -> np.ndarray:
    if keep not in field.axes:
        raise ValueError(f"Result has no {keep} axis.")
    array = field.value if values is None else values
    indices: list[int | slice] = []
    for axis, size in zip(field.axes, array.shape, strict=True):
        if axis == keep:
            indices.append(slice(None))
        elif axis in selectors:
            indices.append(selectors[axis])
        elif size == 1:
            indices.append(0)
        else:
            raise ValueError(f"Select a {axis} index before plotting.")
    return array[tuple(indices)]


def plot_val_metric_map(
    result: ValResult,
    observable: str,
    metric: EMetricType,
    *,
    selectors: dict[EResultAxis, int] | None = None,
    direction: str = "value",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter a metric field over its registered 2D point coordinates."""
    field = result.get(observable, metric)
    values = getattr(field, direction)
    if values is None:
        raise ValueError(f"Metric has no {direction} field.")
    selected = _select_field(field, selectors or {}, EResultAxis.POINT, values)
    coords = result.domain.observables[observable].geometry.coords
    if coords is None:
        raise ValueError("Metric map needs spatial point coordinates.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    else:
        fig = ax.figure
    plot = ax.scatter(coords[:, 0], coords[:, 1], c=selected, cmap="viridis", s=65)
    ax.set_aspect("equal")
    coordinate_units = result.domain.observables[observable].geometry.coordinate_units
    suffix = f" [{coordinate_units}]" if coordinate_units else ""
    ax.set_xlabel(f"x{suffix}")
    ax.set_ylabel(f"y{suffix}")
    ax.set_title(f"{observable}: {metric.name} {direction}")
    fig.colorbar(plot, ax=ax, label=_metric_units(result, observable, metric))
    return fig, ax


def plot_val_metric_time(
    result: ValResult,
    observable: str,
    metric: EMetricType,
    *,
    selectors: dict[EResultAxis, int] | None = None,
    direction: str = "value",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one selected metric scenario against physical time."""
    field = result.get(observable, metric)
    values = getattr(field, direction)
    if values is None:
        raise ValueError(f"Metric has no {direction} field.")
    selected = _select_field(field, selectors or {}, EResultAxis.TIME, values)
    times = result.domain.observables[observable].times
    if times is None:
        raise ValueError("Time plot requires physical time coordinates.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    else:
        fig = ax.figure
    ax.plot(times, selected, marker="o")
    time_units = result.domain.observables[observable].time_units
    ax.set_xlabel(f"Time [{time_units}]" if time_units else "Time")
    ax.set_ylabel(f"{metric.name} [{_metric_units(result, observable, metric)}]")
    ax.set_title(observable)
    ax.grid(alpha=0.3)
    return fig, ax


def plot_val_metric_distribution(
    result: ValResult,
    observable: str,
    metric: EMetricType,
    *,
    selectors: dict[EResultAxis, int] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Histogram finite metric values over selected physical/scenario cells."""
    field = result.get(observable, metric)
    index = tuple((selectors or {}).get(axis, slice(None)) for axis in field.axes)
    values = field.value[index].ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite metric values selected.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    else:
        fig = ax.figure
    ax.hist(values, bins=min(20, len(values)))
    ax.set_xlabel(f"{metric.name} [{_metric_units(result, observable, metric)}]")
    ax.set_ylabel("Cell count")
    ax.set_title(observable)
    return fig, ax
