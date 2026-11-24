# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Epistemic bounds comparisons, metric reductions and nuisance profiling."""

from collections.abc import Iterator
from math import prod

import numpy as np
from scipy import stats

from pyvale.validation.constants import MAVM_AREA_TOLERANCE
from pyvale.validation.ecdftypes import (
    ECDFNuisanceRank,
    ECDFMetricBounds,
    EECDFBoundAxis,
    EECDFNuisanceScore,
    ECDF,
    ECDFEpisBounds,
)
from pyvale.validation.imetric import EMetricOut, EMetricType
from pyvale.validation.metrics import calc_mavm_1d
from pyvale.validation.validecdf import (
    _evaluate_ecdf_quantiles,
    _find_quantile_limits,
    _reduce_ecdf_quantiles,
    calc_ecdf,
)
from pyvale.validation.validtools import _build_metric_selection
from pyvale.validation.validtypes import (
    EEpisOffset,
    EMAVMDuplicates,
    EValidResAxis,
    MAVMBoundPairs,
    MetricField,
    ValidCase,
    ValidMetricExtrema,
    ValidResult,
)

EPIS_AXES = (
    EValidResAxis.SIM_TRANS,
    EValidResAxis.SIM_NUIS,
    EValidResAxis.EXP_TRANS,
    EValidResAxis.EXP_NUIS,
)
NUIS_AXES = (EValidResAxis.SIM_NUIS, EValidResAxis.EXP_NUIS)


def _select_profile_field(
    field: MetricField,
    reduce_axes: tuple[EValidResAxis, ...],
    selectors: dict[EValidResAxis, int] | None,
) -> tuple[
    tuple[int | slice, ...], tuple[EValidResAxis, ...], tuple[EValidResAxis, ...]
]:
    allowed = (*EPIS_AXES, EValidResAxis.SIM_OFFSET, EValidResAxis.EXP_OFFSET)
    if any(
        not isinstance(axis, EValidResAxis) or axis not in allowed
        for axis in reduce_axes
    ):
        raise ValueError(
            "Reduce only epistemic or offset axes; select physical cells instead."
        )
    if len(set(reduce_axes)) != len(reduce_axes):
        raise ValueError("Reduction axes must be unique.")
    selection = _build_metric_selection(field, selectors)
    selected_axes = tuple(
        axis
        for axis, index in zip(field.axes, selection, strict=True)
        if isinstance(index, slice)
    )
    reduced = tuple(axis for axis in selected_axes if axis in reduce_axes)
    retained = tuple(axis for axis in selected_axes if axis not in reduce_axes)
    return selection, retained, reduced


def find_metric_epis_extrema(
    valid_result: ValidResult,
    observable: str,
    metric: EMetricType,
    *,
    reduce_axes: tuple[EValidResAxis, ...] = EPIS_AXES,
    selectors: dict[EValidResAxis, int] | None = None,
) -> ValidMetricExtrema:
    """Reduce stored metric outputs without recalculating comparisons.

    Physical and unselected offset axes remain. Joint minima/maxima are taken
    over the requested axes, not a mixed minimax operation. Singleton omitted
    epistemic axes need no special handling. All-missing cells retain NaN/-1.
    """
    field = valid_result.metrics[observable][metric]
    selection, retained, reduced = _select_profile_field(field, reduce_axes, selectors)
    selected_axes = tuple(
        axis
        for axis, index in zip(field.axes, selection, strict=True)
        if isinstance(index, slice)
    )
    permutation = tuple(selected_axes.index(axis) for axis in (*retained, *reduced))
    minimum = {}
    maximum = {}
    minimum_indices = {}
    maximum_indices = {}
    for output, full_values in field.metric_vals.items():
        values = full_values[selection]
        ordered = np.transpose(values, permutation)
        retained_shape = ordered.shape[: len(retained)]
        reduced_shape = ordered.shape[len(retained) :]
        flattened = ordered.reshape(*retained_shape, prod(reduced_shape))
        finite = np.isfinite(flattened)
        missing = ~np.any(finite, axis=-1)
        lower_index = np.argmin(np.where(finite, flattened, np.inf), axis=-1)
        upper_index = np.argmax(np.where(finite, flattened, -np.inf), axis=-1)
        lower = np.take_along_axis(flattened, lower_index[..., None], axis=-1)[..., 0]
        upper = np.take_along_axis(flattened, upper_index[..., None], axis=-1)[..., 0]
        minimum[output] = np.where(missing, np.nan, lower)
        maximum[output] = np.where(missing, np.nan, upper)
        if reduced_shape:
            lower_origin = np.stack(
                np.unravel_index(lower_index, reduced_shape), axis=-1
            )
            upper_origin = np.stack(
                np.unravel_index(upper_index, reduced_shape), axis=-1
            )
            lower_origin = np.where(missing[..., None], -1, lower_origin)
            upper_origin = np.where(missing[..., None], -1, upper_origin)
        else:
            lower_origin = np.empty((*retained_shape, 0), dtype=int)
            upper_origin = np.empty((*retained_shape, 0), dtype=int)
        minimum_indices[output] = lower_origin
        maximum_indices[output] = upper_origin
    return ValidMetricExtrema(
        retained,
        reduced,
        minimum,
        maximum,
        minimum_indices,
        maximum_indices,
        dict(selectors or {}),
    )


def profile_epis_nuisance_extrema(
    valid_result: ValidResult,
    observable: str,
    metric: EMetricType,
    *,
    selectors: dict[EValidResAxis, int] | None = None,
) -> ValidMetricExtrema:
    """Profile both nuisance axes, retaining transferable and physical axes."""
    return find_metric_epis_extrema(
        valid_result, observable, metric, reduce_axes=NUIS_AXES, selectors=selectors
    )


def _iterate_profile_cells(
    field: MetricField,
    selection: tuple[int | slice, ...],
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    shape = next(iter(field.metric_vals.values())).shape
    selected_shape = tuple(
        size
        for size, index in zip(shape, selection, strict=True)
        if isinstance(index, slice)
    )
    for selected_index in np.ndindex(selected_shape):
        iterator = iter(selected_index)
        full_index = tuple(
            next(iterator) if isinstance(index, slice) else int(index)
            for index in selection
        )
        yield selected_index, full_index


def find_metric_epis_supports(
    valid_case: ValidCase,
    valid_result: ValidResult,
    observable: str,
    metric: EMetricType,
    *,
    reduce_axes: tuple[EValidResAxis, ...] = EPIS_AXES,
    selectors: dict[EValidResAxis, int] | None = None,
) -> ECDFMetricBounds:
    """Envelope original simulation quantiles shifted by stored AVM/MAVM values.

    MAVM uses Q_sim-d_minus and Q_sim+d_plus. AVM uses symmetric Q_sim+/-distance
    as a magnitude diagnostic, not a probability confidence statement. Origins
    may change with probability; a globally largest distance is never substituted
    for this search. No metric calculations are repeated.
    """
    if metric not in (EMetricType.AVM, EMetricType.MAVM):
        raise ValueError("ECDF support shifts are defined here only for AVM and MAVM.")
    field = valid_result.metrics[observable][metric]
    selection, retained, _ = _select_profile_field(field, reduce_axes, selectors)
    shape = next(iter(field.metric_vals.values())).shape
    retained_shape = tuple(shape[field.axes.index(axis)] for axis in retained)
    selected_axes = tuple(
        axis
        for axis, index in zip(field.axes, selection, strict=True)
        if isinstance(index, slice)
    )
    retained_positions = tuple(selected_axes.index(axis) for axis in retained)
    plus = field.metric_vals[
        EMetricOut.D_PLUS if metric is EMetricType.MAVM else EMetricOut.DISTANCE
    ]
    minus = field.metric_vals[
        EMetricOut.D_MINUS if metric is EMetricType.MAVM else EMetricOut.DISTANCE
    ]
    cdfs: dict[tuple[int, ...], ECDF] = {}
    # One ECDF per original sim cell, irrespective of how many exp cases use it.
    sim_axes = (
        EValidResAxis.POINT,
        EValidResAxis.COMPONENT,
        EValidResAxis.TIME,
        EValidResAxis.SIM_TRANS,
        EValidResAxis.SIM_NUIS,
        EValidResAxis.SIM_OFFSET,
    )
    for _, full_index in _iterate_profile_cells(field, selection):
        if not np.isfinite(plus[full_index]) and not np.isfinite(minus[full_index]):
            continue
        indices = dict(zip(field.axes, full_index, strict=True))
        key = tuple(indices.get(axis, 0) for axis in sim_axes)
        if key not in cdfs:
            samples = valid_case.sim[observable]._select_aleatory(
                key[0], key[1], key[2], key[3], key[4], EEpisOffset(key[5])
            )
            cdfs[key] = calc_ecdf(samples)
    probabilities = (
        np.unique(np.concatenate([cdf.probs for cdf in cdfs.values()]))
        if cdfs
        else np.array([1.0])
    )
    quantiles = {
        key: _evaluate_ecdf_quantiles(cdf, probabilities) for key, cdf in cdfs.items()
    }
    output_shape = (*retained_shape, probabilities.size)
    lower = np.full(output_shape, np.inf)
    upper = np.full(output_shape, -np.inf)
    sim_lower = np.full(output_shape, np.inf)
    sim_upper = np.full(output_shape, -np.inf)
    lower_indices = np.full((*output_shape, len(field.axes)), -1, dtype=int)
    upper_indices = np.full((*output_shape, len(field.axes)), -1, dtype=int)
    for selected_index, full_index in _iterate_profile_cells(field, selection):
        if not np.isfinite(plus[full_index]) and not np.isfinite(minus[full_index]):
            continue
        indices = dict(zip(field.axes, full_index, strict=True))
        key = tuple(indices.get(axis, 0) for axis in sim_axes)
        group = tuple(selected_index[position] for position in retained_positions)
        curve = quantiles[key]
        np.minimum(sim_lower[group], curve, out=sim_lower[group])
        np.maximum(sim_upper[group], curve, out=sim_upper[group])
        if np.isfinite(minus[full_index]):
            candidate = curve - minus[full_index]
            changed = candidate < lower[group]
            lower[group][changed] = candidate[changed]
            lower_indices[group][changed] = full_index
        if np.isfinite(plus[full_index]):
            candidate = curve + plus[full_index]
            changed = candidate > upper[group]
            upper[group][changed] = candidate[changed]
            upper_indices[group][changed] = full_index
    for array in (lower, upper, sim_lower, sim_upper):
        array[~np.isfinite(array)] = np.nan
    return ECDFMetricBounds(
        retained,
        probabilities,
        lower,
        upper,
        sim_lower,
        sim_upper,
        field.axes,
        lower_indices,
        upper_indices,
        dict(selectors or {}),
        metric=metric,
    )


def profile_epis_nuisance_supports(
    valid_case: ValidCase,
    valid_result: ValidResult,
    observable: str,
    metric: EMetricType,
    *,
    selectors: dict[EValidResAxis, int] | None = None,
) -> ECDFMetricBounds:
    """Profile shifted supports over nuisance, retaining transferable axes."""
    return find_metric_epis_supports(
        valid_case,
        valid_result,
        observable,
        metric,
        reduce_axes=NUIS_AXES,
        selectors=selectors,
    )


def rank_epis_nuisance_cases(
    supports: ECDFMetricBounds,
    *,
    score: EECDFNuisanceScore = EECDFNuisanceScore.TOTAL_WIDTH,
) -> ECDFNuisanceRank:
    """Rank coherent nuisance cases after enveloping transferable uncertainty.

    Supply supports with nuisance axes retained and transferable axes reduced.
    TOTAL_WIDTH integrates shifted width; ADDED_WIDTH integrates the outward
    extension beyond the unshifted sim envelope. Ranks select actual cases,
    never a pointwise synthetic median. Equal scores preserve canonical order.
    """
    if not isinstance(score, EECDFNuisanceScore):
        raise TypeError("Nuisance score must be EECDFNuisanceScore.")
    if any(
        axis in supports.axes
        for axis in (EValidResAxis.SIM_TRANS, EValidResAxis.EXP_TRANS)
    ):
        raise ValueError("Reduce transferable axes before ranking nuisance envelopes.")
    case_axes = tuple(axis for axis in supports.axes if axis in NUIS_AXES)
    if not case_axes:
        raise ValueError("Retain at least one nuisance axis before ranking cases.")
    retained = tuple(axis for axis in supports.axes if axis not in case_axes)
    widths = supports.upper - supports.lower
    if score is EECDFNuisanceScore.ADDED_WIDTH:
        widths = np.maximum(0.0, supports.upper - supports.sim_upper) + np.maximum(
            0.0, supports.sim_lower - supports.lower
        )
    probabilities = np.diff(np.r_[0.0, supports.probs])
    scores = np.sum(widths * probabilities, axis=-1)
    permutation = tuple(supports.axes.index(axis) for axis in (*retained, *case_axes))
    ordered_scores = np.transpose(scores, permutation)
    case_shape = ordered_scores.shape[len(retained) :]
    scores = ordered_scores.reshape(
        *ordered_scores.shape[: len(retained)], prod(case_shape)
    )
    cases = np.asarray(tuple(np.ndindex(case_shape)), dtype=int)
    order = np.argsort(
        np.where(np.isfinite(scores), scores, np.inf), axis=-1, kind="stable"
    )
    finite = np.isfinite(scores)
    valid_counts = np.asarray(np.count_nonzero(finite, axis=-1))
    order = np.where(np.take_along_axis(finite, order, axis=-1), order, -1)
    return ECDFNuisanceRank(
        retained,
        case_axes,
        cases,
        scores,
        order,
        valid_counts,
        score,
        dict(supports.selectors),
    )


def _calc_metric_over_bounds(
    sim: ECDFEpisBounds,
    exp_lower: ECDF,
    exp_upper: ECDF,
    *,
    directional: bool,
) -> ECDFMetricBounds:
    probabilities = np.unique(
        np.r_[sim.view.probs, exp_lower.probs, exp_upper.probs]
    )
    sim_quantiles = np.stack(
        [
            _evaluate_ecdf_quantiles(ECDF(curve, sim.view.probs), probabilities)
            for curve in (sim.lower, sim.upper)
        ]
    )
    exp_quantiles = np.stack(
        [_evaluate_ecdf_quantiles(cdf, probabilities) for cdf in (exp_lower, exp_upper)]
    )
    weights = np.diff(np.r_[0.0, probabilities])
    plus = np.empty((2, 2))
    minus = np.empty((2, 2))
    for sim_index in range(2):
        for exp_index in range(2):
            residual = exp_quantiles[exp_index] - sim_quantiles[sim_index]
            if directional:
                plus[sim_index, exp_index] = np.dot(weights, np.maximum(residual, 0.0))
                minus[sim_index, exp_index] = np.dot(
                    weights, np.maximum(-residual, 0.0)
                )
            else:
                plus[sim_index, exp_index] = np.dot(weights, np.abs(residual))
                minus[sim_index, exp_index] = plus[sim_index, exp_index]
    lower_candidates = (sim_quantiles[:, None, :] - minus[..., None]).reshape(4, -1)
    upper_candidates = (sim_quantiles[:, None, :] + plus[..., None]).reshape(4, -1)
    lower_origin = np.argmin(lower_candidates, axis=0)
    upper_origin = np.argmax(upper_candidates, axis=0)
    probability_indices = np.arange(probabilities.size)
    pair_indices = np.asarray(tuple(np.ndindex(2, 2)))
    outputs = (
        {EMetricOut.D_PLUS: plus, EMetricOut.D_MINUS: minus}
        if directional
        else {EMetricOut.DISTANCE: plus}
    )
    return ECDFMetricBounds(
        (),
        probabilities,
        lower_candidates[lower_origin, probability_indices],
        upper_candidates[upper_origin, probability_indices],
        sim_quantiles[0],
        sim_quantiles[1],
        (EECDFBoundAxis.SIM_BOUND, EECDFBoundAxis.EXP_BOUND),
        pair_indices[lower_origin],
        pair_indices[upper_origin],
        {},
        outputs,
        metric=EMetricType.MAVM if directional else EMetricType.AVM,
    )


def calc_avm_over_epis_bounds(
    sim: ECDFEpisBounds, exp: ECDFEpisBounds
) -> ECDFMetricBounds:
    """Compare all four bound combinations using exact empirical probability mass."""
    return _calc_metric_over_bounds(
        sim,
        ECDF(exp.lower, exp.view.probs),
        ECDF(exp.upper, exp.view.probs),
        directional=False,
    )


def _mavm_confidence_shifts(
    sample_sets: tuple[np.ndarray, ...], alpha: float
) -> np.ndarray:
    shifts = []
    for samples in sample_sets:
        support = np.unique(samples[np.isfinite(samples)])
        count = support.size
        shift = (
            0.0
            if count == 1
            else stats.t.ppf(1 - alpha, count - 1) * np.std(support) / np.sqrt(count)
        )
        shifts.append(shift)
    return np.asarray(shifts)


def calc_mavm_over_epis_bounds(
    sim: ECDFEpisBounds,
    exp: ECDFEpisBounds,
    *,
    alpha: float = 0.05,
) -> ECDFMetricBounds:
    """Construct experimental confidence bounds before epistemic enveloping.

    Each original experimental scenario uses the existing MAVM convention:
    Student-t(1-alpha, n_unique-1) times population std(unique support)/sqrt(n_unique).
    The chosen nuisance reduction is then applied to those confidence curves.
    Synthetic envelope vertices are never treated as experimental observations.
    All four sim/confidence-bound combinations compete for both shifted limits.
    This is distinct from MAVM over every original scenario pair.
    """
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not np.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError("MAVM alpha must be between zero and one.")
    view = exp.view
    original = view.orig_quants
    shifts = _mavm_confidence_shifts(view.samples, alpha)
    shifts = shifts.reshape(*original.shape[:-1], 1)
    lower_quantiles = _reduce_ecdf_quantiles(original - shifts, view.opts)
    upper_quantiles = _reduce_ecdf_quantiles(original + shifts, view.opts)
    lower, _, _, _ = _find_quantile_limits(
        lower_quantiles, view.orig_epis_offsets, view.opts
    )
    _, upper, _, _ = _find_quantile_limits(
        upper_quantiles, view.orig_epis_offsets, view.opts
    )
    return _calc_metric_over_bounds(
        sim,
        ECDF(lower, exp.view.probs),
        ECDF(upper, exp.view.probs),
        directional=True,
    )


def calc_mavm_bound_pairs(
    sim_lower: np.ndarray,
    sim_upper: np.ndarray,
    exp_lower: np.ndarray,
    exp_upper: np.ndarray,
    *,
    alpha: float = 0.05,
    duplicates: EMAVMDuplicates = EMAVMDuplicates.INCLUDE,
    tol: float = MAVM_AREA_TOLERANCE,
) -> MAVMBoundPairs:
    """Compare four endpoint pairs and anchor corrections to the supplied sim limits.

    The largest d_minus on sim_lower selects the lower correction; the largest
    d_plus on sim_upper selects the upper correction. These are support-shift
    diagnostics, not a reduction to the largest MAVM across every sim scenario.
    """
    comparisons = tuple(
        calc_mavm_1d(sim, exp, alpha=alpha, duplicates=duplicates, tol=tol)
        for exp in (exp_lower, exp_upper)
        for sim in (sim_lower, sim_upper)
    )
    lower_exp_idx = int(np.argmax([comparisons[0].d_minus, comparisons[2].d_minus]))
    upper_exp_idx = int(np.argmax([comparisons[1].d_plus, comparisons[3].d_plus]))
    return MAVMBoundPairs(
        comparisons[2 * lower_exp_idx],
        comparisons[2 * upper_exp_idx + 1],
        (comparisons[0], comparisons[1], comparisons[2], comparisons[3]),
        lower_exp_idx,
        upper_exp_idx,
    )
