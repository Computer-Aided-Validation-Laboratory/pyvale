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

from pyvale.valid.constants import MAVM_AREA_TOLERANCE
from pyvale.valid.metrics import calc_mavm_1d
from pyvale.valid.metricstrategy import EMetricType
from pyvale.valid.validecdf import (
    _evaluate_ecdf_quantiles,
    _find_quantile_limits,
    _reduce_ecdf_quantiles,
    calc_ecdf,
)
from pyvale.valid.validtools import build_metric_selection
from pyvale.valid.validtypes import (
    EEpisOffset,
    EMAVMDuplicates,
    EMetricOutput,
    EValidBoundAxis,
    EValidNuisanceScore,
    EValidResAxis,
    MAVMCorrection,
    ValidCase,
    ValidECDF,
    ValidECDFBounds,
    ValidMetricExtrema,
    ValidMetricField,
    ValidNuisanceRanking,
    ValidResult,
    ValidSupportBounds,
)

EPIS_AXES = (
    EValidResAxis.SIM_TRANS,
    EValidResAxis.SIM_NUIS,
    EValidResAxis.EXP_TRANS,
    EValidResAxis.EXP_NUIS,
)
NUIS_AXES = (EValidResAxis.SIM_NUIS, EValidResAxis.EXP_NUIS)


def _select_profile_field(
    field: ValidMetricField,
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
    selection = build_metric_selection(field, selectors)
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
    for output, full_values in field.outputs.items():
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
    field: ValidMetricField,
    selection: tuple[int | slice, ...],
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    shape = next(iter(field.outputs.values())).shape
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
) -> ValidSupportBounds:
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
    shape = next(iter(field.outputs.values())).shape
    retained_shape = tuple(shape[field.axes.index(axis)] for axis in retained)
    selected_axes = tuple(
        axis
        for axis, index in zip(field.axes, selection, strict=True)
        if isinstance(index, slice)
    )
    retained_positions = tuple(selected_axes.index(axis) for axis in retained)
    plus = field.outputs[
        EMetricOutput.D_PLUS if metric is EMetricType.MAVM else EMetricOutput.DISTANCE
    ]
    minus = field.outputs[
        EMetricOutput.D_MINUS if metric is EMetricType.MAVM else EMetricOutput.DISTANCE
    ]
    cdfs: dict[tuple[int, ...], ValidECDF] = {}
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
        np.unique(np.concatenate([cdf.probabilities for cdf in cdfs.values()]))
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
    return ValidSupportBounds(
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
    )


def profile_epis_nuisance_supports(
    valid_case: ValidCase,
    valid_result: ValidResult,
    observable: str,
    metric: EMetricType,
    *,
    selectors: dict[EValidResAxis, int] | None = None,
) -> ValidSupportBounds:
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
    supports: ValidSupportBounds,
    *,
    score: EValidNuisanceScore = EValidNuisanceScore.TOTAL_WIDTH,
) -> ValidNuisanceRanking:
    """Rank coherent nuisance cases after enveloping transferable uncertainty.

    Supply supports with nuisance axes retained and transferable axes reduced.
    TOTAL_WIDTH integrates shifted width; ADDED_WIDTH integrates the outward
    extension beyond the unshifted sim envelope. Ranks select actual cases,
    never a pointwise synthetic median. Equal scores preserve canonical order.
    """
    if not isinstance(score, EValidNuisanceScore):
        raise TypeError("Nuisance score must be EValidNuisanceScore.")
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
    if score is EValidNuisanceScore.ADDED_WIDTH:
        widths = np.maximum(0.0, supports.upper - supports.sim_upper) + np.maximum(
            0.0, supports.sim_lower - supports.lower
        )
    probabilities = np.diff(np.r_[0.0, supports.probabilities])
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
    return ValidNuisanceRanking(
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
    sim: ValidECDFBounds,
    exp_lower: ValidECDF,
    exp_upper: ValidECDF,
    *,
    directional: bool,
) -> ValidSupportBounds:
    probabilities = np.unique(
        np.r_[sim.probabilities, exp_lower.probabilities, exp_upper.probabilities]
    )
    sim_quantiles = np.stack(
        [
            _evaluate_ecdf_quantiles(ValidECDF(curve, sim.probabilities), probabilities)
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
        {EMetricOutput.D_PLUS: plus, EMetricOutput.D_MINUS: minus}
        if directional
        else {EMetricOutput.DISTANCE: plus}
    )
    return ValidSupportBounds(
        (),
        probabilities,
        lower_candidates[lower_origin, probability_indices],
        upper_candidates[upper_origin, probability_indices],
        sim_quantiles[0],
        sim_quantiles[1],
        (EValidBoundAxis.SIM_BOUND, EValidBoundAxis.EXP_BOUND),
        pair_indices[lower_origin],
        pair_indices[upper_origin],
        {},
        outputs,
    )


def calc_avm_over_epis_bounds(
    sim: ValidECDFBounds, exp: ValidECDFBounds
) -> ValidSupportBounds:
    """Compare all four bound combinations using exact empirical probability mass."""
    return _calc_metric_over_bounds(
        sim,
        ValidECDF(exp.lower, exp.probabilities),
        ValidECDF(exp.upper, exp.probabilities),
        directional=False,
    )


def calc_mavm_over_epis_bounds(
    sim: ValidECDFBounds,
    exp: ValidECDFBounds,
    *,
    alpha: float = 0.05,
) -> ValidSupportBounds:
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
    shifts = []
    for samples in exp.data.samples:
        support = np.unique(samples[np.isfinite(samples)])
        count = support.size
        shift = (
            0.0
            if count == 1
            else stats.t.ppf(1 - alpha, count - 1) * np.std(support) / np.sqrt(count)
        )
        shifts.append(shift)
    shape = (
        len({s[0] for s in exp.data.original_scenarios}),
        len({s[1] for s in exp.data.original_scenarios}),
        len({s[2] for s in exp.data.original_scenarios}),
        exp.probabilities.size,
    )
    original = exp.data.original_quantiles
    lower_quantiles = _reduce_ecdf_quantiles(
        (original - np.asarray(shifts)[:, None]).reshape(shape), exp.view
    )
    upper_quantiles = _reduce_ecdf_quantiles(
        (original + np.asarray(shifts)[:, None]).reshape(shape), exp.view
    )
    lower, _, _, _ = _find_quantile_limits(
        lower_quantiles, exp.data.original_scenarios, exp.view
    )
    _, upper, _, _ = _find_quantile_limits(
        upper_quantiles, exp.data.original_scenarios, exp.view
    )
    return _calc_metric_over_bounds(
        sim,
        ValidECDF(lower, exp.probabilities),
        ValidECDF(upper, exp.probabilities),
        directional=True,
    )


def calc_mavm_correction(
    model_lower: np.ndarray,
    model_upper: np.ndarray,
    exp_lower: np.ndarray,
    exp_upper: np.ndarray,
    *,
    alpha: float = 0.05,
    duplicates: EMAVMDuplicates = EMAVMDuplicates.INCLUDE,
    tol: float = MAVM_AREA_TOLERANCE,
) -> MAVMCorrection:
    """Compare four endpoint pairs and anchor corrections to the supplied model limits.

    The largest d_minus on model_lower selects the lower correction; the largest
    d_plus on model_upper selects the upper correction. These are support-shift
    diagnostics, not a reduction to the largest MAVM across every model scenario.
    """
    comparisons = tuple(
        calc_mavm_1d(model, exp, alpha=alpha, duplicates=duplicates, tol=tol)
        for exp in (exp_lower, exp_upper)
        for model in (model_lower, model_upper)
    )
    lower_exp_idx = int(np.argmax([comparisons[0].d_minus, comparisons[2].d_minus]))
    upper_exp_idx = int(np.argmax([comparisons[1].d_plus, comparisons[3].d_plus]))
    return MAVMCorrection(
        comparisons[2 * lower_exp_idx],
        comparisons[2 * upper_exp_idx + 1],
        (comparisons[0], comparisons[1], comparisons[2], comparisons[3]),
        lower_exp_idx,
        upper_exp_idx,
    )
