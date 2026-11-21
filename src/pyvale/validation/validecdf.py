# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Extract conditional empirical distributions without plotting or pooling axes."""

from copy import copy
from itertools import product

import numpy as np
from scipy import stats

from pyvale.validation.arraychecks import _validate_real_array
from pyvale.validation.validchecks import (
    _validate_ecdf,
    _validate_finite_sample,
)
from pyvale.validation.ecdftypes import (
    EValidECDFBounds,
    EValidECDFCentre,
    EValidECDFReduce,
    ValidECDF,
    ValidECDFBounds,
    ValidECDFSummary,
    ValidECDFView,
    ValidECDFViewData,
)
from pyvale.validation.validtypes import (
    EEpisOffset,
    EValidAxis,
    ValidArray,
)


def calc_ecdf(samples: np.ndarray) -> ValidECDF:
    """Calculate a 1D ECDF, omitting NaNs and preserving duplicate mass."""
    observations = _validate_finite_sample(samples, "samples")
    cdf = stats.ecdf(observations).cdf
    return ValidECDF(cdf.quantiles, cdf.probabilities)


def extract_ecdf(
    quantity: ValidArray,
    *,
    point_idx: int = 0,
    component_idx: int = 0,
    time_idx: int = 0,
    epis_trans_idx: int = 0,
    epis_nuis_idx: int = 0,
    offset: EEpisOffset = EEpisOffset.NOMINAL,
) -> ValidECDF:
    """Extract one conditional ECDF using physical and epistemic axis indices.

    Only ALEATORY supplies observations. Epistemic scenarios are selected, never
    pooled. The offset coherently shifts every observation in the selected cell.
    """
    samples = quantity.select_aleatory(
        point_idx,
        component_idx,
        time_idx,
        epis_trans_idx=epis_trans_idx,
        epis_nuis_idx=epis_nuis_idx,
        offset=offset,
    )
    return calc_ecdf(samples)


def calc_ecdf_envelope(
    samples: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise ECDF bounds over coherent, supplied sample scenarios."""
    if not samples:
        raise ValueError("At least one sample scenario is required.")
    clean = [np.sort(_validate_finite_sample(sample, "sample")) for sample in samples]
    support = np.unique(np.concatenate(clean))
    cdfs = np.stack([np.searchsorted(s, support, side="right") / len(s) for s in clean])
    return support, np.min(cdfs, axis=0), np.max(cdfs, axis=0)


def extract_ecdf_envelope(
    quantity: ValidArray,
    *,
    point_idx: int = 0,
    component_idx: int = 0,
    time_idx: int = 0,
    epis_trans_idx: int | None = None,
    epis_nuis_idx: int | None = None,
    offset: EEpisOffset | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract pointwise ECDF bounds, fixing or spanning epistemic axes.

    An integer fixes that axis index; None spans all indices on that axis.
    offset=None includes nominal/lower/upper when offsets exist, otherwise only
    nominal. Returns (support, lower_probability, upper_probability), each (n,).
    """
    samples = extract_aleatory_scenarios(
        quantity,
        point_idx=point_idx,
        component_idx=component_idx,
        time_idx=time_idx,
        epis_trans_idx=epis_trans_idx,
        epis_nuis_idx=epis_nuis_idx,
        offset=offset,
    )
    return calc_ecdf_envelope(samples)


def extract_aleatory_scenarios(
    quantity: ValidArray,
    *,
    point_idx: int = 0,
    component_idx: int = 0,
    time_idx: int = 0,
    epis_trans_idx: int | None = None,
    epis_nuis_idx: int | None = None,
    offset: EEpisOffset | None = EEpisOffset.NOMINAL,
) -> list[np.ndarray]:
    """Select complete conditional samples in trans/nuis/offset product order.

    None spans an epistemic axis; offset=None includes all available offsets.
    The returned arrays can share the input's memory for nominal selections.
    """
    selections = []
    for axis, selection in (
        (EValidAxis.EPISTEMIC_TRANS, epis_trans_idx),
        (EValidAxis.EPISTEMIC_NUIS, epis_nuis_idx),
    ):
        size = (
            quantity.data.shape[quantity.axes.index(axis)]
            if axis in quantity.axes
            else 1
        )
        selections.append(range(size) if selection is None else (selection,))
    offsets = (
        (tuple(EEpisOffset) if quantity.offsets is not None else (EEpisOffset.NOMINAL,))
        if offset is None
        else (offset,)
    )
    samples = []
    for trans_index, nuisance_index, selected_offset in product(*selections, offsets):
        samples.append(
            quantity.select_aleatory(
                point_idx,
                component_idx,
                time_idx,
                epis_trans_idx=trans_index,
                epis_nuis_idx=nuisance_index,
                offset=selected_offset,
            )
        )
    return samples


def evaluate_ecdf_quantiles(cdf: ValidECDF, probabilities: np.ndarray) -> np.ndarray:
    """Evaluate the inverse ECDF using the first support reaching each probability."""
    _validate_ecdf(cdf)
    probabilities = _validate_real_array(probabilities, "probabilities", allow_nan=False)
    if probabilities.ndim != 1:
        raise ValueError("Probabilities must be a finite one-dimensional array.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Probabilities must lie in [0, 1].")
    return _evaluate_ecdf_quantiles(cdf, probabilities)


def _evaluate_ecdf_quantiles(cdf: ValidECDF, probabilities: np.ndarray) -> np.ndarray:
    return cdf.quantiles[np.searchsorted(cdf.probabilities, probabilities, side="left")]


def calc_ecdf_summary(
    samples: list[np.ndarray],
    *,
    bounds: EValidECDFBounds = EValidECDFBounds.POINTWISE,
) -> ValidECDFSummary:
    """Summarise conditional distributions without mixing epistemic scenarios.

    SUPPORT_SUM reproduces the paper script's complete-scenario selection by
    sums of unique support values. It differs from pointwise probability boxes.
    """
    if not samples:
        raise ValueError("Supply at least one conditional sample.")
    if not isinstance(bounds, EValidECDFBounds):
        raise TypeError("Bounds must be EValidECDFBounds.")
    cdfs = [calc_ecdf(sample) for sample in samples]
    probabilities = np.unique(np.concatenate([cdf.probabilities for cdf in cdfs]))
    nominal = np.zeros_like(probabilities)
    lower = np.full_like(probabilities, np.inf)
    upper = np.full_like(probabilities, -np.inf)
    lower_idx = None
    upper_idx = None
    if bounds is EValidECDFBounds.SUPPORT_SUM:
        support_sums = np.array([np.sum(cdf.quantiles) for cdf in cdfs])
        lower_idx = int(np.argmin(support_sums))
        upper_idx = int(np.argmax(support_sums))
    for selection_idx, cdf in enumerate(cdfs):
        quantiles = _evaluate_ecdf_quantiles(cdf, probabilities)
        nominal += quantiles
        if bounds is EValidECDFBounds.POINTWISE:
            np.minimum(lower, quantiles, out=lower)
            np.maximum(upper, quantiles, out=upper)
        else:
            if selection_idx == lower_idx:
                lower[:] = quantiles
            if selection_idx == upper_idx:
                upper[:] = quantiles
    nominal /= len(cdfs)
    return ValidECDFSummary(
        probabilities,
        lower,
        upper,
        nominal,
        bounds,
        lower_idx,
        upper_idx,
    )


def calc_ecdf_view(
    quantity: ValidArray,
    *,
    view: ValidECDFView | None = None,
    point_idx: int = 0,
    component_idx: int = 0,
    time_idx: int = 0,
    offset: EEpisOffset | None = EEpisOffset.NOMINAL,
) -> ValidECDFViewData:
    """Calculate conditional quantile curves without pooling epistemic samples.

    Select each axis first, then reduce nuisance before transferable uncertainty.
    Mean/median are pointwise quantile operations with equal scenario weights;
    they do not assign probabilities to epistemic uncertainty. Original scenario
    overlays remain unchanged by reductions. None offset explicitly spans every
    available coherent offset; the default retains nominal values only.
    """
    selected_view = ValidECDFView() if view is None else view
    if not isinstance(selected_view, ValidECDFView):
        raise TypeError("ECDF view must be ValidECDFView.")
    if not isinstance(selected_view.centre, EValidECDFCentre):
        raise TypeError("ECDF centre must be EValidECDFCentre.")
    axis_indices: list[tuple[int, ...]] = []
    for axis, indices, reduction in (
        (
            EValidAxis.EPISTEMIC_TRANS,
            selected_view.trans_idxs,
            selected_view.trans_reduce,
        ),
        (EValidAxis.EPISTEMIC_NUIS, selected_view.nuis_idxs, selected_view.nuis_reduce),
    ):
        if not isinstance(reduction, EValidECDFReduce):
            raise TypeError("ECDF reduction must be EValidECDFReduce.")
        size = (
            quantity.data.shape[quantity.axes.index(axis)]
            if axis in quantity.axes
            else 1
        )
        if indices is None:
            indices = tuple(range(size))
        if not isinstance(indices, tuple) or not indices:
            raise ValueError("Epistemic index selections must be nonempty tuples.")
        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            for index in indices
        ):
            raise TypeError("Epistemic indices must be integers.")
        if any(index < 0 or index >= size for index in indices):
            raise IndexError("Epistemic index is outside the selected axis.")
        if len(set(indices)) != len(indices):
            raise ValueError("Epistemic index selections must be unique.")
        axis_indices.append(indices)
    for reduction, probability in (
        (selected_view.nuis_reduce, selected_view.nuis_quantile),
        (selected_view.trans_reduce, selected_view.trans_quantile),
    ):
        if reduction is EValidECDFReduce.QUANTILE:
            if (
                isinstance(probability, (bool, np.bool_))
                or not isinstance(probability, (int, float, np.integer, np.floating))
                or not np.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("Epistemic quantile probability must be in [0, 1].")
        elif probability is not None:
            raise ValueError("A quantile probability requires QUANTILE reduction.")
    trans_indices, nuis_indices = axis_indices
    if offset is not None and not isinstance(offset, EEpisOffset):
        raise TypeError("Offset must be EEpisOffset or None.")
    offsets = (
        (tuple(EEpisOffset) if quantity.offsets is not None else (EEpisOffset.NOMINAL,))
        if offset is None
        else (offset,)
    )
    scenario_pairs = selected_view.scenario_idxs
    if scenario_pairs is None:
        scenario_pairs = tuple(product(trans_indices, nuis_indices))
    if not isinstance(scenario_pairs, tuple):
        raise TypeError("Scenario selections must be a tuple of joint index pairs.")
    for pair in scenario_pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("Select scenarios using (transferable, nuisance) pairs.")
        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            for index in pair
        ):
            raise TypeError("Scenario indices must be integers.")
        if pair[0] not in trans_indices or pair[1] not in nuis_indices:
            raise IndexError("Scenario pair is outside the selected epistemic indices.")
    if len(set(scenario_pairs)) != len(scenario_pairs):
        raise ValueError("Scenario pairs must be unique.")
    scenarios = tuple(product(trans_indices, nuis_indices, offsets))
    cdfs = []
    sample_arrays = []
    for trans_idx, nuis_idx, selected_offset in scenarios:
        samples = quantity.select_aleatory(
            point_idx,
            component_idx,
            time_idx,
            epis_trans_idx=trans_idx,
            epis_nuis_idx=nuis_idx,
            offset=selected_offset,
        )
        cdfs.append(calc_ecdf(samples))
        sample_arrays.append(samples)
    probabilities = np.unique(np.concatenate([cdf.probabilities for cdf in cdfs]))
    original_quantiles = np.stack(
        [_evaluate_ecdf_quantiles(cdf, probabilities) for cdf in cdfs]
    )
    quantiles = original_quantiles.reshape(
        len(trans_indices), len(nuis_indices), len(offsets), probabilities.size
    )
    quantiles = _reduce_ecdf_quantiles(quantiles, selected_view)
    selected_scenarios = tuple(
        (trans_idx, nuis_idx, selected_offset)
        for trans_idx, nuis_idx in scenario_pairs
        for selected_offset in offsets
    )
    scenario_positions = {scenario: index for index, scenario in enumerate(scenarios)}
    overlay_indices = [scenario_positions[scenario] for scenario in selected_scenarios]
    overlay_quantiles = original_quantiles[overlay_indices]
    if (
        np.prod(quantiles.shape[:-1]) > 1
        and not selected_view.bounds
        and selected_view.centre is EValidECDFCentre.NONE
        and not selected_scenarios
    ):
        raise ValueError("Select bounds, a centre curve or scenario overlays.")
    return ValidECDFViewData(
        probabilities,
        quantiles,
        overlay_quantiles,
        selected_scenarios,
        original_quantiles,
        scenarios,
        tuple(sample_arrays),
    )


def _reduce_ecdf_quantiles(quantiles: np.ndarray, view: ValidECDFView) -> np.ndarray:
    """Reduce trusted quantile arrays in the same order for views and confidence bounds."""
    for axis, reduction, probability in (
        (1, view.nuis_reduce, view.nuis_quantile),
        (0, view.trans_reduce, view.trans_quantile),
    ):
        if reduction is EValidECDFReduce.MEAN:
            quantiles = np.mean(quantiles, axis=axis, keepdims=True)
        elif reduction is EValidECDFReduce.MEDIAN:
            quantiles = np.median(quantiles, axis=axis, keepdims=True)
        elif reduction is EValidECDFReduce.QUANTILE:
            assert probability is not None
            quantiles = np.quantile(quantiles, probability, axis=axis, keepdims=True)
    return quantiles


def _find_quantile_limits(
    quantiles: np.ndarray,
    scenarios: tuple[tuple[int, int, EEpisOffset], ...],
    view: ValidECDFView,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trans_indices = tuple(dict.fromkeys(scenario[0] for scenario in scenarios))
    nuis_indices = tuple(dict.fromkeys(scenario[1] for scenario in scenarios))
    offsets = tuple(dict.fromkeys(scenario[2] for scenario in scenarios))
    if view.trans_reduce is not EValidECDFReduce.KEEP:
        trans_indices = (-1,)
    if view.nuis_reduce is not EValidECDFReduce.KEEP:
        nuis_indices = (-1,)
    origins = np.asarray(
        tuple(product(trans_indices, nuis_indices, offsets)), dtype=int
    )
    flattened = quantiles.reshape(-1, quantiles.shape[-1])
    lower_indices = np.argmin(flattened, axis=0)
    upper_indices = np.argmax(flattened, axis=0)
    probability_indices = np.arange(flattened.shape[-1])
    return (
        flattened[lower_indices, probability_indices],
        flattened[upper_indices, probability_indices],
        origins[lower_indices],
        origins[upper_indices],
    )


def find_ecdf_epis_bounds(
    quantity: ValidArray,
    *,
    point_idx: int = 0,
    component_idx: int = 0,
    time_idx: int = 0,
    trans_idxs: tuple[int, ...] | None = None,
    nuis_idxs: tuple[int, ...] | None = None,
    nuis_reduce: EValidECDFReduce = EValidECDFReduce.KEEP,
    nuis_quantile: float | None = None,
    offset: EEpisOffset | None = EEpisOffset.NOMINAL,
) -> ValidECDFBounds:
    """Resolve nuisance uncertainty, then envelope all selected transferable cases.

    KEEP envelopes both axes without pooling observations. A singleton nuis_idxs
    conditions on one actual nuisance case. MEAN/MEDIAN/QUANTILE construct
    synthetic nuisance quantile summaries separately for each transferable case.
    Quantiles use linear interpolation between nuisance order statistics.
    """
    view = ValidECDFView(
        trans_idxs=trans_idxs,
        nuis_idxs=nuis_idxs,
        nuis_reduce=nuis_reduce,
        nuis_quantile=nuis_quantile,
    )
    data = calc_ecdf_view(
        quantity,
        view=view,
        point_idx=point_idx,
        component_idx=component_idx,
        time_idx=time_idx,
        offset=offset,
    )
    lower, upper, lower_indices, upper_indices = _find_quantile_limits(
        data.quantiles, data.original_scenarios, view
    )
    return ValidECDFBounds(
        data.probabilities, lower, upper, lower_indices, upper_indices, data, copy(view)
    )
