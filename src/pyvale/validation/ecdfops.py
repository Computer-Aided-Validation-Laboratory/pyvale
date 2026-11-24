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
    EECDFReduce,
    ECDF,
    ECDFEpisBounds,
    ECDFViewSel,
    ECDFView,
)
from pyvale.validation.validtypes import (
    EEpisOffset,
    EValidAxis,
    ValidArray,
    _get_axis_size,
    _select_aleatory_cases,
)


def calc_ecdf(samples: np.ndarray) -> ECDF:
    """Calculate a 1D ECDF, omitting NaNs and preserving duplicate mass.

    Parameters
    ----------
    samples : ndarray
        Aleatory observations of shape (n_aleatory,) for one conditional
        distribution.

    Returns
    -------
    ECDF
        Sorted distinct observations and cumulative empirical masses, each with
        shape (n_support,). NaNs are omitted; repeated values contribute their full
        mass.
    """
    observations = _validate_finite_sample(samples, "samples")
    cdf = stats.ecdf(observations).cdf
    return ECDF(cdf.quantiles, cdf.probabilities)


def evaluate_ecdf_quantiles(cdf: ECDF, probabilities: np.ndarray) -> np.ndarray:
    """Evaluate the inverse ECDF using the first support reaching each probability.

    Parameters
    ----------
    cdf : ECDF
        Empirical distribution with matching one-dimensional support and probability
        arrays.
    probabilities : ndarray
        One-dimensional cumulative probabilities, with shape (n_probs,).

    Returns
    -------
    ndarray
        Quantiles of shape (n_probs,), in input probability order. Each value is the
        first support point whose cumulative mass reaches the requested probability.
    """
    _validate_ecdf(cdf)
    probabilities = _validate_real_array(probabilities, "probabilities", allow_nan=False)

    if probabilities.ndim != 1:
        raise ValueError("Probabilities must be a finite one-dimensional array.")

    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Probabilities must lie in [0, 1].")

    return _evaluate_ecdf_quantiles(cdf, probabilities)


def _evaluate_ecdf_quantiles(cdf: ECDF, probabilities: np.ndarray) -> np.ndarray:
    """Evaluate inverse ECDF quantiles on a trusted grid.

    Parameters
    ----------
    cdf : ECDF
        Empirical distribution with matching one-dimensional support and probability
        arrays.
    probabilities : ndarray
        One-dimensional cumulative probabilities, with shape (n_probs,).

    Returns
    -------
    ndarray
        Quantiles of shape (n_probs,), selected using a left-sided probability
        search.
    """
    return cdf.quants[np.searchsorted(cdf.probs, probabilities, side="left")]


def select_ecdf_support_sum_cases(samples: tuple[np.ndarray, ...]) -> tuple[int, int]:
    """Select whole cases by the minimum and maximum sums of unique support.

    Parameters
    ----------
    samples : tuple[ndarray, ...]
        Tuple of original conditional sample arrays. Each array has shape
        (n_aleatory,) for its case; lengths may differ.

    Returns
    -------
    tuple[int, int]
        Indices of the cases with the minimum and maximum sum of distinct finite
        support values, in that order. Ties choose the first case.
    """

    if not samples:
        raise ValueError("Supply at least one conditional sample.")

    support_sums = [
        np.sum(np.unique(_validate_finite_sample(sample, "sample")))
        for sample in samples
    ]

    return int(np.argmin(support_sums)), int(np.argmax(support_sums))


def calc_ecdf_view(
    quantity: ValidArray,
    *,
    view: ECDFViewSel | None = None,
    point_idx: int = 0,
    component_idx: int = 0,
    time_idx: int = 0,
    offset: EEpisOffset | None = EEpisOffset.NOMINAL,
) -> ECDFView:
    """Calculate conditional quantile curves without pooling epistemic samples.

    Select each axis first, then reduce nuisance before transferable uncertainty.
    Mean/median are pointwise quantile operations with equal case weights;
    they do not assign probabilities to epistemic uncertainty. Original case
    overlays remain unchanged by reductions. None offset explicitly spans every
    available coherent offset; the default retains nominal values only.

    Parameters
    ----------
    quantity : ValidArray
        Observations paired with their canonical named axes.
    view : ECDFViewSel or None, optional
        Epistemic selection and reduction settings. Use None for the default
        selection.
    point_idx : int, optional
        Zero-based physical sampling point index.
    component_idx : int, optional
        Zero-based observable component index.
    time_idx : int, optional
        Zero-based evaluation time index.
    offset : EEpisOffset or None, optional
        Coherent offset to apply. Use None to include nominal and all available
        offsets.

    Returns
    -------
    ECDFView
        Original and reduced quantiles on a common grid, with axes (transferable,
        nuisance, offset, probability). Reduced dimensions remain singleton axes.
    """
    selected_view = ECDFViewSel() if view is None else view

    if not isinstance(selected_view, ECDFViewSel):
        raise TypeError("ECDF view must be ECDFViewSel.")

    axis_indices: list[tuple[int, ...]] = []

    for axis, indices, reduction in (
        (
            EValidAxis.EPISTEMIC_TRANS,
            selected_view.trans_idxs,
            selected_view.trans_reduce,
        ),
        (EValidAxis.EPISTEMIC_NUIS, selected_view.nuis_idxs, selected_view.nuis_reduce),
    ):
        if not isinstance(reduction, EECDFReduce):
            raise TypeError("ECDF reduction must be EECDFReduce.")

        size = _get_axis_size(quantity, axis)

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
        if reduction is EECDFReduce.QUANTILE:
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

    case_labels, sample_arrays = _select_aleatory_cases(
        quantity,
        point_idx=point_idx,
        component_idx=component_idx,
        time_idx=time_idx,
        trans_idxs=trans_indices,
        nuis_idxs=nuis_indices,
        offset=offset,
    )
    cdfs = [calc_ecdf(samples) for samples in sample_arrays]

    probabilities = np.unique(np.concatenate([cdf.probs for cdf in cdfs]))

    original_quantiles = np.stack(
        [_evaluate_ecdf_quantiles(cdf, probabilities) for cdf in cdfs]
    )

    original_quantiles = original_quantiles.reshape(
        len(trans_indices), len(nuis_indices), -1, probabilities.size
    )

    quantiles = _reduce_ecdf_quantiles(original_quantiles, selected_view)

    return ECDFView(
        probabilities,
        quantiles,
        original_quantiles,
        case_labels,
        sample_arrays,
        copy(selected_view),
    )


def _reduce_ecdf_quantiles(quantiles: np.ndarray, view: ECDFViewSel) -> np.ndarray:
    """Reduce trusted quantile arrays in the same order for views and confidence bounds.

    Parameters
    ----------
    quantiles : ndarray
        Quantile array of shape (n_trans, n_nuis, n_offsets, n_probs).
    view : ECDFViewSel
        Selection settings that declare the nuisance and transferable reductions.

    Returns
    -------
    ndarray
        Reduced quantiles retaining both epistemic axes as singleton dimensions
        where reduced.
    """

    for axis, reduction, probability in (
        (1, view.nuis_reduce, view.nuis_quantile),
        (0, view.trans_reduce, view.trans_quantile),
    ):
        if reduction is EECDFReduce.MEAN:
            quantiles = np.mean(quantiles, axis=axis, keepdims=True)
        elif reduction is EECDFReduce.MEDIAN:
            quantiles = np.median(quantiles, axis=axis, keepdims=True)
        elif reduction is EECDFReduce.QUANTILE:
            assert probability is not None
            quantiles = np.quantile(quantiles, probability, axis=axis, keepdims=True)

    return quantiles


def _find_quantile_limits(
    quantiles: np.ndarray,
    case_labels: tuple[tuple[int, int, EEpisOffset], ...],
    view: ECDFViewSel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find pointwise inverse ECDF extrema and their original case labels.

    Parameters
    ----------
    quantiles : ndarray
        Reduced quantiles of shape (n_trans, n_nuis, n_offsets, n_probs).
    case_labels : tuple[tuple[int, int, EEpisOffset], ...]
        Original (transferable index, nuisance index, offset) labels in row-major
        order.
    view : ECDFViewSel
        Reduction settings that identify axes with synthetic cases.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray]
        Lower and upper quantiles of shape (n_probs,), followed by their two origin
        arrays of shape (n_probs, 3). An origin index of -1 denotes a reduced axis.
    """
    trans_indices = tuple(dict.fromkeys(case_label[0] for case_label in case_labels))
    nuis_indices = tuple(dict.fromkeys(case_label[1] for case_label in case_labels))
    offsets = tuple(dict.fromkeys(case_label[2] for case_label in case_labels))

    if view.trans_reduce is not EECDFReduce.KEEP:
        trans_indices = (-1,)

    if view.nuis_reduce is not EECDFReduce.KEEP:
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


def calc_ecdf_epis_bounds(view: ECDFView) -> ECDFEpisBounds:
    """Find pointwise quantile bounds and their controlling epistemic cases.

    Parameters
    ----------
    view : ECDFView
        Computed ECDF view with aligned quantile curves and original case metadata.

    Returns
    -------
    ECDFEpisBounds
        Pointwise lower and upper quantiles on the view's shared probability grid,
        with controlling case indices.
    """
    lower, upper, lower_indices, upper_indices = _find_quantile_limits(
        view.quants, view.orig_epis_offsets, view.opts
    )
    return ECDFEpisBounds(lower, upper, lower_indices, upper_indices, view)
