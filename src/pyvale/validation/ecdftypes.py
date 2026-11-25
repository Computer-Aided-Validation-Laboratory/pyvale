# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""ECDF records, epistemic reductions and shifted support bounds."""

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from pyvale.validation.imetric import EMetricOut, EMetricType
from pyvale.validation.validtypes import EEpisOffset, EValidResAxis


@dataclass(slots=True)
class ECDF:
    """Borrow support/probabilities, each shaped (n_support,).

    Support and cumulative probabilities increase strictly; probabilities end
    at one. Arrays must remain unchanged while the ECDF is in use.

    Attributes
    ----------
    quants : ndarray
        Strictly increasing observed support points of shape (n_support,), in
        observable units.
    probs : ndarray
        Strictly increasing cumulative mass of shape (n_support,), ending at one and
        aligned with quants.
    """

    quants: np.ndarray
    probs: np.ndarray


class EECDFReduce(Enum):
    """Operations that reduce an epistemic axis at fixed cumulative probability."""

    KEEP = auto()
    MEAN = auto()
    MEDIAN = auto()
    QUANTILE = auto()


@dataclass(slots=True)
class ECDFViewSel:
    """Select epistemic cases before nuisance-first quantile reduction.

    None index selections retain every index. Offsets are selected separately.

    Attributes
    ----------
    trans_reduce : EECDFReduce
        Reduction applied to the transferable axis after the nuisance reduction.
    nuis_reduce : EECDFReduce
        Reduction applied first, separately within each transferable case.
    trans_quantile : float or None
        Quantile level in [0, 1] for a transferable QUANTILE reduction.
    nuis_quantile : float or None
        Quantile level in [0, 1] for a nuisance QUANTILE reduction.
    trans_idxs : tuple[int, ...] or None
        Transferable case indices in selection order. Use None for all cases.
    nuis_idxs : tuple[int, ...] or None
        Nuisance case indices in selection order. Use None for all cases.
    """

    trans_reduce: EECDFReduce = EECDFReduce.KEEP
    nuis_reduce: EECDFReduce = EECDFReduce.KEEP
    trans_quantile: float | None = None
    nuis_quantile: float | None = None

    trans_idxs: tuple[int, ...] | None = None
    nuis_idxs: tuple[int, ...] | None = None


@dataclass(slots=True)
class ECDFView:
    """Quantiles on a shared probability grid, retaining singleton reduced axes.

    quants has axes (transferable, nuisance, offset, probability).
    orig_quants has the same axis order before reduction. orig_epis_offsets and
    samples follow the flattened (transferable, nuisance, offset) order.

    Attributes
    ----------
    probs : ndarray
        Shared cumulative probability grid of shape (n_probs,).
    quants : ndarray
        Reduced quantiles of shape (n_trans, n_nuis, n_offsets, n_probs); reduced
        axes have length one.
    orig_quants : ndarray
        Quantiles before reduction, in the same axis order as quants.
    orig_epis_offsets : tuple[tuple[int, int, EEpisOffset], ...]
        Original (transferable index, nuisance index, offset) for each flattened
        original curve.
    samples : tuple[ndarray, ...]
        Original one-dimensional aleatory sample arrays, ordered like
        orig_epis_offsets.
    opts : ECDFViewSel
        Selection and reduction settings used to produce the view.
    """

    probs: np.ndarray
    quants: np.ndarray
    orig_quants: np.ndarray

    orig_epis_offsets: tuple[tuple[int, int, EEpisOffset], ...]
    samples: tuple[np.ndarray, ...]
    opts: ECDFViewSel


@dataclass(slots=True)
class ECDFEpisBounds:
    """Pointwise quantile limits and their controlling epistemic indices.

    Index columns are (transferable, nuisance, offset). A reduced axis has
    index -1. The source view retains inputs for confidence calculations.

    Attributes
    ----------
    lower : ndarray
        Lowest quantile at each grid probability, with shape (n_probs,).
    upper : ndarray
        Highest quantile at each grid probability, with shape (n_probs,).
    lower_indices : ndarray
        Controlling lower-case labels of shape (n_probs, 3).
    upper_indices : ndarray
        Controlling upper-case labels of shape (n_probs, 3).
    view : ECDFView
        Source view, including original samples used for confidence calculations.
    """

    lower: np.ndarray
    upper: np.ndarray

    lower_indices: np.ndarray
    upper_indices: np.ndarray
    view: ECDFView


class EECDFBoundAxis(Enum):
    """Identify the simulation and experiment bound choices in endpoint comparisons."""

    SIM_BOUND = auto()
    EXP_BOUND = auto()


@dataclass(slots=True)
class ECDFMetricBounds:
    """Shifted quantile limits, with controlling original comparison indices.

    lower/upper and unshifted limits have shape ``(*retained_axes, probability)``.
    Origin indices append an index_axes dimension; NaN limits have index -1.
    Probability grids exclude zero: the first quantile also applies down to zero.

    Attributes
    ----------
    axes : tuple[EValidResAxis, ...]
        Retained result axes leading the probability dimension.
    probs : ndarray
        Shared cumulative probability grid of shape (n_probs,).
    lower : ndarray
        Metric-shifted lower quantiles of shape ``(*retained_shape, n_probs)``.
    upper : ndarray
        Metric-shifted upper quantiles with the same shape as lower.
    sim_lower : ndarray
        Unshifted lower simulation quantiles with the same shape as lower.
    sim_upper : ndarray
        Unshifted upper simulation quantiles with the same shape as upper.
    index_axes : tuple[EValidResAxis or EECDFBoundAxis, ...]
        Labels for the final dimension of the controlling-index arrays.
    lower_indices : ndarray
        Controlling comparison indices of shape ``(*retained_shape, n_probs,
        n_index_axes)`` for lower.
    upper_indices : ndarray
        Controlling comparison indices with the same shape as lower_indices, for
        upper.
    selectors : dict[EValidResAxis, int]
        Fixed result indices used to construct the bounds.
    metric_outputs : dict[EMetricOut, ndarray] or None
        Optional metric values for each simulation-bound and experimental-bound
        combination, of shape (2, 2).
    metric : EMetricType or None
        Metric used to shift the simulation support, if available.
    """

    axes: tuple[EValidResAxis, ...]
    probs: np.ndarray

    lower: np.ndarray
    upper: np.ndarray
    sim_lower: np.ndarray
    sim_upper: np.ndarray

    index_axes: tuple[EValidResAxis | EECDFBoundAxis, ...]
    lower_indices: np.ndarray
    upper_indices: np.ndarray
    selectors: dict[EValidResAxis, int]

    metric_outputs: dict[EMetricOut, np.ndarray] | None = None
    metric: EMetricType | None = None


class EECDFNuisanceScore(Enum):
    """Score widths by their total value or their increase over the simulation width."""

    TOTAL_WIDTH = auto()
    ADDED_WIDTH = auto()


@dataclass(slots=True)
class ECDFNuisanceRank:
    """Whole-case ranking; widths never splice different nuisance cases together.

    case_axes identifies columns in cases. Scores and order have shape
    ``(*retained_axes, n_cases)``. order contains row indices into cases, with -1
    for missing ranks at the end. valid_counts gives the usable rank count per
    retained cell. Equal scores preserve canonical case order.

    Attributes
    ----------
    axes : tuple[EValidResAxis, ...]
        Retained result axes before the case or rank dimension.
    case_axes : tuple[EValidResAxis, ...]
        Reduced nuisance axes represented by the columns of cases.
    cases : ndarray
        Indexed nuisance combinations of shape (n_cases, n_case_axes).
    scores : ndarray
        Integrated widths of shape ``(*retained_shape, n_cases)``.
    order : ndarray
        Case row indices in ascending score order, of shape ``(*retained_shape,
        n_cases)``.
    valid_counts : ndarray
        Number of finite scores per retained cell, of shape retained_shape.
    score : EECDFNuisanceScore
        Width definition used for the ranking.
    selectors : dict[EValidResAxis, int]
        Fixed result indices used to construct the ranked bounds.
    """

    axes: tuple[EValidResAxis, ...]
    case_axes: tuple[EValidResAxis, ...]
    cases: np.ndarray

    scores: np.ndarray
    order: np.ndarray
    valid_counts: np.ndarray

    score: EECDFNuisanceScore
    selectors: dict[EValidResAxis, int] = field(default_factory=dict)
