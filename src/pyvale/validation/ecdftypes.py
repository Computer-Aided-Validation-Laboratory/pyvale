# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""ECDF records, scenario reductions and shifted support bounds."""

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
    """

    quants: np.ndarray
    probs: np.ndarray


class EECDFReduce(Enum):
    KEEP = auto()
    MEAN = auto()
    MEDIAN = auto()
    QUANTILE = auto()


class EECDFCentre(Enum):
    NONE = auto()
    MEAN = auto()
    MEDIAN = auto()


@dataclass(slots=True)
class ECDFViewOpts:
    """Select scenarios before nuisance-first, then transferable quantile reduction.

    None index selections retain every index. scenario_idxs overlays original
    (transferable, nuisance) pairs: () draws none, None draws all selected pairs.
    Bounds and centre summarise the reduced family, never pooled observations.
    Offsets are selected separately by the plotting/calculation function.
    """

    trans_reduce: EECDFReduce = EECDFReduce.KEEP
    nuis_reduce: EECDFReduce = EECDFReduce.KEEP
    trans_quantile: float | None = None
    nuis_quantile: float | None = None
    trans_idxs: tuple[int, ...] | None = None
    nuis_idxs: tuple[int, ...] | None = None
    bounds: bool = True
    centre: EECDFCentre = EECDFCentre.NONE
    scenario_idxs: tuple[tuple[int, int], ...] | None = ()


@dataclass(slots=True)
class ECDFView:
    """Quantiles on a shared probability grid, retaining singleton reduced axes.

    quants has axes (transferable, nuisance, offset, probability).
    orig_quants has the same axis order before reduction. orig_epis_offsets and
    samples follow the flattened (transferable, nuisance, offset) order.
    overlay_indices select rows in that order.
    """

    probs: np.ndarray
    quants: np.ndarray
    orig_quants: np.ndarray
    orig_epis_offsets: tuple[tuple[int, int, EEpisOffset], ...]
    samples: tuple[np.ndarray, ...]
    overlay_indices: tuple[int, ...]
    opts: ECDFViewOpts


@dataclass(slots=True)
class ECDFEpisBounds:
    """Pointwise quantile limits and their controlling epistemic indices.

    Index columns are (transferable, nuisance, offset). A reduced axis has
    index -1. The source view retains inputs for confidence calculations.
    """

    lower: np.ndarray
    upper: np.ndarray
    lower_indices: np.ndarray
    upper_indices: np.ndarray
    view: ECDFView


class EECDFBoundAxis(Enum):
    SIM_BOUND = auto()
    EXP_BOUND = auto()


@dataclass(slots=True)
class ECDFMetricBounds:
    """Shifted quantile limits, with controlling original comparison indices.

    lower/upper and unshifted limits have shape ``(*retained_axes, probability)``.
    Origin indices append an index_axes dimension; NaN limits have index -1.
    Probability grids exclude zero: the first quantile also applies down to zero.
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
    TOTAL_WIDTH = auto()
    ADDED_WIDTH = auto()


@dataclass(slots=True)
class ECDFNuisanceRank:
    """Whole-case ranking; widths never splice different nuisance cases together.

    case_axes identifies columns in cases. Scores and order have shape
    ``(*retained_axes, n_cases)``. order contains row indices into cases, with -1
    for missing ranks at the end. valid_counts gives the usable rank count per
    retained cell. Equal scores preserve canonical case order.
    """

    axes: tuple[EValidResAxis, ...]
    case_axes: tuple[EValidResAxis, ...]
    cases: np.ndarray
    scores: np.ndarray
    order: np.ndarray
    valid_counts: np.ndarray
    score: EECDFNuisanceScore
    selectors: dict[EValidResAxis, int] = field(default_factory=dict)
