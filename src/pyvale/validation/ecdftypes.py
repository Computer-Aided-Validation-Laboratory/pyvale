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
class ValidECDF:
    """Borrow support/probabilities, each shaped (n_support,).

    Support and cumulative probabilities increase strictly; probabilities end
    at one. Arrays must remain unchanged while the ECDF is in use.
    """

    quantiles: np.ndarray
    probabilities: np.ndarray


class EValidECDFReduce(Enum):
    KEEP = auto()
    MEAN = auto()
    MEDIAN = auto()
    QUANTILE = auto()


class EValidECDFCentre(Enum):
    NONE = auto()
    MEAN = auto()
    MEDIAN = auto()


@dataclass(slots=True)
class ValidECDFView:
    """Select scenarios before nuisance-first, then transferable quantile reduction.

    None index selections retain every index. scenario_idxs overlays original
    (transferable, nuisance) pairs: () draws none, None draws all selected pairs.
    Bounds and centre summarise the reduced family, never pooled observations.
    Offsets are selected separately by the plotting/calculation function.
    """

    trans_reduce: EValidECDFReduce = EValidECDFReduce.KEEP
    nuis_reduce: EValidECDFReduce = EValidECDFReduce.KEEP
    trans_quantile: float | None = None
    nuis_quantile: float | None = None
    trans_idxs: tuple[int, ...] | None = None
    nuis_idxs: tuple[int, ...] | None = None
    bounds: bool = True
    centre: EValidECDFCentre = EValidECDFCentre.NONE
    scenario_idxs: tuple[tuple[int, int], ...] | None = ()


@dataclass(slots=True)
class ValidECDFViewData:
    """Quantiles on a shared probability grid, retaining singleton reduced axes.

    quantiles has axes (transferable, nuisance, offset, probability).
    scenario_quantiles has axes (selected original scenario, probability).
    scenarios preserves original joint indices and coherent offset labels.
    """

    probabilities: np.ndarray
    quantiles: np.ndarray
    scenario_quantiles: np.ndarray
    scenarios: tuple[tuple[int, int, EEpisOffset], ...]
    original_quantiles: np.ndarray
    original_scenarios: tuple[tuple[int, int, EEpisOffset], ...]
    samples: tuple[np.ndarray, ...]


@dataclass(slots=True)
class ValidECDFBounds:
    """Pointwise quantile limits with original samples and reduction provenance.

    Scenario index arrays have columns (transferable, nuisance, offset).
    -1 identifies a synthetic mean/median/quantile axis rather than an original
    scenario. data and view retain the information needed for confidence bounds.
    """

    probabilities: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    lower_indices: np.ndarray
    upper_indices: np.ndarray
    data: ValidECDFViewData
    view: ValidECDFView


class EValidECDFBounds(Enum):
    POINTWISE = auto()
    SUPPORT_SUM = auto()


@dataclass(slots=True)
class ValidECDFSummary:
    """Conditional quantile curves on a shared probability grid, each shaped (n,).

    nominal is the arithmetic mean of conditional quantiles, not a pooled CDF.
    SUPPORT_SUM selects whole scenarios and does not guarantee pointwise bounds.
    Indices refer to input scenario order; pointwise bounds have no single index.
    """

    probabilities: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    nominal: np.ndarray
    bounds: EValidECDFBounds
    lower_idx: int | None
    upper_idx: int | None


class EValidBoundAxis(Enum):
    SIM_BOUND = auto()
    EXP_BOUND = auto()


@dataclass(slots=True)
class ECDFSuppBounds:
    """Shifted quantile limits, with controlling original comparison indices.

    lower/upper and unshifted limits have shape (*retained_axes, probability).
    Origin indices append an index_axes dimension; NaN limits have index -1.
    Probability grids exclude zero: the first quantile also applies down to zero.
    """

    axes: tuple[EValidResAxis, ...]
    probabilities: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sim_lower: np.ndarray
    sim_upper: np.ndarray
    index_axes: tuple[EValidResAxis | EValidBoundAxis, ...]
    lower_indices: np.ndarray
    upper_indices: np.ndarray
    selectors: dict[EValidResAxis, int]
    metric_outputs: dict[EMetricOut, np.ndarray] | None = None
    metric: EMetricType | None = None


class EValidNuisanceScore(Enum):
    TOTAL_WIDTH = auto()
    ADDED_WIDTH = auto()


@dataclass(slots=True)
class ECDFNuisanceRank:
    """Whole-case ranking; widths never splice different nuisance cases together.

    case_axes identifies columns in cases. Scores and order have shape
    (*retained_axes, n_cases). order contains row indices into cases, with -1
    for missing ranks at the end. valid_counts gives the usable rank count per
    retained cell. Equal scores preserve canonical case order.
    """

    axes: tuple[EValidResAxis, ...]
    case_axes: tuple[EValidResAxis, ...]
    cases: np.ndarray
    scores: np.ndarray
    order: np.ndarray
    valid_counts: np.ndarray
    score: EValidNuisanceScore
    selectors: dict[EValidResAxis, int] = field(default_factory=dict)
