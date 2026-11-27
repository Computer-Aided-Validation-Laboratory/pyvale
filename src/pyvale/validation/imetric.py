# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Interface and shared types for validation metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class EMetricScale(Enum):
    """Classify metric outputs by physical units or dimensionless range."""

    PHYSICAL = auto()
    DIMLESS = auto()
    DIMLESS_01 = auto()


class EMetricCompare(Enum):
    """Identify how simulation and experimental samples are compared."""

    PAIRED = auto()
    DISTRIB = auto()
    CALIB = auto()


class EMetricType(Enum):
    """Identify the metric algorithm used for a result field."""

    MAVM = auto()
    AVM = auto()
    KS = auto()
    CVM = auto()
    RMSE = auto()
    REL_ERR = auto()


class EMetricOut(Enum):
    """Identify an individual numerical output of a metric."""

    D_PLUS = auto()
    D_MINUS = auto()
    DISTANCE = auto()
    STATISTIC = auto()
    ERROR = auto()


@dataclass(slots=True)
class MAVMResult:
    """Store Modified Area Validation Metric results for a conditional pair.

    Attributes
    ----------
    d_plus : float
        Positive support-shift mismatch area in observable units.
    d_minus : float
        Negative support-shift mismatch area magnitude in observable units.
    sim_quantiles : ndarray
        Distinct simulation support points of shape (n_sim_support,).
    sim_probs : ndarray
        Cumulative masses aligned with sim_quantiles, of shape (n_sim_support,).
    exp_quantiles : ndarray
        Distinct experimental support points of shape (n_exp_support,).
    exp_probs : ndarray
        Cumulative masses aligned with exp_quantiles, of shape (n_exp_support,).
    exp_conf_lower : ndarray
        Experimental lower confidence quantiles of shape (n_exp_support,).
    exp_conf_upper : ndarray
        Experimental upper confidence quantiles of shape (n_exp_support,).
    alpha : float
        Significance level used to compute the experimental shift.
    """

    d_plus: float
    d_minus: float

    sim_quantiles: np.ndarray
    sim_probs: np.ndarray
    exp_quantiles: np.ndarray
    exp_probs: np.ndarray

    exp_conf_lower: np.ndarray
    exp_conf_upper: np.ndarray
    alpha: float


@dataclass(slots=True)
class ScalarMetricResult:
    """Store the single value returned by a scalar metric.

    Attributes
    ----------
    kind : EMetricType
        Metric identifier.
    value : float
        Metric value in the scale declared by its strategy.
    """

    kind: EMetricType
    value: float


@dataclass(slots=True)
class MultiMetricResult:
    """Store ordered numerical outputs returned by a multi-output metric.

    Attributes
    ----------
    kind : EMetricType
        Metric identifier.
    values : tuple[float, ...]
        Numerical outputs in the order declared by get_outputs.
    """

    kind: EMetricType
    values: tuple[float, ...]


MetricResult = ScalarMetricResult | MultiMetricResult | MAVMResult


def extract_metric_outputs(result: MetricResult) -> tuple[float, ...]:
    """Return numerical outputs in the metric's declared order.

    Parameters
    ----------
    result : MetricResult
        Tagged scalar, multi-output, or MAVM result.

    Returns
    -------
    tuple[float, ...]
        Numerical values in the order declared by the metric strategy.
    """

    if isinstance(result, ScalarMetricResult):
        return (result.value,)

    if isinstance(result, MultiMetricResult):
        return result.values

    return result.d_plus, result.d_minus


class IValidMetric(ABC):
    """Calculate a metric from one-dimensional NumPy observations."""

    @abstractmethod
    def calc(self, sim: np.ndarray, exp: np.ndarray) -> MetricResult:
        """Return a tagged metric result.

        Parameters
        ----------
        sim : ndarray
            Simulation observations of shape (n_sim_aleatory,) for one comparison.
        exp : ndarray
            Experimental observations of shape (n_exp_aleatory,) in the same units as
            sim.

        Returns
        -------
        MetricResult
            Tagged metric result for one conditional comparison.
        """

    @abstractmethod
    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> MetricResult:
        """Calculate on validated 1D samples after engine NaN filtering.

        Paired metrics may receive empty arrays when no complete pairs remain.

        Parameters
        ----------
        sim : ndarray
            Validated simulation samples of shape (n_sim_aleatory,), with NaNs removed.
        exp : ndarray
            Validated experimental samples of shape (n_exp_aleatory,), with NaNs
            removed. Paired metrics receive the same joint mask on both arrays.

        Returns
        -------
        MetricResult
            Tagged metric result for the supplied comparison.
        """

    @abstractmethod
    def get_kind(self) -> EMetricType:
        """Return the metric tag without calculating a result.

        Returns
        -------
        EMetricType
            Metric identifier used to key result fields.
        """

    @abstractmethod
    def get_scale(self) -> EMetricScale:
        """Declare the numerical scale of the metric outputs.

        Returns
        -------
        EMetricScale
            Physical or dimensionless scale used to interpret numerical outputs.
        """

    @abstractmethod
    def get_comparison(self) -> EMetricCompare:
        """Declare how the metric compares sim and exp observations.

        Returns
        -------
        EMetricCompare
            Comparison mode declaring how observations are paired or treated as
            distributions.
        """

    @abstractmethod
    def get_outputs(self) -> tuple[EMetricOut, ...]:
        """Declare outputs in the order returned by extract_metric_outputs.

        Returns
        -------
        tuple[EMetricOut, ...]
            Output tags in the order returned by extract_metric_outputs.
        """
