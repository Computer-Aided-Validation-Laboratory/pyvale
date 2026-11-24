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
    PHYSICAL = auto()
    DIMLESS = auto()
    DIMLESS_01 = auto()


class EMetricCompare(Enum):
    PAIRED = auto()
    DISTRIB = auto()
    CALIB = auto()


class EMetricType(Enum):
    MAVM = auto()
    AVM = auto()
    KS = auto()
    CVM = auto()
    RMSE = auto()
    REL_ERR = auto()


class EMetricOut(Enum):
    D_PLUS = auto()
    D_MINUS = auto()
    DISTANCE = auto()
    STATISTIC = auto()
    ERROR = auto()


@dataclass(slots=True)
class MAVMResult:
    """Container for Modified Area Validation Metric (MAVM) calculation
    results.
    """

    d_plus: float
    """Positive support-shift mismatch area, matching fullfieldvalmetrics."""

    d_minus: float
    """Negative support-shift mismatch area, matching fullfieldvalmetrics."""

    sim_quantiles: np.ndarray
    """Quantiles (support values) of the sim empirical CDF."""

    sim_probs: np.ndarray
    """Cumulative probabilities corresponding to sim quantiles."""

    exp_quantiles: np.ndarray
    """Quantiles (support values) of the experimental empirical CDF."""

    exp_probs: np.ndarray
    """Cumulative probabilities corresponding to experimental quantiles."""

    exp_conf_lower: np.ndarray
    """Lower confidence bound quantiles for the experimental CDF."""

    exp_conf_upper: np.ndarray
    """Upper confidence bound quantiles for the experimental CDF."""

    alpha: float
    """Significance level used for the confidence interval (e.g. 0.05)."""


@dataclass(slots=True)
class ScalarMetricResult:
    kind: EMetricType
    value: float


@dataclass(slots=True)
class MultiMetricResult:
    kind: EMetricType
    values: tuple[float, ...]


MetricResult = ScalarMetricResult | MultiMetricResult | MAVMResult


def extract_metric_outputs(result: MetricResult) -> tuple[float, ...]:
    """Return numerical outputs in the metric's declared order."""
    if isinstance(result, ScalarMetricResult):
        return (result.value,)
    if isinstance(result, MultiMetricResult):
        return result.values
    return result.d_plus, result.d_minus


class IValidMetric(ABC):
    """Calculate a metric from one-dimensional NumPy observations."""

    @abstractmethod
    def calc(self, sim: np.ndarray, exp: np.ndarray) -> MetricResult:
        """Return a tagged metric result."""

    @abstractmethod
    def get_kind(self) -> EMetricType:
        """Return the metric tag without calculating a result."""

    @abstractmethod
    def get_scale(self) -> EMetricScale:
        """Declare the numerical scale of the metric outputs."""

    @abstractmethod
    def get_comparison(self) -> EMetricCompare:
        """Declare how the metric compares sim and exp observations."""

    @abstractmethod
    def get_outputs(self) -> tuple[EMetricOut, ...]:
        """Declare outputs in the order returned by extract_metric_outputs."""
