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
    DIMENSIONLESS = auto()
    DIMENSIONLESS_01 = auto()


class EMetricComparison(Enum):
    PAIRED = auto()
    DISTRIBUTIONAL = auto()
    CALIBRATION = auto()


class EMetricType(Enum):
    MAVM = auto()
    AVM = auto()
    KS = auto()
    CVM = auto()
    RMSE = auto()
    RELATIVE_ERROR = auto()


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

    model_quantiles: np.ndarray
    """Quantiles (support values) of the model empirical CDF."""

    model_probs: np.ndarray
    """Cumulative probabilities corresponding to model quantiles."""

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


MetricResult = ScalarMetricResult | MAVMResult


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
    def get_comparison(self) -> EMetricComparison:
        """Declare how the metric compares sim and exp observations."""

    @abstractmethod
    def get_outputs(self) -> tuple[EMetricOut, ...]:
        """Declare outputs in the order returned by extract_outputs."""

    @abstractmethod
    def extract_outputs(self, result: MetricResult) -> tuple[float, ...]:
        """Extract numerical outputs from this metric's tagged result."""
