"""Metric strategies and compact tagged metric results."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from pyvale.valid.constants import MAVM_AREA_TOLERANCE
from pyvale.valid.metrics import (
    EMAVMDuplicates,
    MAVMResult,
    calc_avm_1d,
    calc_cvm_1d,
    calc_ks_1d,
    calc_mavm_1d,
)


class EMetricType(Enum):
    MAVM = auto()
    AVM = auto()
    KS = auto()
    CVM = auto()
    RMSE = auto()
    RELATIVE_ERROR = auto()


@dataclass(slots=True)
class ScalarMetricResult:
    kind: EMetricType
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EMetricType):
            raise TypeError("Metric kind must be an EMetricType.")
        if self.kind is EMetricType.MAVM:
            raise ValueError("MAVM requires directional result fields.")


MetricResult = ScalarMetricResult | MAVMResult


class IValMetric(ABC):
    """Calculate a metric from one-dimensional NumPy observations."""

    @abstractmethod
    def calc(self, sim: np.ndarray, exp: np.ndarray) -> MetricResult:
        """Return a tagged metric result."""

    @abstractmethod
    def get_kind(self) -> EMetricType:
        """Return the metric tag without calculating a result."""


class MetricMAVM(IValMetric):
    __slots__ = ("alpha", "duplicates", "tol")

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        duplicates: EMAVMDuplicates = EMAVMDuplicates.INCLUDE,
        tol: float = MAVM_AREA_TOLERANCE,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("MAVM alpha must be between zero and one.")
        self.alpha = alpha
        self.duplicates = duplicates
        self.tol = tol

    def get_kind(self) -> EMetricType:
        return EMetricType.MAVM

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> MAVMResult:
        return calc_mavm_1d(sim, exp, self.alpha, self.duplicates, self.tol)


class MetricAVM(IValMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.AVM

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        return ScalarMetricResult(EMetricType.AVM, calc_avm_1d(sim, exp))


class MetricKS(IValMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.KS

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        return ScalarMetricResult(EMetricType.KS, calc_ks_1d(sim, exp))


class MetricCVM(IValMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.CVM

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        return ScalarMetricResult(EMetricType.CVM, calc_cvm_1d(sim, exp))


class MetricRMSE(IValMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.RMSE

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        sim_values = np.asarray(sim, dtype=np.float64)
        exp_values = np.asarray(exp, dtype=np.float64)
        if sim_values.ndim != 1 or sim_values.shape != exp_values.shape:
            raise ValueError("RMSE requires equal-length paired 1D observations.")
        if np.any(np.isinf(sim_values)) or np.any(np.isinf(exp_values)):
            raise ValueError("RMSE observations cannot contain infinity.")
        valid = np.isfinite(sim_values) & np.isfinite(exp_values)
        if not np.any(valid):
            raise ValueError("RMSE requires at least one valid pair.")
        residual = sim_values[valid] - exp_values[valid]
        return ScalarMetricResult(
            EMetricType.RMSE,
            float(np.sqrt(np.mean(residual * residual))),
        )


class MetricRelativeError(IValMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.RELATIVE_ERROR

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        sim_values = np.asarray(sim, dtype=np.float64)
        exp_values = np.asarray(exp, dtype=np.float64)
        if sim_values.ndim != 1 or sim_values.shape != exp_values.shape:
            raise ValueError("Relative error requires paired 1D observations.")
        if np.any(np.isinf(sim_values)) or np.any(np.isinf(exp_values)):
            raise ValueError("Relative error observations cannot contain infinity.")
        valid = np.isfinite(sim_values) & np.isfinite(exp_values)
        if not np.any(valid):
            raise ValueError("Relative error requires at least one valid pair.")
        denominator = np.maximum(np.abs(exp_values[valid]), 1.0e-12)
        value = np.mean(np.abs(sim_values[valid] - exp_values[valid]) / denominator)
        return ScalarMetricResult(EMetricType.RELATIVE_ERROR, float(value))
