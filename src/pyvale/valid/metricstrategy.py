# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Metric strategies and compact tagged metric results."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from scipy import stats

from pyvale.valid.constants import MAVM_AREA_TOLERANCE
from pyvale.valid.cython.mavm import _calculate_mavm
from pyvale.valid.inputvalidation import validate_mavm_config
from pyvale.valid.metrics import (
    ERelErrNorm,
    _calculate_cvm_statistic,
    _calculate_ks_statistic,
    calc_avm_1d,
    calc_cvm_1d,
    calc_ks_1d,
    calc_mavm_1d,
    calculate_relative_error,
    validate_paired_samples,
    validate_relative_error_config,
)
from pyvale.valid.valtypes import EMAVMDuplicates, EMetricOutput, MAVMResult


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

    @abstractmethod
    def get_outputs(self) -> tuple[EMetricOutput, ...]:
        """Declare outputs in the order returned by extract_outputs."""

    @abstractmethod
    def extract_outputs(self, result: MetricResult) -> tuple[float, ...]:
        """Extract numerical outputs from this metric's tagged result."""


class MetricMAVM(IValMetric):
    __slots__ = ("alpha", "duplicates", "tol")
    alpha: float
    duplicates: EMAVMDuplicates
    tol: float

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        duplicates: EMAVMDuplicates = EMAVMDuplicates.INCLUDE,
        tol: float = MAVM_AREA_TOLERANCE,
    ) -> None:
        validate_mavm_config(alpha, duplicates, tol)
        self.alpha = alpha
        self.duplicates = duplicates
        self.tol = tol

    def get_kind(self) -> EMetricType:
        return EMetricType.MAVM

    def get_outputs(self) -> tuple[EMetricOutput, ...]:
        return (
            EMetricOutput.D_PLUS,
            EMetricOutput.D_MINUS,
        )

    def extract_outputs(self, result: MetricResult) -> tuple[float, ...]:
        if not isinstance(result, MAVMResult):
            raise TypeError("MAVM requires MAVMResult.")
        return result.d_plus, result.d_minus

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> MAVMResult:
        return calc_mavm_1d(
            sim, exp, alpha=self.alpha, duplicates=self.duplicates, tol=self.tol
        )


class MetricAVM(IValMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.AVM

    def get_outputs(self) -> tuple[EMetricOutput, ...]:
        return (EMetricOutput.DISTANCE,)

    def extract_outputs(self, result: MetricResult) -> tuple[float, ...]:
        if (
            not isinstance(result, ScalarMetricResult)
            or result.kind is not self.get_kind()
        ):
            raise TypeError("Result must match the metric kind.")
        return (result.value,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        return ScalarMetricResult(EMetricType.AVM, calc_avm_1d(sim, exp))


class MetricKS(IValMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.KS

    def get_outputs(self) -> tuple[EMetricOutput, ...]:
        return (EMetricOutput.STATISTIC,)

    def extract_outputs(self, result: MetricResult) -> tuple[float, ...]:
        if (
            not isinstance(result, ScalarMetricResult)
            or result.kind is not self.get_kind()
        ):
            raise TypeError("Result must match the metric kind.")
        return (result.value,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        return ScalarMetricResult(EMetricType.KS, calc_ks_1d(sim, exp))


class MetricCVM(IValMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.CVM

    def get_outputs(self) -> tuple[EMetricOutput, ...]:
        return (EMetricOutput.STATISTIC,)

    def extract_outputs(self, result: MetricResult) -> tuple[float, ...]:
        if (
            not isinstance(result, ScalarMetricResult)
            or result.kind is not self.get_kind()
        ):
            raise TypeError("Result must match the metric kind.")
        return (result.value,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        return ScalarMetricResult(EMetricType.CVM, calc_cvm_1d(sim, exp))


class MetricRMSE(IValMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.RMSE

    def get_outputs(self) -> tuple[EMetricOutput, ...]:
        return (EMetricOutput.ERROR,)

    def extract_outputs(self, result: MetricResult) -> tuple[float, ...]:
        if (
            not isinstance(result, ScalarMetricResult)
            or result.kind is not self.get_kind()
        ):
            raise TypeError("Result must match the metric kind.")
        return (result.value,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        sim_values, exp_values = validate_paired_samples(sim, exp)
        residual = sim_values - exp_values
        return ScalarMetricResult(
            EMetricType.RMSE,
            float(np.sqrt(np.mean(residual * residual))),
        )


class MetricRelErr(IValMetric):
    __slots__ = ("constant", "normalisation")
    normalisation: ERelErrNorm
    constant: float | None

    def __init__(
        self,
        *,
        normalisation: ERelErrNorm = ERelErrNorm.EXPERIMENT_MAX,
        constant: float | None = None,
    ) -> None:
        validate_relative_error_config(normalisation, constant)
        self.normalisation = normalisation
        self.constant = constant

    def get_kind(self) -> EMetricType:
        return EMetricType.RELATIVE_ERROR

    def get_outputs(self) -> tuple[EMetricOutput, ...]:
        return (EMetricOutput.ERROR,)

    def extract_outputs(self, result: MetricResult) -> tuple[float, ...]:
        if (
            not isinstance(result, ScalarMetricResult)
            or result.kind is not self.get_kind()
        ):
            raise TypeError("Result must match the metric kind.")
        return (result.value,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        simulation, experiment = validate_paired_samples(sim, exp)
        return ScalarMetricResult(
            EMetricType.RELATIVE_ERROR,
            calculate_relative_error(
                simulation, experiment, self.normalisation, self.constant
            ),
        )


def calculate_trusted_metric(
    metric: IValMetric, sim: np.ndarray, exp: np.ndarray
) -> MetricResult:
    """Calculate on validated engine arrays, omitting missing observations."""
    kind = metric.get_kind()
    simulation, experiment = sim, exp
    if type(metric) in (MetricRMSE, MetricRelErr):
        if not len(simulation):
            return ScalarMetricResult(kind, float("nan"))
        if isinstance(metric, MetricRelErr):
            value = calculate_relative_error(
                simulation, experiment, metric.normalisation, metric.constant
            )
        else:
            residual = simulation - experiment
            value = float(np.sqrt(np.mean(residual * residual)))
        return ScalarMetricResult(kind, value)
    if type(metric) is MetricMAVM:
        return _calculate_mavm(
            simulation, experiment, metric.alpha, metric.duplicates, metric.tol
        )
    if type(metric) is MetricAVM:
        return ScalarMetricResult(
            kind, float(stats.wasserstein_distance(simulation, experiment))
        )
    if type(metric) is MetricKS:
        return ScalarMetricResult(kind, _calculate_ks_statistic(simulation, experiment))
    if type(metric) is MetricCVM:
        return ScalarMetricResult(
            kind, _calculate_cvm_statistic(simulation, experiment)
        )
    return metric.calc(sim, exp)
