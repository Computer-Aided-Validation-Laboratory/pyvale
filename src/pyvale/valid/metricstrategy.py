# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Metric strategies and compact tagged metric results."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import cast

import numpy as np
from scipy import stats

from pyvale.valid.constants import MAVM_AREA_TOLERANCE
from pyvale.valid.cython.mavm import _calculate_mavm
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
from pyvale.valid.validtypes import (
    EMAVMDuplicates,
    EMetricOutput,
    MAVMResult,
    validate_mavm_config,
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
    def get_outputs(self) -> tuple[EMetricOutput, ...]:
        """Declare outputs in the order returned by extract_outputs."""

    @abstractmethod
    def extract_outputs(self, result: MetricResult) -> tuple[float, ...]:
        """Extract numerical outputs from this metric's tagged result."""


class MetricMAVM(IValidMetric):
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
        mavm_result = cast(MAVMResult, result)
        return mavm_result.d_plus, mavm_result.d_minus

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> MAVMResult:
        return calc_mavm_1d(
            sim, exp, alpha=self.alpha, duplicates=self.duplicates, tol=self.tol
        )


class MetricAVM(IValidMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.AVM

    def get_outputs(self) -> tuple[EMetricOutput, ...]:
        return (EMetricOutput.DISTANCE,)

    def extract_outputs(self, result: MetricResult) -> tuple[float, ...]:
        return (cast(ScalarMetricResult, result).value,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        return ScalarMetricResult(EMetricType.AVM, calc_avm_1d(sim, exp))


class MetricKS(IValidMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.KS

    def get_outputs(self) -> tuple[EMetricOutput, ...]:
        return (EMetricOutput.STATISTIC,)

    def extract_outputs(self, result: MetricResult) -> tuple[float, ...]:
        return (cast(ScalarMetricResult, result).value,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        return ScalarMetricResult(EMetricType.KS, calc_ks_1d(sim, exp))


class MetricCVM(IValidMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.CVM

    def get_outputs(self) -> tuple[EMetricOutput, ...]:
        return (EMetricOutput.STATISTIC,)

    def extract_outputs(self, result: MetricResult) -> tuple[float, ...]:
        return (cast(ScalarMetricResult, result).value,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        return ScalarMetricResult(EMetricType.CVM, calc_cvm_1d(sim, exp))


class MetricRMSE(IValidMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.RMSE

    def get_outputs(self) -> tuple[EMetricOutput, ...]:
        return (EMetricOutput.ERROR,)

    def extract_outputs(self, result: MetricResult) -> tuple[float, ...]:
        return (cast(ScalarMetricResult, result).value,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        sim_values, exp_values = validate_paired_samples(sim, exp)
        residual = sim_values - exp_values
        return ScalarMetricResult(
            EMetricType.RMSE,
            float(np.sqrt(np.mean(residual * residual))),
        )


class MetricRelErr(IValidMetric):
    __slots__ = ("constant", "normalisation")
    normalisation: ERelErrNorm
    constant: float | None

    def __init__(
        self,
        *,
        normalisation: ERelErrNorm = ERelErrNorm.EXP_MAX,
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
        return (cast(ScalarMetricResult, result).value,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        sim, exp = validate_paired_samples(sim, exp)
        return ScalarMetricResult(
            EMetricType.RELATIVE_ERROR,
            calculate_relative_error(sim, exp, self.normalisation, self.constant),
        )


def calculate_trusted_metric(
    metric: IValidMetric, sim: np.ndarray, exp: np.ndarray
) -> MetricResult:
    """Calculate on validated engine arrays, omitting missing observations."""
    kind = metric.get_kind()
    if type(metric) in (MetricRMSE, MetricRelErr):
        if not len(sim):
            return ScalarMetricResult(kind, float("nan"))
        if isinstance(metric, MetricRelErr):
            value = calculate_relative_error(
                sim, exp, metric.normalisation, metric.constant
            )
        else:
            residual = sim - exp
            value = float(np.sqrt(np.mean(residual * residual)))
        return ScalarMetricResult(kind, value)
    if type(metric) is MetricMAVM:
        return _calculate_mavm(sim, exp, metric.alpha, metric.duplicates, metric.tol)
    if type(metric) is MetricAVM:
        return ScalarMetricResult(kind, float(stats.wasserstein_distance(sim, exp)))
    if type(metric) is MetricKS:
        return ScalarMetricResult(kind, _calculate_ks_statistic(sim, exp))
    if type(metric) is MetricCVM:
        return ScalarMetricResult(kind, _calculate_cvm_statistic(sim, exp))
    return metric.calc(sim, exp)
