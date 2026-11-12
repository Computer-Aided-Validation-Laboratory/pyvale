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
    ERelativeErrorNormalization,
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
from pyvale.valid.valtypes import EMAVMDuplicates, MAVMResult


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

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> MAVMResult:
        return calc_mavm_1d(
            sim, exp, alpha=self.alpha, duplicates=self.duplicates, tol=self.tol
        )


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
        sim_values, exp_values = validate_paired_samples(sim, exp)
        residual = sim_values - exp_values
        return ScalarMetricResult(
            EMetricType.RMSE,
            float(np.sqrt(np.mean(residual * residual))),
        )


class MetricRelErr(IValMetric):
    __slots__ = ("constant", "normalization")
    normalization: ERelativeErrorNormalization
    constant: float | None

    def __init__(
        self,
        *,
        normalization: ERelativeErrorNormalization = ERelativeErrorNormalization.EXPERIMENT_MAX,
        constant: float | None = None,
    ) -> None:
        validate_relative_error_config(normalization, constant)
        self.normalization = normalization
        self.constant = constant

    def get_kind(self) -> EMetricType:
        return EMetricType.RELATIVE_ERROR

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        simulation, experiment = validate_paired_samples(sim, exp)
        return ScalarMetricResult(
            EMetricType.RELATIVE_ERROR,
            calculate_relative_error(
                simulation, experiment, self.normalization, self.constant
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
                simulation, experiment, metric.normalization, metric.constant
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
        return ScalarMetricResult(
            kind, _calculate_ks_statistic(simulation, experiment)
        )
    if type(metric) is MetricCVM:
        return ScalarMetricResult(
            kind, _calculate_cvm_statistic(simulation, experiment)
        )
    return metric.calc(sim, exp)
