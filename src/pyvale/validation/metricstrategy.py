# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Metric strategies and compact tagged metric results."""

import numpy as np
from scipy import stats

from pyvale.validation.constants import MAVM_AREA_TOLERANCE
from pyvale.validation.cython.mavm import _calculate_mavm
from pyvale.validation.imetric import (
    EMetricCompare,
    EMetricOut,
    EMetricScale,
    EMetricType,
    IValidMetric,
    MAVMResult,
    MetricResult,
    ScalarMetricResult,
)
from pyvale.validation.metrics import (
    ERelErrNorm,
    _calculate_cvm_statistic,
    _calculate_ks_statistic,
    calc_avm_1d,
    calc_cvm_1d,
    calc_ks_1d,
    calc_mavm_1d,
    _calculate_relative_error,
    _validate_paired_samples,
    _validate_relative_error_config,
)
from pyvale.validation.validchecks import _validate_mavm_config
from pyvale.validation.validtypes import EMAVMDuplicates


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
        _validate_mavm_config(alpha, duplicates, tol)
        self.alpha = alpha
        self.duplicates = duplicates
        self.tol = tol

    def get_kind(self) -> EMetricType:
        return EMetricType.MAVM

    def get_scale(self) -> EMetricScale:
        return EMetricScale.PHYSICAL

    def get_comparison(self) -> EMetricCompare:
        return EMetricCompare.DISTRIB

    def get_outputs(self) -> tuple[EMetricOut, ...]:
        return (
            EMetricOut.D_PLUS,
            EMetricOut.D_MINUS,
        )

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> MAVMResult:
        return calc_mavm_1d(
            sim, exp, alpha=self.alpha, duplicates=self.duplicates, tol=self.tol
        )


class MetricAVM(IValidMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.AVM

    def get_scale(self) -> EMetricScale:
        return EMetricScale.PHYSICAL

    def get_comparison(self) -> EMetricCompare:
        return EMetricCompare.DISTRIB

    def get_outputs(self) -> tuple[EMetricOut, ...]:
        return (EMetricOut.DISTANCE,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        return ScalarMetricResult(EMetricType.AVM, calc_avm_1d(sim, exp))


class MetricKS(IValidMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.KS

    def get_scale(self) -> EMetricScale:
        return EMetricScale.DIMLESS_01

    def get_comparison(self) -> EMetricCompare:
        return EMetricCompare.DISTRIB

    def get_outputs(self) -> tuple[EMetricOut, ...]:
        return (EMetricOut.STATISTIC,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        return ScalarMetricResult(EMetricType.KS, calc_ks_1d(sim, exp))


class MetricCVM(IValidMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.CVM

    def get_scale(self) -> EMetricScale:
        return EMetricScale.DIMLESS

    def get_comparison(self) -> EMetricCompare:
        return EMetricCompare.DISTRIB

    def get_outputs(self) -> tuple[EMetricOut, ...]:
        return (EMetricOut.STATISTIC,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        return ScalarMetricResult(EMetricType.CVM, calc_cvm_1d(sim, exp))


class MetricRMSE(IValidMetric):
    __slots__ = ()

    def get_kind(self) -> EMetricType:
        return EMetricType.RMSE

    def get_scale(self) -> EMetricScale:
        return EMetricScale.PHYSICAL

    def get_comparison(self) -> EMetricCompare:
        return EMetricCompare.PAIRED

    def get_outputs(self) -> tuple[EMetricOut, ...]:
        return (EMetricOut.ERROR,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        sim_values, exp_values = _validate_paired_samples(sim, exp)
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
        _validate_relative_error_config(normalisation, constant)
        self.normalisation = normalisation
        self.constant = constant

    def get_kind(self) -> EMetricType:
        return EMetricType.REL_ERR

    def get_scale(self) -> EMetricScale:
        return EMetricScale.DIMLESS

    def get_comparison(self) -> EMetricCompare:
        return EMetricCompare.PAIRED

    def get_outputs(self) -> tuple[EMetricOut, ...]:
        return (EMetricOut.ERROR,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        sim, exp = _validate_paired_samples(sim, exp)
        return ScalarMetricResult(
            EMetricType.REL_ERR,
            _calculate_relative_error(sim, exp, self.normalisation, self.constant),
        )


def _calculate_trusted_metric(
    metric: IValidMetric, sim: np.ndarray, exp: np.ndarray
) -> MetricResult:
    """Calculate on validated engine arrays, omitting missing observations."""
    kind = metric.get_kind()
    if type(metric) in (MetricRMSE, MetricRelErr):
        if not len(sim):
            return ScalarMetricResult(kind, float("nan"))
        if isinstance(metric, MetricRelErr):
            value = _calculate_relative_error(
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
