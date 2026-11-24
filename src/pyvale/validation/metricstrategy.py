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
    ScalarMetricResult,
)
from pyvale.validation.metrics import (
    ERelErrNorm,
    _calculate_cvm_statistic,
    _calculate_ks_statistic,
    calc_mavm_1d,
    _calculate_relative_error,
    _validate_paired_samples,
    _validate_relative_error_config,
)
from pyvale.validation.validchecks import _validate_finite_sample, _validate_mavm_config
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

    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> MAVMResult:
        return _calculate_mavm(sim, exp, self.alpha, self.duplicates, self.tol)


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
        return self._calc_trusted(
            _validate_finite_sample(sim, "model_data"),
            _validate_finite_sample(exp, "exp_data"),
        )

    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        distance = float(stats.wasserstein_distance(sim, exp))
        return ScalarMetricResult(EMetricType.AVM, distance)


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
        return self._calc_trusted(
            _validate_finite_sample(sim, "model_data"),
            _validate_finite_sample(exp, "exp_data"),
        )

    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        return ScalarMetricResult(EMetricType.KS, _calculate_ks_statistic(sim, exp))


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
        return self._calc_trusted(
            _validate_finite_sample(sim, "model_data"),
            _validate_finite_sample(exp, "exp_data"),
        )

    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        return ScalarMetricResult(EMetricType.CVM, _calculate_cvm_statistic(sim, exp))


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
        return self._calc_trusted(*_validate_paired_samples(sim, exp))

    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        if not len(sim):
            return ScalarMetricResult(EMetricType.RMSE, float("nan"))
        residual = sim - exp
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
        return self._calc_trusted(*_validate_paired_samples(sim, exp))

    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        if not len(sim):
            return ScalarMetricResult(EMetricType.REL_ERR, float("nan"))
        return ScalarMetricResult(
            EMetricType.REL_ERR,
            _calculate_relative_error(sim, exp, self.normalisation, self.constant),
        )
