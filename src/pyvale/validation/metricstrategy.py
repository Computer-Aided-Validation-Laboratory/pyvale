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
from pyvale.validation.validtypes import EMAVMDupes


class MetricMAVM(IValidMetric):
    """Calculate directional area discrepancies with experimental confidence shifts.

    Attributes
    ----------
    alpha : float
        Experimental confidence parameter used for the Student's t shift.
    duplicates : EMAVMDupes
        Duplicate handling and empirical-probability integration policy.
    tol : float
        Absolute area tolerance for signed contributions, in observation units.
    """

    __slots__ = ("alpha", "duplicates", "tol")
    alpha: float
    duplicates: EMAVMDupes
    tol: float

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        duplicates: EMAVMDupes = EMAVMDupes.INCLUDE,
        tol: float = MAVM_AREA_TOLERANCE,
    ) -> None:
        """Configure MAVM confidence and duplicate handling.

        Parameters
        ----------
        alpha : float, optional
            Experimental confidence parameter used for the Student's t shift.
        duplicates : EMAVMDupes, optional
            Duplicate handling and empirical-probability integration policy.
        tol : float, optional
            Absolute area tolerance for signed contributions, in observation units.
        """
        _validate_mavm_config(alpha, duplicates, tol)
        self.alpha = alpha
        self.duplicates = duplicates
        self.tol = tol

    def get_kind(self) -> EMetricType:
        """Identify the validation metric.

        Returns
        -------
        EMetricType
            ``MAVM``.
        """
        return EMetricType.MAVM

    def get_scale(self) -> EMetricScale:
        """Declare the scale of the metric outputs.

        Returns
        -------
        EMetricScale
            ``PHYSICAL``.
        """
        return EMetricScale.PHYSICAL

    def get_comparison(self) -> EMetricCompare:
        """Declare how simulation and experimental samples are compared.

        Returns
        -------
        EMetricCompare
            ``DISTRIB``.
        """
        return EMetricCompare.DISTRIB

    def get_outputs(self) -> tuple[EMetricOut, ...]:
        """Declare the output tags in numerical result order.

        Returns
        -------
        tuple[EMetricOut, ...]
            ``D_PLUS``, ``D_MINUS``.
        """
        return (
            EMetricOut.D_PLUS,
            EMetricOut.D_MINUS,
        )

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> MAVMResult:
        """Calculate the metric for a single pair of conditional samples.

        Parameters
        ----------
        sim : ndarray
            Simulation observations of shape (n_sim_aleatory,). NaNs are omitted.
        exp : ndarray
            Experimental observations of shape (n_exp_aleatory,). NaNs are omitted.

        Returns
        -------
        MAVMResult
            Directional areas in observable units and their supporting ECDF data.
        """
        return calc_mavm_1d(
            sim, exp, alpha=self.alpha, duplicates=self.duplicates, tol=self.tol
        )

    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> MAVMResult:
        """Calculate the metric on prepared conditional samples.

        Parameters
        ----------
        sim : ndarray
            Simulation observations of shape (n_sim_aleatory,). NaNs have been removed.
        exp : ndarray
            Experimental observations of shape (n_exp_aleatory,). NaNs have been
            removed.

        Returns
        -------
        MAVMResult
            Directional areas in observable units and their supporting ECDF data.
        """
        return _calculate_mavm(sim, exp, self.alpha, self.duplicates, self.tol)


class MetricAVM(IValidMetric):
    """Calculate the first Wasserstein distance between empirical distributions."""

    __slots__ = ()

    def get_kind(self) -> EMetricType:
        """Identify the validation metric.

        Returns
        -------
        EMetricType
            ``AVM``.
        """
        return EMetricType.AVM

    def get_scale(self) -> EMetricScale:
        """Declare the scale of the metric outputs.

        Returns
        -------
        EMetricScale
            ``PHYSICAL``.
        """
        return EMetricScale.PHYSICAL

    def get_comparison(self) -> EMetricCompare:
        """Declare how simulation and experimental samples are compared.

        Returns
        -------
        EMetricCompare
            ``DISTRIB``.
        """
        return EMetricCompare.DISTRIB

    def get_outputs(self) -> tuple[EMetricOut, ...]:
        """Declare the output tags in numerical result order.

        Returns
        -------
        tuple[EMetricOut, ...]
            ``DISTANCE``.
        """
        return (EMetricOut.DISTANCE,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        """Calculate the metric for a single pair of conditional samples.

        Parameters
        ----------
        sim : ndarray
            Simulation observations of shape (n_sim_aleatory,). NaNs are omitted.
        exp : ndarray
            Experimental observations of shape (n_exp_aleatory,). NaNs are omitted.

        Returns
        -------
        ScalarMetricResult
            First Wasserstein distance in the original observation units.
        """
        return self._calc_trusted(
            _validate_finite_sample(sim, "model_data"),
            _validate_finite_sample(exp, "exp_data"),
        )

    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        """Calculate the metric on prepared conditional samples.

        Parameters
        ----------
        sim : ndarray
            Simulation observations of shape (n_sim_aleatory,). NaNs have been removed.
        exp : ndarray
            Experimental observations of shape (n_exp_aleatory,). NaNs have been
            removed.

        Returns
        -------
        ScalarMetricResult
            First Wasserstein distance in the original observation units.
        """
        distance = float(stats.wasserstein_distance(sim, exp))
        return ScalarMetricResult(EMetricType.AVM, distance)


class MetricKS(IValidMetric):
    """Calculate the maximum probability difference between two empirical CDFs."""

    __slots__ = ()

    def get_kind(self) -> EMetricType:
        """Identify the validation metric.

        Returns
        -------
        EMetricType
            ``KS``.
        """
        return EMetricType.KS

    def get_scale(self) -> EMetricScale:
        """Declare the scale of the metric outputs.

        Returns
        -------
        EMetricScale
            ``DIMLESS_01``.
        """
        return EMetricScale.DIMLESS_01

    def get_comparison(self) -> EMetricCompare:
        """Declare how simulation and experimental samples are compared.

        Returns
        -------
        EMetricCompare
            ``DISTRIB``.
        """
        return EMetricCompare.DISTRIB

    def get_outputs(self) -> tuple[EMetricOut, ...]:
        """Declare the output tags in numerical result order.

        Returns
        -------
        tuple[EMetricOut, ...]
            ``STATISTIC``.
        """
        return (EMetricOut.STATISTIC,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        """Calculate the metric for a single pair of conditional samples.

        Parameters
        ----------
        sim : ndarray
            Simulation observations of shape (n_sim_aleatory,). NaNs are omitted.
        exp : ndarray
            Experimental observations of shape (n_exp_aleatory,). NaNs are omitted.

        Returns
        -------
        ScalarMetricResult
            Maximum absolute CDF difference in [0, 1].
        """
        return self._calc_trusted(
            _validate_finite_sample(sim, "model_data"),
            _validate_finite_sample(exp, "exp_data"),
        )

    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        """Calculate the metric on prepared conditional samples.

        Parameters
        ----------
        sim : ndarray
            Simulation observations of shape (n_sim_aleatory,). NaNs have been removed.
        exp : ndarray
            Experimental observations of shape (n_exp_aleatory,). NaNs have been
            removed.

        Returns
        -------
        ScalarMetricResult
            Maximum absolute CDF difference in [0, 1].
        """
        return ScalarMetricResult(EMetricType.KS, _calculate_ks_statistic(sim, exp))


class MetricCVM(IValidMetric):
    """Calculate the two-sample Cramér–von Mises statistic."""

    __slots__ = ()

    def get_kind(self) -> EMetricType:
        """Identify the validation metric.

        Returns
        -------
        EMetricType
            ``CVM``.
        """
        return EMetricType.CVM

    def get_scale(self) -> EMetricScale:
        """Declare the scale of the metric outputs.

        Returns
        -------
        EMetricScale
            ``DIMLESS``.
        """
        return EMetricScale.DIMLESS

    def get_comparison(self) -> EMetricCompare:
        """Declare how simulation and experimental samples are compared.

        Returns
        -------
        EMetricCompare
            ``DISTRIB``.
        """
        return EMetricCompare.DISTRIB

    def get_outputs(self) -> tuple[EMetricOut, ...]:
        """Declare the output tags in numerical result order.

        Returns
        -------
        tuple[EMetricOut, ...]
            ``STATISTIC``.
        """
        return (EMetricOut.STATISTIC,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        """Calculate the metric for a single pair of conditional samples.

        Parameters
        ----------
        sim : ndarray
            Simulation observations of shape (n_sim_aleatory,). NaNs are omitted.
        exp : ndarray
            Experimental observations of shape (n_exp_aleatory,). NaNs are omitted.

        Returns
        -------
        ScalarMetricResult
            Dimensionless two-sample Cramér–von Mises statistic.
        """
        return self._calc_trusted(
            _validate_finite_sample(sim, "model_data"),
            _validate_finite_sample(exp, "exp_data"),
        )

    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        """Calculate the metric on prepared conditional samples.

        Parameters
        ----------
        sim : ndarray
            Simulation observations of shape (n_sim_aleatory,). NaNs have been removed.
        exp : ndarray
            Experimental observations of shape (n_exp_aleatory,). NaNs have been
            removed.

        Returns
        -------
        ScalarMetricResult
            Dimensionless two-sample Cramér–von Mises statistic.
        """
        return ScalarMetricResult(EMetricType.CVM, _calculate_cvm_statistic(sim, exp))


class MetricRMSE(IValidMetric):
    """Calculate root mean square error for paired observations."""

    __slots__ = ()

    def get_kind(self) -> EMetricType:
        """Identify the validation metric.

        Returns
        -------
        EMetricType
            ``RMSE``.
        """
        return EMetricType.RMSE

    def get_scale(self) -> EMetricScale:
        """Declare the scale of the metric outputs.

        Returns
        -------
        EMetricScale
            ``PHYSICAL``.
        """
        return EMetricScale.PHYSICAL

    def get_comparison(self) -> EMetricCompare:
        """Declare how simulation and experimental samples are compared.

        Returns
        -------
        EMetricCompare
            ``PAIRED``.
        """
        return EMetricCompare.PAIRED

    def get_outputs(self) -> tuple[EMetricOut, ...]:
        """Declare the output tags in numerical result order.

        Returns
        -------
        tuple[EMetricOut, ...]
            ``ERROR``.
        """
        return (EMetricOut.ERROR,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        """Calculate the metric for a single pair of conditional samples.

        Parameters
        ----------
        sim : ndarray
            Simulation observations of shape (n_pairs,). Both arrays follow the same
            pair order. Pairs containing NaN are omitted.
        exp : ndarray
            Experimental observations of shape (n_pairs,). Both arrays follow the same
            pair order. Pairs containing NaN are omitted.

        Returns
        -------
        ScalarMetricResult
            Root mean square error in the original units.
        """
        return self._calc_trusted(*_validate_paired_samples(sim, exp))

    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        """Calculate the metric on prepared conditional samples.

        Parameters
        ----------
        sim : ndarray
            Simulation observations of shape (n_pairs,). Both arrays follow the same
            pair order. Both arrays have been jointly filtered and may be empty.
        exp : ndarray
            Experimental observations of shape (n_pairs,). Both arrays follow the same
            pair order. Both arrays have been jointly filtered and may be empty.

        Returns
        -------
        ScalarMetricResult
            Root mean square error in the original units.
        """
        if not len(sim):
            return ScalarMetricResult(EMetricType.RMSE, float("nan"))

        residual = sim - exp
        return ScalarMetricResult(
            EMetricType.RMSE,
            float(np.sqrt(np.mean(residual * residual))),
        )


class MetricRelErr(IValidMetric):
    """Calculate a normalised mean absolute error for paired observations.

    Attributes
    ----------
    normalisation : ERelErrNorm
        Reference used to scale the mean absolute paired error.
    constant : float or None
        Positive fixed normalisation value for CONSTANT mode; otherwise None.
    """

    __slots__ = ("constant", "normalisation")
    normalisation: ERelErrNorm
    constant: float | None

    def __init__(
        self,
        *,
        normalisation: ERelErrNorm = ERelErrNorm.EXP_MAX,
        constant: float | None = None,
    ) -> None:
        """Configure the reference magnitude for relative error.

        Parameters
        ----------
        normalisation : ERelErrNorm, optional
            Reference used to scale the mean absolute paired error.
        constant : float or None, optional
            Positive fixed normalisation value for CONSTANT mode; otherwise None.
        """
        _validate_relative_error_config(normalisation, constant)
        self.normalisation = normalisation
        self.constant = constant

    def get_kind(self) -> EMetricType:
        """Identify the validation metric.

        Returns
        -------
        EMetricType
            ``REL_ERR``.
        """
        return EMetricType.REL_ERR

    def get_scale(self) -> EMetricScale:
        """Declare the scale of the metric outputs.

        Returns
        -------
        EMetricScale
            ``DIMLESS``.
        """
        return EMetricScale.DIMLESS

    def get_comparison(self) -> EMetricCompare:
        """Declare how simulation and experimental samples are compared.

        Returns
        -------
        EMetricCompare
            ``PAIRED``.
        """
        return EMetricCompare.PAIRED

    def get_outputs(self) -> tuple[EMetricOut, ...]:
        """Declare the output tags in numerical result order.

        Returns
        -------
        tuple[EMetricOut, ...]
            ``ERROR``.
        """
        return (EMetricOut.ERROR,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        """Calculate the metric for a single pair of conditional samples.

        Parameters
        ----------
        sim : ndarray
            Simulation observations of shape (n_pairs,). Both arrays follow the same
            pair order. Pairs containing NaN are omitted.
        exp : ndarray
            Experimental observations of shape (n_pairs,). Both arrays follow the same
            pair order. Pairs containing NaN are omitted.

        Returns
        -------
        ScalarMetricResult
            Dimensionless relative error using the configured normalisation.
        """
        return self._calc_trusted(*_validate_paired_samples(sim, exp))

    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> ScalarMetricResult:
        """Calculate the metric on prepared conditional samples.

        Parameters
        ----------
        sim : ndarray
            Simulation observations of shape (n_pairs,). Both arrays follow the same
            pair order. Both arrays have been jointly filtered and may be empty.
        exp : ndarray
            Experimental observations of shape (n_pairs,). Both arrays follow the same
            pair order. Both arrays have been jointly filtered and may be empty.

        Returns
        -------
        ScalarMetricResult
            Dimensionless relative error using the configured normalisation.
        """
        if not len(sim):
            return ScalarMetricResult(EMetricType.REL_ERR, float("nan"))

        return ScalarMetricResult(
            EMetricType.REL_ERR,
            _calculate_relative_error(sim, exp, self.normalisation, self.constant),
        )
