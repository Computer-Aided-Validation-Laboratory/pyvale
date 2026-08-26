# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Strategy pattern interfaces for validation metrics."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from pyvale.valid.metrics import (
    MAVMResult,
    calc_mavm_1d,
    calc_avm_1d,
    calc_ks_1d,
    calc_cvm_1d,
    calc_deterministic_metrics_1d,
)


class IValMetric(ABC):
    """Abstract base class for all validation metric calculators."""

    @abstractmethod
    def calc(
        self,
        model_data: np.ndarray,
        exp_data: np.ndarray,
    ) -> Any:
        """Calculate the validation metric between model and experimental data.

        Parameters
        ----------
        model_data : np.ndarray
            Model realizations array.
        exp_data : np.ndarray
            Experimental observations array.

        Returns
        -------
        Any
            Metric result (e.g. float or MAVMResult dataclass).
        """


class MetricMAVM(IValMetric):
    """Modified Area Validation Metric (MAVM) strategy."""

    __slots__ = ("_alpha", "_tol")

    def __init__(self, alpha: float = 0.05, tol: float = 1e-12) -> None:
        self._alpha = alpha
        self._tol = tol

    def calc(
        self,
        model_data: np.ndarray,
        exp_data: np.ndarray,
    ) -> MAVMResult:
        return calc_mavm_1d(
            model_data=model_data,
            exp_data=exp_data,
            alpha=self._alpha,
            tol=self._tol,
        )


class MetricAVM(IValMetric):
    """Classical Area Validation Metric (1-Wasserstein) strategy."""

    def calc(
        self,
        model_data: np.ndarray,
        exp_data: np.ndarray,
    ) -> float:
        return calc_avm_1d(model_data, exp_data)


class MetricKS(IValMetric):
    """Kolmogorov-Smirnov distance strategy (L-infinity norm on CDFs)."""

    def calc(
        self,
        model_data: np.ndarray,
        exp_data: np.ndarray,
    ) -> float:
        return calc_ks_1d(model_data, exp_data)


class MetricCVM(IValMetric):
    """Cramér-von Mises distance strategy (L2 norm on CDFs)."""

    def calc(
        self,
        model_data: np.ndarray,
        exp_data: np.ndarray,
    ) -> float:
        return calc_cvm_1d(model_data, exp_data)


class MetricRMSE(IValMetric):
    """Root Mean Square Error strategy."""

    def calc(
        self,
        model_data: np.ndarray,
        exp_data: np.ndarray,
    ) -> float:
        res = calc_deterministic_metrics_1d(model_data, exp_data)
        return res["rmse"]


class MetricRelativeError(IValMetric):
    """Relative error strategy."""

    def calc(
        self,
        model_data: np.ndarray,
        exp_data: np.ndarray,
    ) -> float:
        res = calc_deterministic_metrics_1d(model_data, exp_data)
        return res["relative_error"]
