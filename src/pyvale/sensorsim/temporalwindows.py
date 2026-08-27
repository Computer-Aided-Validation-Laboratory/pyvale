# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Temporal integration windows and response kernels for time-domain filtering.
"""

from abc import ABC, abstractmethod
import numpy as np

from pyvale.sensorsim.enums import EIntegrationMode
from pyvale.sensorsim.integrationrules import (
    IIntegrationRule,
    IntegrationGaussLegendre,
)


class ITemporalKernel(ABC):
    """Abstract interface for continuous temporal weighting kernels."""

    @abstractmethod
    def eval_weights(self, tau: np.ndarray, duration: float) -> np.ndarray:
        """Evaluates continuous temporal sensitivity weights.

        Parameters
        ----------
        tau : np.ndarray
            Time offsets relative to target measurement time t0, shape=(n_pts,).
        duration : float
            Total temporal window duration.

        Returns
        -------
        np.ndarray
            Weighting factors, shape=(n_pts,).
        """


class TemporalKernelUniform(ITemporalKernel):
    """Uniform temporal weighting kernel."""

    __slots__ = ()

    def eval_weights(self, tau: np.ndarray, duration: float) -> np.ndarray:
        return np.ones_like(tau, dtype=float)


class TemporalKernelExponentialDecay(ITemporalKernel):
    """First-order exponential response kernel:
    w(tau) = exp(tau / tau_const) for tau <= 0 (causal memory decay).
    """

    __slots__ = ("_time_constant",)

    def __init__(self, time_constant: float) -> None:
        """
        Parameters
        ----------
        time_constant : float
            System time constant (e.g. RC time or thermocouple thermal lag).
        """
        self._time_constant = float(time_constant)

    def get_time_constant(self) -> float:
        return self._time_constant

    def eval_weights(self, tau: np.ndarray, duration: float) -> np.ndarray:
        return np.exp(tau / self._time_constant)


class TemporalKernelGaussian(ITemporalKernel):
    """Gaussian temporal sensitivity kernel:
    w(tau) = exp(-0.5 * (tau / sigma)^2).
    """

    __slots__ = ("_sigma",)

    def __init__(self, sigma: float) -> None:
        """
        Parameters
        ----------
        sigma : float
            Standard deviation of temporal Gaussian filter.
        """
        self._sigma = float(sigma)

    def get_sigma(self) -> float:
        return self._sigma

    def eval_weights(self, tau: np.ndarray, duration: float) -> np.ndarray:
        return np.exp(-0.5 * (tau / self._sigma) ** 2)


class TemporalKernelHann(ITemporalKernel):
    """Hanning temporal window kernel."""

    __slots__ = ()

    def eval_weights(self, tau: np.ndarray, duration: float) -> np.ndarray:
        if duration <= 0.0:
            return np.ones_like(tau, dtype=float)
        # Map tau to [-pi, pi]
        scaled = np.clip(tau / (0.5 * duration), -1.0, 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * scaled))


class ITemporalWindow(ABC):
    """Abstract interface for temporal integration windows."""

    @abstractmethod
    def get_duration(self) -> float:
        """Total time duration of the temporal window."""

    @abstractmethod
    def get_effective_duration(self) -> float:
        """Integrated effective temporal duration (integral of kernel over
        window).
        """

    @abstractmethod
    def get_sample_offsets_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates time offsets relative to nominal measurement time t0
        and corresponding weights.

        Parameters
        ----------
        mode : EIntegrationMode, optional
            Integration mode (AVERAGE or ACCUMULATE), by default AVERAGE.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of:
            - offsets: shape (n_quad_pts,), time offsets tau
            - weights: shape (n_quad_pts,), integration weights
        """


class TemporalWindowInstant(ITemporalWindow):
    """Instantaneous sampling window (duration = 0, delta-function in time)."""

    __slots__ = ()

    def get_duration(self) -> float:
        return 0.0

    def get_effective_duration(self) -> float:
        return 1.0

    def get_sample_offsets_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([0.0], dtype=float), np.array([1.0], dtype=float))


class TemporalWindowCentered(ITemporalWindow):
    """Symmetric temporal integration window centered at target time t0:
    [t0 - duration/2, t0 + duration/2].
    """

    __slots__ = ("_duration", "_integ_rule", "_kernel")

    def __init__(
        self,
        duration: float,
        integ_rule: IIntegrationRule | None = None,
        kernel: ITemporalKernel | None = None,
    ) -> None:
        """
        Parameters
        ----------
        duration : float
            Exposure duration.
        integ_rule : IIntegrationRule | None, optional
            Numerical integration rule, by default IntegrationGaussLegendre(3).
        kernel : ITemporalKernel | None, optional
            Temporal weighting kernel, by default TemporalKernelUniform().
        """
        self._duration = float(duration)
        if integ_rule is None:
            integ_rule = IntegrationGaussLegendre(3)
        self._integ_rule = integ_rule

        if kernel is None:
            kernel = TemporalKernelUniform()
        self._kernel = kernel

    def get_duration(self) -> float:
        return self._duration

    def get_integ_rule(self) -> IIntegrationRule:
        return self._integ_rule

    def get_kernel(self) -> ITemporalKernel:
        return self._kernel

    def get_effective_duration(self) -> float:
        _, weights = self.get_sample_offsets_and_weights(
            mode=EIntegrationMode.ACCUMULATE
        )
        return float(np.sum(weights))

    def get_sample_offsets_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._duration <= 0.0:
            return (np.array([0.0], dtype=float), np.array([1.0], dtype=float))

        canonical_nodes, canonical_weights = (
            self._integ_rule.get_nodes_and_weights(dims=1)
        )
        nodes_1d = canonical_nodes.ravel()

        # Map [-1, 1] to [-duration/2, duration/2]
        jacobian = 0.5 * self._duration
        offsets = nodes_1d * jacobian
        raw_weights = canonical_weights * jacobian

        kernel_weights = self._kernel.eval_weights(offsets, self._duration)
        composite_weights = raw_weights * kernel_weights

        if mode == EIntegrationMode.AVERAGE:
            total_sum = np.sum(composite_weights)
            if total_sum > 0.0:
                normalized_weights = composite_weights / total_sum
            else:
                normalized_weights = composite_weights
            return (offsets, normalized_weights)

        return (offsets, composite_weights)


class TemporalWindowCausal(ITemporalWindow):
    """Causal backward-integrating temporal window: [t0 - duration, t0].
    Used for modeling thermal inertia, sensor response lag, or RC filters.
    """

    __slots__ = ("_duration", "_integ_rule", "_kernel")

    def __init__(
        self,
        duration: float,
        integ_rule: IIntegrationRule | None = None,
        kernel: ITemporalKernel | None = None,
    ) -> None:
        """
        Parameters
        ----------
        duration : float
            Backward integration history length.
        integ_rule : IIntegrationRule | None, optional
            Numerical integration rule, by default IntegrationGaussLegendre(4).
        kernel : ITemporalKernel | None, optional
            Temporal weighting kernel, by default TemporalKernelUniform().
        """
        self._duration = float(duration)
        if integ_rule is None:
            integ_rule = IntegrationGaussLegendre(4)
        self._integ_rule = integ_rule

        if kernel is None:
            kernel = TemporalKernelUniform()
        self._kernel = kernel

    def get_duration(self) -> float:
        return self._duration

    def get_integ_rule(self) -> IIntegrationRule:
        return self._integ_rule

    def get_kernel(self) -> ITemporalKernel:
        return self._kernel

    def get_effective_duration(self) -> float:
        _, weights = self.get_sample_offsets_and_weights(
            mode=EIntegrationMode.ACCUMULATE
        )
        return float(np.sum(weights))

    def get_sample_offsets_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._duration <= 0.0:
            return (np.array([0.0], dtype=float), np.array([1.0], dtype=float))

        canonical_nodes, canonical_weights = (
            self._integ_rule.get_nodes_and_weights(dims=1)
        )
        nodes_1d = canonical_nodes.ravel()

        # Map [-1, 1] to [-duration, 0.0]
        jacobian = 0.5 * self._duration
        offsets = (nodes_1d - 1.0) * jacobian
        raw_weights = canonical_weights * jacobian

        kernel_weights = self._kernel.eval_weights(offsets, self._duration)
        composite_weights = raw_weights * kernel_weights

        if mode == EIntegrationMode.AVERAGE:
            total_sum = np.sum(composite_weights)
            if total_sum > 0.0:
                normalized_weights = composite_weights / total_sum
            else:
                normalized_weights = composite_weights
            return (offsets, normalized_weights)

        return (offsets, composite_weights)
