# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Spatial sensitivity kernels for weighting field points over spatial windows.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np


class ISpatialKernel(ABC):
    """Abstract interface for continuous spatial sensitivity kernels."""

    @abstractmethod
    def eval_weights(self, local_coords: np.ndarray) -> np.ndarray:
        """Evaluates continuous sensitivity weights at local spatial
        coordinates.

        Parameters
        ----------
        local_coords : np.ndarray
            Local coordinates within the sensor spatial window support,
            shape=(n_pts, dims).

        Returns
        -------
        np.ndarray
            Continuous sensitivity weights, shape=(n_pts,).
        """


class SpatialKernelUniform(ISpatialKernel):
    """Uniform spatial sensitivity kernel with constant weighting w(x) = 1.0.
    """

    __slots__ = ()

    def eval_weights(self, local_coords: np.ndarray) -> np.ndarray:
        n_pts = local_coords.shape[0]
        return np.ones(n_pts, dtype=float)


class SpatialKernelGaussian(ISpatialKernel):
    """Continuous Gaussian sensitivity kernel:
    w(xi) = exp(-0.5 * sum((xi_i / sigma_i)^2)).
    """

    __slots__ = ("_sigma",)

    def __init__(self, sigma: float | tuple[float, ...]) -> None:
        """
        Parameters
        ----------
        sigma : float | tuple[float, ...]
            Standard deviation of the Gaussian profile along each dimension.
        """
        self._sigma = sigma

    def get_sigma(self) -> float | tuple[float, ...]:
        return self._sigma

    def eval_weights(self, local_coords: np.ndarray) -> np.ndarray:
        if isinstance(self._sigma, (int, float)):
            sigmas = np.array([float(self._sigma)], dtype=float)
        else:
            sigmas = np.array(self._sigma, dtype=float)

        k_dims = min(len(sigmas), local_coords.shape[1])
        coords = local_coords[:, :k_dims]
        scaled = coords / sigmas[:k_dims]
        r_sq = np.sum(scaled**2, axis=1)
        return np.exp(-0.5 * r_sq)


class SpatialKernelTriangular(ISpatialKernel):
    """Triangular (conical/tent) sensitivity kernel decaying linearly to zero
    at specified radii: w(xi) = max(0, 1 - sqrt(sum((xi_i / R_i)^2))).
    """

    __slots__ = ("_radii",)

    def __init__(self, radii: float | tuple[float, ...]) -> None:
        """
        Parameters
        ----------
        radii : float | tuple[float, ...]
            Boundary radius along each dimension where weight reaches 0.
        """
        self._radii = radii

    def get_radii(self) -> float | tuple[float, ...]:
        return self._radii

    def eval_weights(self, local_coords: np.ndarray) -> np.ndarray:
        if isinstance(self._radii, (int, float)):
            rads = np.array([float(self._radii)], dtype=float)
        else:
            rads = np.array(self._radii, dtype=float)

        k_dims = min(len(rads), local_coords.shape[1])
        coords = local_coords[:, :k_dims]
        scaled = coords / rads[:k_dims]
        norm_r = np.sqrt(np.sum(scaled**2, axis=1))
        weights = 1.0 - norm_r
        weights[weights < 0.0] = 0.0
        return weights


class SpatialKernelCustom(ISpatialKernel):
    """Custom user-defined spatial sensitivity kernel."""

    __slots__ = ("_func",)

    def __init__(
        self, func: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        """
        Parameters
        ----------
        func : Callable[[np.ndarray], np.ndarray]
            User function accepting local coordinates (n_pts, dims) and
            returning 1D array of weights (n_pts,).
        """
        self._func = func

    def eval_weights(self, local_coords: np.ndarray) -> np.ndarray:
        return np.asarray(self._func(local_coords), dtype=float)
