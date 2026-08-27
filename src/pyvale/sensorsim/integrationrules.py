# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Numerical integration rules for spatial and temporal window integration.
"""

from abc import ABC, abstractmethod
import numpy as np


class IIntegrationRule(ABC):
    """Abstract base interface for N-dimensional numerical integration rules.
    Provides quadrature nodes and weights on canonical reference domains
    [-1, 1]^d.
    """

    @abstractmethod
    def get_nodes_and_weights(
        self, dims: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates integration nodes and weights on canonical domain
        [-1, 1]^d.

        Parameters
        ----------
        dims : int
            Number of spatial/temporal dimensions (1, 2, or 3).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of:
            - nodes: shape (n_quad_pts, dims), coordinates in [-1, 1]
            - weights: shape (n_quad_pts,), metric weights summing to 2^dims
        """


class IntegrationGaussLegendre(IIntegrationRule):
    """Gauss-Legendre quadrature rule for exact polynomial integration.
    An n-point rule exactly integrates polynomials up to degree 2n - 1.
    """

    __slots__ = ("_order",)

    def __init__(self, order: int | tuple[int, ...] = 2) -> None:
        """
        Parameters
        ----------
        order : int | tuple[int, ...], optional
            Number of quadrature points per dimension, by default 2. If a tuple
            is provided, each entry specifies the order along that axis.
        """
        self._order = order

    def get_order(self) -> int | tuple[int, ...]:
        return self._order

    def get_nodes_and_weights(
        self, dims: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self._order, int):
            orders = (self._order,) * dims
        else:
            orders = self._order
            if len(orders) != dims:
                msg = (
                    f"Order tuple length ({len(orders)}) must match "
                    f"dimensions ({dims})."
                )
                raise ValueError(msg)

        nodes_1d = []
        weights_1d = []
        for n_pts in orders:
            pts, wts = np.polynomial.legendre.leggauss(n_pts)
            nodes_1d.append(pts)
            weights_1d.append(wts)

        if dims == 1:
            nodes = nodes_1d[0].reshape(-1, 1)
            weights = weights_1d[0]
            return (nodes, weights)

        # Tensor product grid for multi-dimensional quadrature
        grids = np.meshgrid(*nodes_1d, indexing="ij")
        nodes = np.column_stack([g.ravel() for g in grids])

        w_grids = np.meshgrid(*weights_1d, indexing="ij")
        weights = np.prod([w.ravel() for w in w_grids], axis=0)

        return (nodes, weights)


class IntegrationMidpoint(IIntegrationRule):
    """Uniform piecewise midpoint integration rule on [-1, 1]^d."""

    __slots__ = ("_divisions",)

    def __init__(self, divisions: int | tuple[int, ...] = 4) -> None:
        """
        Parameters
        ----------
        divisions : int | tuple[int, ...], optional
            Number of uniform divisions per dimension, by default 4.
        """
        self._divisions = divisions

    def get_divisions(self) -> int | tuple[int, ...]:
        return self._divisions

    def get_nodes_and_weights(
        self, dims: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self._divisions, int):
            divs = (self._divisions,) * dims
        else:
            divs = self._divisions
            if len(divs) != dims:
                msg = (
                    f"Divisions tuple length ({len(divs)}) must match "
                    f"dimensions ({dims})."
                )
                raise ValueError(msg)

        nodes_1d = []
        weights_1d = []
        for n_div in divs:
            edges = np.linspace(-1.0, 1.0, n_div + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            w = np.full(n_div, 2.0 / n_div)
            nodes_1d.append(centers)
            weights_1d.append(w)

        if dims == 1:
            return (nodes_1d[0].reshape(-1, 1), weights_1d[0])

        grids = np.meshgrid(*nodes_1d, indexing="ij")
        nodes = np.column_stack([g.ravel() for g in grids])

        w_grids = np.meshgrid(*weights_1d, indexing="ij")
        weights = np.prod([w.ravel() for w in w_grids], axis=0)

        return (nodes, weights)


class IntegrationSimpson(IIntegrationRule):
    """Composite Simpson's 1/3 rule on [-1, 1]^d. Requires an even number of
    sub-intervals (divisions), yielding an odd number of sample points.
    """

    __slots__ = ("_divisions",)

    def __init__(self, divisions: int | tuple[int, ...] = 4) -> None:
        """
        Parameters
        ----------
        divisions : int | tuple[int, ...], optional
            Number of subdivisions (must be even), by default 4.
        """
        self._divisions = divisions

    def get_divisions(self) -> int | tuple[int, ...]:
        return self._divisions

    def get_nodes_and_weights(
        self, dims: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self._divisions, int):
            divs = (self._divisions,) * dims
        else:
            divs = self._divisions
            if len(divs) != dims:
                msg = (
                    f"Divisions tuple length ({len(divs)}) must match "
                    f"dimensions ({dims})."
                )
                raise ValueError(msg)

        nodes_1d = []
        weights_1d = []
        for n_div in divs:
            if n_div % 2 != 0:
                n_div = n_div + 1
            pts = np.linspace(-1.0, 1.0, n_div + 1)
            h = 2.0 / n_div
            w = np.ones(n_div + 1)
            w[1:-1:2] = 4.0
            w[2:-1:2] = 2.0
            w = w * (h / 3.0)
            nodes_1d.append(pts)
            weights_1d.append(w)

        if dims == 1:
            return (nodes_1d[0].reshape(-1, 1), weights_1d[0])

        grids = np.meshgrid(*nodes_1d, indexing="ij")
        nodes = np.column_stack([g.ravel() for g in grids])

        w_grids = np.meshgrid(*weights_1d, indexing="ij")
        weights = np.prod([w.ravel() for w in w_grids], axis=0)

        return (nodes, weights)


class IntegrationTrapezoidal(IIntegrationRule):
    """Composite Trapezoidal rule on [-1, 1]^d."""

    __slots__ = ("_divisions",)

    def __init__(self, divisions: int | tuple[int, ...] = 4) -> None:
        """
        Parameters
        ----------
        divisions : int | tuple[int, ...], optional
            Number of subdivisions, by default 4.
        """
        self._divisions = divisions

    def get_divisions(self) -> int | tuple[int, ...]:
        return self._divisions

    def get_nodes_and_weights(
        self, dims: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self._divisions, int):
            divs = (self._divisions,) * dims
        else:
            divs = self._divisions
            if len(divs) != dims:
                msg = (
                    f"Divisions tuple length ({len(divs)}) must match "
                    f"dimensions ({dims})."
                )
                raise ValueError(msg)

        nodes_1d = []
        weights_1d = []
        for n_div in divs:
            pts = np.linspace(-1.0, 1.0, n_div + 1)
            h = 2.0 / n_div
            w = np.full(n_div + 1, h)
            w[0] = 0.5 * h
            w[-1] = 0.5 * h
            nodes_1d.append(pts)
            weights_1d.append(w)

        if dims == 1:
            return (nodes_1d[0].reshape(-1, 1), weights_1d[0])

        grids = np.meshgrid(*nodes_1d, indexing="ij")
        nodes = np.column_stack([g.ravel() for g in grids])

        w_grids = np.meshgrid(*weights_1d, indexing="ij")
        weights = np.prod([w.ravel() for w in w_grids], axis=0)

        return (nodes, weights)


class IntegrationMonteCarlo(IIntegrationRule):
    """Monte Carlo quasi-random uniform sampling on [-1, 1]^d."""

    __slots__ = ("_num_samples", "_seed")

    def __init__(
        self, num_samples: int = 100, seed: int | None = None
    ) -> None:
        """
        Parameters
        ----------
        num_samples : int, optional
            Number of random evaluation points, by default 100.
        seed : int | None, optional
            Random seed for reproducibility, by default None.
        """
        self._num_samples = num_samples
        self._seed = seed

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_nodes_and_weights(
        self, dims: int
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self._seed)
        nodes = rng.uniform(-1.0, 1.0, size=(self._num_samples, dims))
        total_vol = 2.0**dims
        weights = np.full(self._num_samples, total_vol / self._num_samples)
        return (nodes, weights)
