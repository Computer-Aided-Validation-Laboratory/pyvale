# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Riley-ordered two-dimensional element shape functions."""

import numpy as np

from ..mesh import EElementType


def shape_functions(
    element_type: EElementType,
    xi: float,
    eta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Riley-ordered shape functions and natural derivatives.

    Parameters
    ----------
    element_type : EElementType
        Surface topology of the evaluated element.
    xi, eta : float
        Natural coordinates. Triangles use ``xi >= 0``, ``eta >= 0``, and
        ``xi + eta <= 1``; quadrilaterals use ``[-1, 1]`` coordinates.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Shape values and derivatives with respect to ``xi`` and ``eta``.
    """
    if element_type is EElementType.TRI3:
        values = np.array((1.0 - xi - eta, xi, eta))
        d_xi = np.array((-1.0, 1.0, 0.0))
        d_eta = np.array((-1.0, 0.0, 1.0))
        return values, d_xi, d_eta
    if element_type is EElementType.TRI6:
        one = 1.0 - xi - eta
        values = np.array(
            (
                one * (2.0 * one - 1.0),
                xi * (2.0 * xi - 1.0),
                eta * (2.0 * eta - 1.0),
                4.0 * one * xi,
                4.0 * xi * eta,
                4.0 * eta * one,
            )
        )
        d_xi = np.array(
            (
                -(4.0 * one - 1.0),
                4.0 * xi - 1.0,
                0.0,
                4.0 * (one - xi),
                4.0 * eta,
                -4.0 * eta,
            )
        )
        d_eta = np.array(
            (
                -(4.0 * one - 1.0),
                0.0,
                4.0 * eta - 1.0,
                -4.0 * xi,
                4.0 * xi,
                4.0 * (one - eta),
            )
        )
        return values, d_xi, d_eta
    if element_type is EElementType.QUAD4:
        values = 0.25 * np.array(
            (
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            )
        )
        d_xi = 0.25 * np.array(
            (
                -(1.0 - eta),
                1.0 - eta,
                1.0 + eta,
                -(1.0 + eta),
            )
        )
        d_eta = 0.25 * np.array(
            (
                -(1.0 - xi),
                -(1.0 + xi),
                1.0 + xi,
                1.0 - xi,
            )
        )
        return values, d_xi, d_eta
    if element_type is EElementType.QUAD8:
        values = np.array(
            (
                -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta),
                -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta),
                -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta),
                -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta),
                0.5 * (1 - xi * xi) * (1 - eta),
                0.5 * (1 + xi) * (1 - eta * eta),
                0.5 * (1 - xi * xi) * (1 + eta),
                0.5 * (1 - xi) * (1 - eta * eta),
            )
        )
        d_xi = np.array(
            (
                0.25 * (1 - eta) * (2 * xi + eta),
                0.25 * (1 - eta) * (2 * xi - eta),
                0.25 * (1 + eta) * (2 * xi + eta),
                0.25 * (1 + eta) * (2 * xi - eta),
                -xi * (1 - eta),
                0.5 * (1 - eta * eta),
                -xi * (1 + eta),
                -0.5 * (1 - eta * eta),
            )
        )
        d_eta = np.array(
            (
                0.25 * (1 - xi) * (xi + 2 * eta),
                0.25 * (1 + xi) * (2 * eta - xi),
                0.25 * (1 + xi) * (xi + 2 * eta),
                0.25 * (1 - xi) * (2 * eta - xi),
                -0.5 * (1 - xi * xi),
                -eta * (1 + xi),
                0.5 * (1 - xi * xi),
                -eta * (1 - xi),
            )
        )
        return values, d_xi, d_eta
    if element_type is EElementType.QUAD9:
        phi = np.array(
            (0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0))
        )
        psi = np.array(
            (0.5 * eta * (eta - 1.0), 1.0 - eta * eta, 0.5 * eta * (eta + 1.0))
        )
        d_phi = np.array((xi - 0.5, -2.0 * xi, xi + 0.5))
        d_psi = np.array((eta - 0.5, -2.0 * eta, eta + 0.5))
        pairs = (
            (0, 0),
            (2, 0),
            (2, 2),
            (0, 2),
            (1, 0),
            (2, 1),
            (1, 2),
            (0, 1),
            (1, 1),
        )
        values = np.array([phi[ix] * psi[iy] for ix, iy in pairs])
        d_xi = np.array([d_phi[ix] * psi[iy] for ix, iy in pairs])
        d_eta = np.array([phi[ix] * d_psi[iy] for ix, iy in pairs])
        return values, d_xi, d_eta
    raise ValueError(f"Unsupported PixInt2D element type {element_type!r}.")


def in_natural_domain(
    element_type: EElementType,
    xi: float,
    eta: float,
    tolerance: float = 1.0e-9,
) -> bool:
    """Return whether natural coordinates lie in an element domain."""
    if element_type in (EElementType.TRI3, EElementType.TRI6):
        return (
            xi >= -tolerance
            and eta >= -tolerance
            and xi + eta <= 1.0 + tolerance
        )
    return abs(xi) <= 1.0 + tolerance and abs(eta) <= 1.0 + tolerance


__all__ = ["in_natural_domain", "shape_functions"]
