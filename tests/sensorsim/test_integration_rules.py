# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Unit tests for mesh-free numerical integration rules, spatial kernels,
and temporal integration windows.
"""

import numpy as np
import pytest

from pyvale.sensorsim.enums import EIntegrationMode
from pyvale.sensorsim.integrationrules import (
    IntegrationGaussLegendre,
    IntegrationMidpoint,
    IntegrationSimpson,
    IntegrationTrapezoidal,
    IntegrationMonteCarlo,
)
from pyvale.sensorsim.spatialkernels import (
    SpatialKernelUniform,
    SpatialKernelGaussian,
    SpatialKernelTriangular,
    SpatialKernelCustom,
)
from pyvale.sensorsim.temporalwindows import (
    TemporalKernelUniform,
    TemporalKernelExponentialDecay,
    TemporalKernelGaussian,
    TemporalKernelHann,
    TemporalWindowInstant,
    TemporalWindowCentered,
    TemporalWindowCausal,
)


def test_gauss_legendre_polynomial_exactness_1d() -> None:
    """Gauss-Legendre n-point rule is exact for polynomials up to
    degree 2n-1.
    """
    # Degree 1: f(x) = 3x + 2, exact integral on [-1, 1] is 4.0
    gl1 = IntegrationGaussLegendre(order=1)
    pts1, wts1 = gl1.get_nodes_and_weights(dims=1)
    f1 = 3.0 * pts1[:, 0] + 2.0
    assert np.isclose(np.sum(f1 * wts1), 4.0)

    # Degree 3: f(x) = 4x^3 - 2x^2 + 5x + 1
    # Integral = 4*(0) - 2*(2/3) + 5*(0) + 1*(2) = -4/3 + 2 = 2/3 ≈ 0.6666667
    gl2 = IntegrationGaussLegendre(order=2)
    pts2, wts2 = gl2.get_nodes_and_weights(dims=1)
    f2 = (
        4.0 * pts2[:, 0] ** 3
        - 2.0 * pts2[:, 0] ** 2
        + 5.0 * pts2[:, 0]
        + 1.0
    )
    assert np.isclose(np.sum(f2 * wts2), 2.0 / 3.0)

    # Degree 5: f(x) = x^5 - 3x^4 + 2x^2 - 1
    # Integral = 0 - 3*(2/5) + 2*(2/3) - 2 = -6/5 + 4/3 - 2 = -28/15
    gl3 = IntegrationGaussLegendre(order=3)
    pts3, wts3 = gl3.get_nodes_and_weights(dims=1)
    f3 = (
        pts3[:, 0] ** 5
        - 3.0 * pts3[:, 0] ** 4
        + 2.0 * pts3[:, 0] ** 2
        - 1.0
    )
    assert np.isclose(np.sum(f3 * wts3), -28.0 / 15.0)


def test_simpson_polynomial_exactness_1d() -> None:
    """Simpson's rule is exact for polynomials up to degree 3."""
    # f(x) = 2x^3 - x^2 + 4x - 3 on [-1, 1]
    # Integral = 0 - (2/3) + 0 - 6 = -20/3 ≈ -6.666667
    simp = IntegrationSimpson(divisions=4)
    pts, wts = simp.get_nodes_and_weights(dims=1)
    f = 2.0 * pts[:, 0] ** 3 - pts[:, 0] ** 2 + 4.0 * pts[:, 0] - 3.0
    assert np.isclose(np.sum(f * wts), -20.0 / 3.0)


def test_midpoint_and_trapezoidal_linear_exactness_1d() -> None:
    """Midpoint and Trapezoidal rules are exact for linear polynomials."""
    # f(x) = 5x + 3 on [-1, 1], integral is 6.0
    mid = IntegrationMidpoint(divisions=4)
    pts_m, wts_m = mid.get_nodes_and_weights(dims=1)
    assert np.isclose(np.sum((5.0 * pts_m[:, 0] + 3.0) * wts_m), 6.0)

    trap = IntegrationTrapezoidal(divisions=4)
    pts_t, wts_t = trap.get_nodes_and_weights(dims=1)
    assert np.isclose(np.sum((5.0 * pts_t[:, 0] + 3.0) * wts_t), 6.0)


def test_monte_carlo_integration_1d() -> None:
    """Monte Carlo rule converges to integral on [-1, 1]."""
    mc = IntegrationMonteCarlo(num_samples=50000, seed=42)
    pts, wts = mc.get_nodes_and_weights(dims=1)
    f = pts[:, 0] ** 2  # Integral of x^2 on [-1, 1] is 2/3
    res = np.sum(f * wts)
    assert np.isclose(res, 2.0 / 3.0, atol=0.02)


def test_gauss_legendre_2d_and_3d() -> None:
    """Multi-dimensional Gauss-Legendre integration on [-1, 1]^d."""
    # 2D: f(x, y) = (x^2 + 1)*(y^2 + 2) on [-1, 1]^2
    # Integral = (2/3 + 2) * (2/3 + 4) = (8/3) * (14/3) = 112/9 ≈ 12.444444
    gl2d = IntegrationGaussLegendre(order=2)
    pts2d, wts2d = gl2d.get_nodes_and_weights(dims=2)
    f2d = (pts2d[:, 0] ** 2 + 1.0) * (pts2d[:, 1] ** 2 + 2.0)
    assert np.isclose(np.sum(f2d * wts2d), 112.0 / 9.0)

    # 3D: f(x, y, z) = (x^2 + 1)*(y + 2)*(z^2 + 3) on [-1, 1]^3
    # Integral = (8/3) * (4) * (2/3 + 6) = (32/3) * (20/3) = 640/9
    gl3d = IntegrationGaussLegendre(order=2)
    pts3d, wts3d = gl3d.get_nodes_and_weights(dims=3)
    f3d = (
        (pts3d[:, 0] ** 2 + 1.0)
        * (pts3d[:, 1] + 2.0)
        * (pts3d[:, 2] ** 2 + 3.0)
    )
    assert np.isclose(np.sum(f3d * wts3d), 640.0 / 9.0)


def test_spatial_kernels() -> None:
    """Test spatial sensitivity kernel evaluations."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    # Uniform
    k_uni = SpatialKernelUniform()
    w_uni = k_uni.eval_weights(coords)
    assert np.allclose(w_uni, 1.0)

    # Gaussian with sigma=(1.0, 1.0)
    k_gauss = SpatialKernelGaussian(sigma=(1.0, 1.0))
    w_gauss = k_gauss.eval_weights(coords)
    assert np.isclose(w_gauss[0], 1.0)
    assert np.isclose(w_gauss[1], np.exp(-0.5))
    assert np.isclose(w_gauss[3], np.exp(-1.0))

    # Triangular with radii=(2.0, 2.0)
    k_tri = SpatialKernelTriangular(radii=(2.0, 2.0))
    w_tri = k_tri.eval_weights(coords)
    assert np.isclose(w_tri[0], 1.0)
    assert np.isclose(w_tri[1], 0.5)
    assert np.isclose(w_tri[3], 1.0 - np.sqrt(2.0) / 2.0)

    # Custom
    k_cust = SpatialKernelCustom(lambda c: c[:, 0] + c[:, 1])
    w_cust = k_cust.eval_weights(coords)
    assert np.allclose(w_cust, [0.0, 1.0, 1.0, 2.0])


def test_temporal_windows() -> None:
    """Test temporal window offsets, weights, and average/accumulate modes."""
    # Instant
    tw_inst = TemporalWindowInstant()
    off_i, w_i = tw_inst.get_sample_offsets_and_weights()
    assert np.allclose(off_i, [0.0])
    assert np.allclose(w_i, [1.0])
    assert tw_inst.get_duration() == 0.0

    # Centered window: duration 2.0, interval [-1.0, 1.0]
    tw_cent = TemporalWindowCentered(
        duration=2.0,
        integ_rule=IntegrationGaussLegendre(3),
    )
    assert np.isclose(tw_cent.get_duration(), 2.0)

    off_c, w_avg = tw_cent.get_sample_offsets_and_weights(
        mode=EIntegrationMode.AVERAGE
    )
    assert np.isclose(np.sum(w_avg), 1.0)
    assert np.all(off_c >= -1.0) and np.all(off_c <= 1.0)

    _, w_acc = tw_cent.get_sample_offsets_and_weights(
        mode=EIntegrationMode.ACCUMULATE
    )
    assert np.isclose(np.sum(w_acc), 2.0)

    # Causal window: duration 2.0, interval [-2.0, 0.0]
    tw_caus = TemporalWindowCausal(
        duration=2.0,
        integ_rule=IntegrationGaussLegendre(4),
        kernel=TemporalKernelExponentialDecay(time_constant=1.0),
    )
    off_k, w_caus_avg = tw_caus.get_sample_offsets_and_weights(
        mode=EIntegrationMode.AVERAGE
    )
    assert np.isclose(np.sum(w_caus_avg), 1.0)
    assert np.all(off_k >= -2.0) and np.all(off_k <= 0.0)

    # Integral of exp(t) from -2 to 0 is 1 - exp(-2) ≈ 0.8646647
    eff_dur = tw_caus.get_effective_duration()
    assert np.isclose(eff_dur, 1.0 - np.exp(-2.0), rtol=1e-5)
