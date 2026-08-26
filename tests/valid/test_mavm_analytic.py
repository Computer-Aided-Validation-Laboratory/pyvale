# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Analytic verification tests for the Modified Area Validation Metric (MAVM).

Based on exact discrete overlap cases from Whiting et al. and the validation
benchmarks in fullfieldvalmetrics.
"""

import numpy as np
import pytest

from pyvale.valid.metrics import calc_mavm_1d, calc_mavm_pbox_1d


def test_mavm_analytic_more_exp_no_remainder_no_overlap() -> None:
    """Test 01: N_exp > N_sim, no remainder, no overlap."""
    num_exp = 4
    num_sim = 2
    num_diff = num_exp - num_sim
    exp_data = np.arange(0, num_exp, dtype=np.float64) - num_exp / 2.0
    sim_data = exp_data[num_diff:] + exp_data[-1] + 1.0

    res = calc_mavm_1d(sim_data, exp_data, alpha=0.05)

    assert res.d_total > 0.0
    assert res.d_plus > 0.0
    assert res.d_minus == 0.0


def test_mavm_analytic_more_sim_no_remainder_no_overlap() -> None:
    """Test 02: N_sim > N_exp, no remainder, no overlap."""
    num_exp = 2
    num_sim = 4
    num_diff = num_sim - num_exp
    sim_data = np.arange(0, num_sim, dtype=np.float64) - num_sim / 2.0
    exp_data = sim_data[num_diff:] + sim_data[-1] + 2.0

    res = calc_mavm_1d(sim_data, exp_data, alpha=0.05)

    assert res.d_total > 0.0
    assert res.d_minus > 0.0


def test_mavm_analytic_more_exp_symmetric_overlap() -> None:
    """Test 03: N_exp > N_sim, symmetric overlap."""
    num_exp = 4
    num_sim = 2
    exp_data = np.arange(0, num_exp, dtype=np.float64) - num_exp / 2.0
    sim_data = np.arange(0, num_sim, dtype=np.float64) - num_sim / 2.0

    res = calc_mavm_1d(sim_data, exp_data, alpha=0.05)

    assert res.d_total >= 0.0


def test_mavm_analytic_more_exp_with_remainder_no_overlap() -> None:
    """Test 04: N_exp > N_sim, with remainder step, no overlap."""
    num_exp = 4
    num_sim = 3
    num_diff = num_exp - num_sim
    exp_data = np.arange(0, num_exp, dtype=np.float64) - num_exp / 2.0
    sim_data = exp_data[num_diff:] + exp_data[-1] + 1.0

    res = calc_mavm_1d(sim_data, exp_data, alpha=0.05)

    assert res.d_total > 0.0
    assert res.d_plus > 0.0


def test_mavm_analytic_more_sim_with_remainder_no_overlap() -> None:
    """Test 05: N_sim > N_exp, with remainder step, no overlap."""
    num_exp = 3
    num_sim = 4
    num_diff = num_sim - num_exp
    sim_data = np.arange(0, num_sim, dtype=np.float64) - num_sim / 2.0
    exp_data = sim_data[num_diff:] + sim_data[-1] + 2.0

    res = calc_mavm_1d(sim_data, exp_data, alpha=0.05)

    assert res.d_total > 0.0
    assert res.d_minus > 0.0


def test_mavm_analytic_identical_distributions() -> None:
    """Identical distributions have d_total = 0 when confidence interval
    encompasses model.
    """
    data = np.linspace(10.0, 20.0, 50)
    # Model and experiment have identical distribution
    res = calc_mavm_1d(data, data, alpha=0.05)

    # Since exp confidence bounds expand around empirical CDF, model is within
    assert res.d_plus == pytest.approx(0.0, abs=1e-10)
    assert res.d_minus == pytest.approx(0.0, abs=1e-10)
    assert res.d_total == pytest.approx(0.0, abs=1e-10)


def test_mavm_analytic_known_shift_asymptotic() -> None:
    """A known uniform shift of delta between large samples yields d_plus."""
    rng = np.random.default_rng(42)
    exp_samples = rng.normal(loc=100.0, scale=1.0, size=5000)
    # Shift model to the right by 5.0
    shift = 5.0
    sim_samples = rng.normal(loc=100.0 + shift, scale=1.0, size=5000)

    res = calc_mavm_1d(sim_samples, exp_samples, alpha=0.05)

    assert res.d_minus == pytest.approx(0.0, abs=1e-3)
    # Due to 95% CI on exp, d_plus is approximately shift - t_alpha*SE
    assert 4.8 < res.d_plus < 5.1


def test_mavm_pbox() -> None:
    """Epistemic p-box calculation evaluates outer envelope bounds."""
    rng = np.random.default_rng(123)
    exp_data = rng.normal(loc=50.0, scale=2.0, size=100)

    # Simulation lower bound shifted left, upper bound shifted right
    sim_min = rng.normal(loc=45.0, scale=2.0, size=200)
    sim_max = rng.normal(loc=55.0, scale=2.0, size=200)

    res = calc_mavm_pbox_1d(sim_min, sim_max, exp_data, alpha=0.05)

    assert res.d_plus > 0.0
    assert res.d_minus > 0.0
    assert res.d_total == res.d_plus + res.d_minus
