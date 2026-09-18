# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Unit tests for validation metrics and strategy pattern interfaces."""

import numpy as np
import pytest

from pyvale.valid.metrics import (
    calc_avm_1d,
    calc_ks_1d,
    calc_cvm_1d,
    calc_u_pooling_1d,
    calc_deterministic_metrics_1d,
)
from pyvale.valid.strategy import (
    MetricMAVM,
    MetricAVM,
    MetricKS,
    MetricCVM,
    MetricRMSE,
    MetricRelativeError,
)


def test_avm_1d_shift() -> None:
    """AVM (1-Wasserstein) of a constant shift equals the shift distance."""
    x = np.linspace(0.0, 10.0, 100)
    shift = 3.5
    y = x + shift

    val = calc_avm_1d(y, x)
    assert val == pytest.approx(shift, rel=1e-5)


def test_ks_1d_identical_and_separated() -> None:
    """KS distance is 0 for identical and 1 for separated distributions."""
    x = np.linspace(0.0, 5.0, 50)
    assert calc_ks_1d(x, x) == pytest.approx(0.0, abs=1e-10)

    y = np.linspace(10.0, 15.0, 50)
    assert calc_ks_1d(x, y) == pytest.approx(1.0, abs=1e-5)


def test_cvm_1d() -> None:
    """Cramér-von Mises distance is 0 for identical distributions."""
    x = np.linspace(0.0, 5.0, 50)
    assert calc_cvm_1d(x, x) == pytest.approx(0.0, abs=1e-10)


def test_u_pooling_1d() -> None:
    """U-pooling metric produces near zero for matching observations."""
    probs = np.linspace(0.0, 1.0, 100)
    quants = np.linspace(0.0, 10.0, 100)
    cdfs = [(probs, quants) for _ in range(50)]
    obs = np.linspace(0.0, 10.0, 50)

    u_metric = calc_u_pooling_1d(cdfs, obs)
    assert u_metric >= 0.0
    assert u_metric < 0.05


def test_deterministic_metrics() -> None:
    """Deterministic error metric calculations (RMSE, Rel Err)."""
    sim = np.array([10.0, 20.0, 30.0])
    exp = np.array([10.0, 22.0, 28.0])

    res = calc_deterministic_metrics_1d(sim, exp)
    assert res["absolute_error"] == pytest.approx(4.0 / 3.0)
    assert res["rmse"] == pytest.approx(np.sqrt(8.0 / 3.0))


def test_strategy_pattern_metrics() -> None:
    """Metric strategy classes conform to IValMetric interface."""
    sim = np.linspace(0.0, 10.0, 100)
    exp = np.linspace(1.0, 11.0, 100)

    mavm_strat = MetricMAVM(alpha=0.05)
    res_mavm = mavm_strat.calc(sim, exp)
    assert res_mavm.d_total >= 0.0

    avm_strat = MetricAVM()
    assert avm_strat.calc(sim, exp) == pytest.approx(1.0, rel=1e-3)

    ks_strat = MetricKS()
    assert 0.0 <= ks_strat.calc(sim, exp) <= 1.0

    cvm_strat = MetricCVM()
    assert cvm_strat.calc(sim, exp) >= 0.0

    rmse_strat = MetricRMSE()
    assert rmse_strat.calc(sim, exp) == pytest.approx(1.0)

    rel_strat = MetricRelativeError()
    assert rel_strat.calc(sim, exp) > 0.0
