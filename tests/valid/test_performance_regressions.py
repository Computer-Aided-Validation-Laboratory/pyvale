"""Independent cell comparisons for optimized validation execution."""

from collections.abc import Callable
from itertools import product
from pathlib import Path
import runpy
from typing import cast

import numpy as np
import pytest
from scipy import stats

import pyvale.valid as val


@pytest.mark.parametrize("workers,chunk_size", [(1, 1), (1, 17), (2, 17)])
def test_engine_matches_standalone_uncertainty_products(
    workers: int, chunk_size: int
) -> None:
    generator = np.random.default_rng(1729)
    simulation = generator.integers(-3, 4, (1, 1, 1, 2, 2, 8)).astype(float)
    experiment = generator.integers(-2, 5, (1, 1, 1, 2, 2, 5)).astype(float)
    simulation[..., 0, 0, 0] = np.nan
    experiment[..., 1, 0, :] = np.nan
    experiment[..., 0, 1, 0] = np.nan
    axes = tuple(val.EValAxis)
    simulated = val.ValArray(simulation, axes, val.EpistemicOffsets(0.25, 0.5))
    observed = val.ValArray(experiment, axes, val.EpistemicOffsets(0.4, 0.6))
    domain = val.ValDomain({"response": val.ValObservable(
        val.ValGeometry(val.EValGeometry.POINT, np.zeros((1, 3))),
        val.EValOperator.SAMPLE, ("response",), np.array([0.0]),
    )})
    metrics = [val.MetricMAVM(), val.MetricAVM(), val.MetricKS(), val.MetricCVM()]
    result = val.ValidationEngine(
        metrics, opts=val.ValEngineOpts(workers=workers, chunk_size=chunk_size)
    ).run(val.ValCase(domain, {"response": simulated}, {"response": observed}))
    for sim_trans, sim_nuis, exp_trans, exp_nuis, sim_offset, exp_offset in product(
        range(2), range(2), range(2), range(2), val.EEpisOffset, val.EEpisOffset
    ):
        sim_samples = simulated.select_aleatory(
            0, 0, 0, epis_trans_idx=sim_trans, epis_nuis_idx=sim_nuis,
            offset=sim_offset,
        )
        exp_samples = observed.select_aleatory(
            0, 0, 0, epis_trans_idx=exp_trans, epis_nuis_idx=exp_nuis,
            offset=exp_offset,
        )
        selection = (
            0, 0, 0, sim_trans, sim_nuis, exp_trans, exp_nuis, sim_offset, exp_offset
        )
        for metric in metrics:
            field = result.get("response", metric.get_kind())
            if not np.any(np.isfinite(exp_samples)):
                assert np.isnan(field.metric[selection])
                continue
            expected = metric.calc(sim_samples, exp_samples)
            if isinstance(expected, val.MAVMResult):
                assert field.d_plus is not None and field.d_minus is not None
                np.testing.assert_allclose(
                    [field.metric[selection], field.d_plus[selection],
                     field.d_minus[selection]],
                    [expected.d_total, expected.d_plus, expected.d_minus],
                    rtol=1e-13, atol=1e-12,
                )
            else:
                assert field.metric[selection] == pytest.approx(
                    expected.value, rel=1e-13, abs=1e-12
                )


@pytest.mark.parametrize("sim_count,exp_count", [(2, 3), (8, 8), (64, 8), (11, 29)])
@pytest.mark.parametrize("repeated", [False, True])
def test_distribution_statistics_match_scipy(
    sim_count: int, exp_count: int, *, repeated: bool
) -> None:
    generator = np.random.default_rng(8921)
    if repeated:
        simulation = generator.integers(-2, 4, sim_count).astype(float)
        experiment = generator.integers(-1, 5, exp_count).astype(float)
    else:
        simulation = generator.normal(0.0, 1.0, sim_count)
        experiment = generator.normal(0.3, 1.1, exp_count)
    for sim_samples, exp_samples in (
        (simulation, experiment),
        (experiment[::-1], simulation[::-1]),
        (np.zeros(sim_count), np.zeros(exp_count)),
        (np.zeros(sim_count), np.ones(exp_count)),
    ):
        assert val.calc_ks_1d(sim_samples, exp_samples) == pytest.approx(
            stats.ks_2samp(sim_samples, exp_samples).statistic,
            rel=1e-13, abs=1e-14,
        )
        assert val.calc_cvm_1d(sim_samples, exp_samples) == pytest.approx(
            stats.cramervonmises_2samp(sim_samples, exp_samples).statistic,
            rel=1e-13, abs=1e-14,
        )


@pytest.mark.parametrize("duplicates", list(val.EMAVMDuplicates))
def test_native_and_interpreted_mavm_agree(
    duplicates: val.EMAVMDuplicates,
) -> None:
    source = (
        Path(__file__).resolve().parents[2] / "src/pyvale/valid/cython/mavm.py"
    )
    interpreted = cast(
        Callable[..., val.MAVMResult], runpy.run_path(str(source))["cyth_calc_mavm_1d"]
    )
    generator = np.random.default_rng(2751)
    simulation = generator.normal(size=64)
    experiment = generator.normal(size=8)
    if duplicates is val.EMAVMDuplicates.INCLUDE:
        simulation = np.round(simulation)
        experiment = np.round(experiment)
    simulation[0] = np.nan
    experiment[0] = np.nan
    native = val.calc_mavm_1d(
        simulation, experiment, duplicates=duplicates
    )
    reference = interpreted(simulation, experiment, duplicates=duplicates)
    np.testing.assert_allclose(
        [native.d_total, native.d_plus, native.d_minus],
        [reference.d_total, reference.d_plus, reference.d_minus],
        rtol=1e-13, atol=1e-12,
    )
    np.testing.assert_array_equal(native.model_quantiles, reference.model_quantiles)
    np.testing.assert_array_equal(native.model_probs, reference.model_probs)
    np.testing.assert_array_equal(native.exp_quantiles, reference.exp_quantiles)
    np.testing.assert_array_equal(native.exp_probs, reference.exp_probs)


def test_small_workload_with_worker_cap_matches_analytic_shift() -> None:
    domain = val.ValDomain({"response": val.ValObservable(
        val.ValGeometry(val.EValGeometry.POINT, np.zeros((1, 3))),
        val.EValOperator.SAMPLE, ("response",),
    )})
    axes = (val.EValAxis.POINT, val.EValAxis.COMPONENT, val.EValAxis.ALEATORY)
    simulation = np.array([[[0.0, 1.0]]])
    case = val.ValCase(
        domain, {"response": val.ValArray(simulation, axes)},
        {"response": val.ValArray(simulation + 0.125, axes)},
    )
    metrics = [val.MetricAVM(), val.MetricRMSE(), val.MetricRelErr(
        normalization=val.ERelativeErrorNormalization.CONSTANT, constant=1.0
    )]
    for workers in (1, 4):
        result = val.ValidationEngine(
            metrics, opts=val.ValEngineOpts(workers=workers, chunk_size=1)
        ).run(case)
        for metric in metrics:
            assert result.get("response", metric.get_kind()).metric.item() == 0.125
