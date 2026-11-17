"""Independent cell comparisons for optimized validation execution."""

import runpy
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from scipy import stats

from pyvale import valid


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_engine_matches_standalone_uncertainty_products(workers: int) -> None:
    generator = np.random.default_rng(1729)
    sim_data = generator.integers(-3, 4, (1, 1, 1, 2, 2, 8)).astype(float)
    exp = generator.integers(-2, 5, (1, 1, 1, 2, 2, 5)).astype(float)
    sim_data[..., 0, 0, 0] = np.nan
    exp[..., 1, 0, :] = np.nan
    exp[..., 0, 1, 0] = np.nan
    axes = tuple(valid.EValidAxis)
    simulated = valid.ValidArray(sim_data, axes, valid.EpistemicOffsets(0.25, 0.5))
    observed = valid.ValidArray(exp, axes, valid.EpistemicOffsets(0.4, 0.6))
    observables = {
        "response": valid.ValidObservable(
            components=("response",),
            geometry=valid.EValidGeometry.POINT,
            coords=np.zeros((1, 3)),
            times=np.array([0.0]),
        )
    }
    metrics = [
        valid.MetricMAVM(),
        valid.MetricAVM(),
        valid.MetricKS(),
        valid.MetricCVM(),
    ]
    result = valid.ValidEngine(
        metrics, opts=valid.ValidEngineOpts(workers=workers)
    ).run(valid.ValidCase(observables, {"response": simulated}, {"response": observed}))
    for sim_trans, sim_nuis, exp_trans, exp_nuis, sim_offset, exp_offset in product(
        range(2), range(2), range(2), range(2), valid.EEpisOffset, valid.EEpisOffset
    ):
        sim_samples = simulated.select_aleatory(
            0,
            0,
            0,
            epis_trans_idx=sim_trans,
            epis_nuis_idx=sim_nuis,
            offset=sim_offset,
        )
        exp_samples = observed.select_aleatory(
            0,
            0,
            0,
            epis_trans_idx=exp_trans,
            epis_nuis_idx=exp_nuis,
            offset=exp_offset,
        )
        selection = (
            0,
            0,
            0,
            sim_trans,
            sim_nuis,
            exp_trans,
            exp_nuis,
            sim_offset,
            exp_offset,
        )
        for metric in metrics:
            field = result.metrics["response"][metric.get_kind()]
            assert tuple(field.outputs) == metric.get_outputs()
            if not np.any(np.isfinite(exp_samples)):
                for array in field.outputs.values():
                    assert np.isnan(array[selection])
                continue
            expected = metric.calc(sim_samples, exp_samples)
            expected_outputs = metric.extract_outputs(expected)
            for output, expected_value in zip(
                metric.get_outputs(), expected_outputs, strict=True
            ):
                assert field.outputs[output][selection] == pytest.approx(
                    expected_value, rel=1e-13, abs=1e-12
                )


@pytest.mark.parametrize("sim_count,exp_count", [(2, 3), (8, 8), (64, 8), (11, 29)])
@pytest.mark.parametrize("repeated", [False, True])
def test_distribution_statistics_match_scipy(
    sim_count: int, exp_count: int, *, repeated: bool
) -> None:
    generator = np.random.default_rng(8921)
    if repeated:
        sim_data = generator.integers(-2, 4, sim_count).astype(float)
        exp = generator.integers(-1, 5, exp_count).astype(float)
    else:
        sim_data = generator.normal(0.0, 1.0, sim_count)
        exp = generator.normal(0.3, 1.1, exp_count)
    for sim_samples, exp_samples in (
        (sim_data, exp),
        (exp[::-1], sim_data[::-1]),
        (np.zeros(sim_count), np.zeros(exp_count)),
        (np.zeros(sim_count), np.ones(exp_count)),
    ):
        assert valid.calc_ks_1d(sim_samples, exp_samples) == pytest.approx(
            stats.ks_2samp(sim_samples, exp_samples).statistic,
            rel=1e-13,
            abs=1e-14,
        )
        assert valid.calc_cvm_1d(sim_samples, exp_samples) == pytest.approx(
            stats.cramervonmises_2samp(sim_samples, exp_samples).statistic,
            rel=1e-13,
            abs=1e-14,
        )


@pytest.mark.parametrize("duplicates", list(valid.EMAVMDuplicates))
def test_native_and_interpreted_mavm_agree(
    duplicates: valid.EMAVMDuplicates,
) -> None:
    source = Path(__file__).resolve().parents[2] / "src/pyvale/valid/cython/mavm.py"
    interpreted = cast(
        Callable[..., valid.MAVMResult],
        runpy.run_path(str(source))["cyth_calc_mavm_1d"],
    )
    generator = np.random.default_rng(2751)
    sim_data = generator.normal(size=64)
    exp = generator.normal(size=8)
    if duplicates is valid.EMAVMDuplicates.INCLUDE:
        sim_data = np.round(sim_data)
        exp = np.round(exp)
    sim_data[0] = np.nan
    exp[0] = np.nan
    native = valid.calc_mavm_1d(sim_data, exp, duplicates=duplicates)
    reference = interpreted(sim_data, exp, duplicates=duplicates)
    np.testing.assert_allclose(
        [native.d_plus, native.d_minus],
        [reference.d_plus, reference.d_minus],
        rtol=1e-13,
        atol=1e-12,
    )
    np.testing.assert_array_equal(native.model_quantiles, reference.model_quantiles)
    np.testing.assert_array_equal(native.model_probs, reference.model_probs)
    np.testing.assert_array_equal(native.exp_quantiles, reference.exp_quantiles)
    np.testing.assert_array_equal(native.exp_probs, reference.exp_probs)


def test_small_workload_with_worker_cap_matches_analytic_shift() -> None:
    observables = {
        "response": valid.ValidObservable(
            components=("response",),
            geometry=valid.EValidGeometry.POINT,
            coords=np.zeros((1, 3)),
        )
    }
    axes = (
        valid.EValidAxis.POINT,
        valid.EValidAxis.COMPONENT,
        valid.EValidAxis.ALEATORY,
    )
    sim_data = np.array([[[0.0, 1.0]]])
    case = valid.ValidCase(
        observables,
        {"response": valid.ValidArray(sim_data, axes)},
        {"response": valid.ValidArray(sim_data + 0.125, axes)},
    )
    metrics = [
        valid.MetricAVM(),
        valid.MetricRMSE(),
        valid.MetricRelErr(normalisation=valid.ERelErrNorm.CONSTANT, constant=1.0),
    ]
    for workers in (1, 4):
        result = valid.ValidEngine(
            metrics, opts=valid.ValidEngineOpts(workers=workers)
        ).run(case)
        for metric in metrics:
            assert (
                result.metrics["response"][metric.get_kind()]
                .outputs[metric.get_outputs()[0]]
                .item()
                == 0.125
            )
