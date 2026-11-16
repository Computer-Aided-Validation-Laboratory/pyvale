"""Serial and process-pool metric evaluation parity."""

import numpy as np
import pytest

import pyvale.valid as val


def create_case() -> val.ValCase:
    generator = np.random.default_rng(42)
    observables = {}
    simulations = {}
    experiments = {}
    for name, points in (("camera", 7), ("gauge", 1)):
        observables[name] = val.ValObservable(
            components=("temperature",),
            geometry=val.EValGeometry.POINT,
            coords=np.zeros((points, 3)),
            times=np.array([0.0, 1.0]),
        )
        # Strided input exercises process serialization and point slicing.
        samples = generator.normal(size=(points, 1, 2, 2, 20))[..., ::2]
        samples[0, 0, 0, 0] = np.nan
        simulations[name] = val.ValArray(
            samples,
            (
                val.EValAxis.POINT,
                val.EValAxis.COMPONENT,
                val.EValAxis.TIME,
                val.EValAxis.EPISTEMIC_TRANS,
                val.EValAxis.ALEATORY,
            ),
            val.EpistemicOffsets(0.05, 0.1),
        )
        experiments[name] = val.ValArray(
            generator.normal(size=(points, 1, 2, 10)),
            (
                val.EValAxis.POINT,
                val.EValAxis.COMPONENT,
                val.EValAxis.TIME,
                val.EValAxis.ALEATORY,
            ),
            val.EpistemicOffsets(
                np.full(points, 0.1),
                np.full(points, 0.2),
                (val.EValAxis.POINT,),
            ),
        )
    return val.ValCase(observables, simulations, experiments)


@pytest.mark.parametrize("workers", [2, 5])
def test_parallel_metrics_match_serial(workers: int) -> None:
    case = create_case()
    metrics = [val.MetricMAVM(), val.MetricAVM(), val.MetricKS(), val.MetricCVM()]
    serial = val.ValidationEngine(metrics).run(case)
    parallel = val.ValidationEngine(
        metrics, opts=val.ValEngineOpts(workers=workers)
    ).run(case)
    assert parallel.observables is case.observables
    assert parallel.metric_configs.keys() == serial.metric_configs.keys()
    serial_mavm = serial.metric_configs[val.EMetricType.MAVM]
    parallel_mavm = parallel.metric_configs[val.EMetricType.MAVM]
    assert isinstance(serial_mavm, val.MetricMAVM)
    assert isinstance(parallel_mavm, val.MetricMAVM)
    assert (parallel_mavm.alpha, parallel_mavm.duplicates, parallel_mavm.tol) == (
        serial_mavm.alpha,
        serial_mavm.duplicates,
        serial_mavm.tol,
    )
    assert parallel_mavm is not serial_mavm
    for name in case.observables:
        for metric in metrics:
            expected = serial.metrics[name][metric.get_kind()]
            actual = parallel.metrics[name][metric.get_kind()]
            assert actual.axes == expected.axes
            assert actual.scenarios == expected.scenarios
            assert actual.outputs.keys() == expected.outputs.keys()
            for output in expected.outputs:
                np.testing.assert_allclose(
                    actual.outputs[output],
                    expected.outputs[output],
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
    assert np.isnan(case.sim["camera"].data[0, 0, 0, 0]).all()


class FailingMetric(val.IValMetric):
    def get_outputs(self) -> tuple[val.EMetricOutput, ...]:
        return (val.EMetricOutput.DISTANCE,)

    def extract_outputs(self, result: val.MetricResult) -> tuple[float, ...]:
        raise AssertionError("Failed calculations cannot produce outputs.")

    def get_kind(self) -> val.EMetricType:
        return val.EMetricType.AVM

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> val.MetricResult:
        raise ValueError("Deliberate worker failure.")


def test_worker_failure_propagates_and_engine_can_run_again() -> None:
    case = create_case()
    engine = val.ValidationEngine([FailingMetric()], opts=val.ValEngineOpts(workers=2))
    with pytest.raises(ValueError, match="Deliberate worker failure"):
        engine.run(case)
    engine.metrics = (val.MetricAVM(),)
    assert np.isfinite(
        engine.run(case)
        .metrics["camera"][val.EMetricType.AVM]
        .outputs[val.EMetricOutput.DISTANCE]
    ).any()


def test_invalid_observations_fail_when_case_is_initialized() -> None:
    case = create_case()
    case.sim["camera"].data[0, 0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        val.ValCase(case.observables, case.sim, case.exp)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_options(value: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        val.ValEngineOpts(workers=value)


@pytest.mark.parametrize("workers", [2, 7])
def test_uneven_partitions_preserve_physical_offsets(workers: int) -> None:
    point_count, component_count, time_count = 5, 2, 3
    physical = (
        np.arange(point_count)[:, None, None] * 10.0
        + np.arange(component_count)[None, :, None] * 2.0
        + np.arange(time_count)[None, None, :] * 0.5
    )
    model_scenarios = np.array([0.0, 2.0])
    experimental_scenarios = np.array([0.25, 0.75])
    samples = np.arange(3.0)
    model = physical[..., None, None] + model_scenarios[:, None] + samples
    experiment = physical[..., None, None] + experimental_scenarios[:, None] + samples
    model_lower = np.arange(point_count)[:, None] * 0.1 + np.arange(time_count) * 0.05
    experimental_lower = np.full(physical.shape, 0.3)
    observable = val.ValObservable(
        components=("x", "y"),
        geometry=val.EValGeometry.POINT,
        coords=np.zeros((point_count, 3)),
        times=np.arange(float(time_count)),
    )
    physical_axes = (val.EValAxis.POINT, val.EValAxis.COMPONENT, val.EValAxis.TIME)
    case = val.ValCase(
        {"response": observable},
        {
            "response": val.ValArray(
                model,
                (*physical_axes, val.EValAxis.EPISTEMIC_TRANS, val.EValAxis.ALEATORY),
                val.EpistemicOffsets(
                    model_lower,
                    2 * model_lower,
                    (val.EValAxis.POINT, val.EValAxis.TIME),
                ),
            )
        },
        {
            "response": val.ValArray(
                experiment,
                (*physical_axes, val.EValAxis.EPISTEMIC_NUIS, val.EValAxis.ALEATORY),
                val.EpistemicOffsets(
                    experimental_lower, 2 * experimental_lower, physical_axes
                ),
            )
        },
    )
    result = val.ValidationEngine(
        [val.MetricAVM(), val.MetricRMSE()], opts=val.ValEngineOpts(workers=workers)
    ).run(case)
    # Axes: point, component, time, sim_trans, exp_nuis, sim_offset, exp_offset.
    model_shifts = np.stack(
        (np.zeros_like(model_lower), -model_lower, 2 * model_lower), axis=-1
    )
    experimental_shifts = np.stack(
        (
            np.zeros_like(experimental_lower),
            -experimental_lower,
            2 * experimental_lower,
        ),
        axis=-1,
    )
    expected = np.abs(
        model_scenarios[None, None, None, :, None, None, None]
        - experimental_scenarios[None, None, None, None, :, None, None]
        + model_shifts[:, None, :, None, None, :, None]
        - experimental_shifts[:, :, :, None, None, None, :]
    )
    for metric, output in (
        (val.EMetricType.AVM, val.EMetricOutput.DISTANCE),
        (val.EMetricType.RMSE, val.EMetricOutput.ERROR),
    ):
        np.testing.assert_allclose(
            result.metrics["response"][metric].outputs[output],
            expected,
            rtol=1e-13,
            atol=1e-12,
        )
