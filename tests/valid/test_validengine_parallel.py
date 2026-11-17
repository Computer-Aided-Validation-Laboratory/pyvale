"""Serial and process-pool metric evaluation parity."""

import numpy as np
import pytest

from pyvale import valid


def create_case() -> valid.ValidCase:
    generator = np.random.default_rng(42)
    observables = {}
    sims = {}
    experiments = {}
    for name, points in (("camera", 7), ("gauge", 1)):
        observables[name] = valid.ValidObservable(
            components=("temperature",),
            geometry=valid.EValidGeometry.POINT,
            coords=np.zeros((points, 3)),
            times=np.array([0.0, 1.0]),
        )
        # Strided input exercises process serialization and point slicing.
        samples = generator.normal(size=(points, 1, 2, 2, 20))[..., ::2]
        samples[0, 0, 0, 0] = np.nan
        sims[name] = valid.ValidArray(
            samples,
            (
                valid.EValidAxis.POINT,
                valid.EValidAxis.COMPONENT,
                valid.EValidAxis.TIME,
                valid.EValidAxis.EPISTEMIC_TRANS,
                valid.EValidAxis.ALEATORY,
            ),
            valid.EpistemicOffsets(0.05, 0.1),
        )
        experiments[name] = valid.ValidArray(
            generator.normal(size=(points, 1, 2, 10)),
            (
                valid.EValidAxis.POINT,
                valid.EValidAxis.COMPONENT,
                valid.EValidAxis.TIME,
                valid.EValidAxis.ALEATORY,
            ),
            valid.EpistemicOffsets(
                np.full(points, 0.1),
                np.full(points, 0.2),
                (valid.EValidAxis.POINT,),
            ),
        )
    return valid.ValidCase(observables, sims, experiments)


@pytest.mark.parametrize("workers", [2, 5])
def test_parallel_metrics_match_serial(workers: int) -> None:
    case = create_case()
    metrics = [
        valid.MetricMAVM(),
        valid.MetricAVM(),
        valid.MetricKS(),
        valid.MetricCVM(),
    ]
    serial = valid.ValidEngine(metrics).run(case)
    parallel = valid.ValidEngine(
        metrics, opts=valid.ValidEngineOpts(workers=workers)
    ).run(case)
    assert parallel.observables is case.observables
    assert parallel.metric_configs.keys() == serial.metric_configs.keys()
    serial_mavm = serial.metric_configs[valid.EMetricType.MAVM]
    parallel_mavm = parallel.metric_configs[valid.EMetricType.MAVM]
    assert isinstance(serial_mavm, valid.MetricMAVM)
    assert isinstance(parallel_mavm, valid.MetricMAVM)
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


class FailingMetric(valid.IValidMetric):
    def get_outputs(self) -> tuple[valid.EMetricOutput, ...]:
        return (valid.EMetricOutput.DISTANCE,)

    def extract_outputs(self, result: valid.MetricResult) -> tuple[float, ...]:
        raise AssertionError("Failed calculations cannot produce outputs.")

    def get_kind(self) -> valid.EMetricType:
        return valid.EMetricType.AVM

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> valid.MetricResult:
        raise ValueError("Deliberate worker failure.")


def test_worker_failure_propagates_and_engine_can_run_again() -> None:
    case = create_case()
    engine = valid.ValidEngine([FailingMetric()], opts=valid.ValidEngineOpts(workers=2))
    with pytest.raises(ValueError, match="Deliberate worker failure"):
        engine.run(case)
    engine.metrics = (valid.MetricAVM(),)
    assert np.isfinite(
        engine.run(case)
        .metrics["camera"][valid.EMetricType.AVM]
        .outputs[valid.EMetricOutput.DISTANCE]
    ).any()


def test_invalid_observations_fail_when_case_is_initialized() -> None:
    case = create_case()
    case.sim["camera"].data[0, 0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        valid.ValidCase(case.observables, case.sim, case.exp)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_options(value: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        valid.ValidEngineOpts(workers=value)


@pytest.mark.parametrize("workers", [2, 7])
def test_uneven_partitions_preserve_physical_offsets(workers: int) -> None:
    point_count, component_count, time_count = 5, 2, 3
    physical = (
        np.arange(point_count)[:, None, None] * 10.0
        + np.arange(component_count)[None, :, None] * 2.0
        + np.arange(time_count)[None, None, :] * 0.5
    )
    model_scenarios = np.array([0.0, 2.0])
    exp_scenarios = np.array([0.25, 0.75])
    samples = np.arange(3.0)
    model = physical[..., None, None] + model_scenarios[:, None] + samples
    exp = physical[..., None, None] + exp_scenarios[:, None] + samples
    model_lower = np.arange(point_count)[:, None] * 0.1 + np.arange(time_count) * 0.05
    exp_lower = np.full(physical.shape, 0.3)
    observable = valid.ValidObservable(
        components=("x", "y"),
        geometry=valid.EValidGeometry.POINT,
        coords=np.zeros((point_count, 3)),
        times=np.arange(float(time_count)),
    )
    physical_axes = (
        valid.EValidAxis.POINT,
        valid.EValidAxis.COMPONENT,
        valid.EValidAxis.TIME,
    )
    case = valid.ValidCase(
        {"response": observable},
        {
            "response": valid.ValidArray(
                model,
                (
                    *physical_axes,
                    valid.EValidAxis.EPISTEMIC_TRANS,
                    valid.EValidAxis.ALEATORY,
                ),
                valid.EpistemicOffsets(
                    model_lower,
                    2 * model_lower,
                    (valid.EValidAxis.POINT, valid.EValidAxis.TIME),
                ),
            )
        },
        {
            "response": valid.ValidArray(
                exp,
                (
                    *physical_axes,
                    valid.EValidAxis.EPISTEMIC_NUIS,
                    valid.EValidAxis.ALEATORY,
                ),
                valid.EpistemicOffsets(exp_lower, 2 * exp_lower, physical_axes),
            )
        },
    )
    result = valid.ValidEngine(
        [valid.MetricAVM(), valid.MetricRMSE()],
        opts=valid.ValidEngineOpts(workers=workers),
    ).run(case)
    # Axes: point, component, time, sim_trans, exp_nuis, sim_offset, exp_offset.
    model_shifts = np.stack(
        (np.zeros_like(model_lower), -model_lower, 2 * model_lower), axis=-1
    )
    exp_shifts = np.stack(
        (
            np.zeros_like(exp_lower),
            -exp_lower,
            2 * exp_lower,
        ),
        axis=-1,
    )
    expected = np.abs(
        model_scenarios[None, None, None, :, None, None, None]
        - exp_scenarios[None, None, None, None, :, None, None]
        + model_shifts[:, None, :, None, None, :, None]
        - exp_shifts[:, :, :, None, None, None, :]
    )
    for metric, output in (
        (valid.EMetricType.AVM, valid.EMetricOutput.DISTANCE),
        (valid.EMetricType.RMSE, valid.EMetricOutput.ERROR),
    ):
        np.testing.assert_allclose(
            result.metrics["response"][metric].outputs[output],
            expected,
            rtol=1e-13,
            atol=1e-12,
        )
