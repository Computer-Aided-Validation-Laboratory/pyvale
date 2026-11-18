"""Validation engine behaviour."""

from dataclasses import dataclass

import numpy as np
import pytest

from pyvale import valid


def create_parallel_case() -> valid.ValidCase:
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
                np.full(points, 0.1), np.full(points, 0.2), (valid.EValidAxis.POINT,)
            ),
        )
    return valid.ValidCase(observables, sims, experiments)


@pytest.mark.parametrize("workers", [2])
def test_parallel_metrics_match_serial(workers: int) -> None:
    case = create_parallel_case()
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
    def get_scale(self) -> valid.EMetricScale:
        return valid.EMetricScale.PHYSICAL

    def get_comparison(self) -> valid.EMetricComparison:
        return valid.EMetricComparison.DISTRIBUTIONAL

    def get_outputs(self) -> tuple[valid.EMetricOutput, ...]:
        return (valid.EMetricOutput.DISTANCE,)

    def extract_outputs(self, result: valid.MetricResult) -> tuple[float, ...]:
        raise AssertionError("Failed calculations cannot produce outputs.")

    def get_kind(self) -> valid.EMetricType:
        return valid.EMetricType.AVM

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> valid.MetricResult:
        raise ValueError("Deliberate worker failure.")


def test_worker_failure_propagates_and_engine_can_run_again() -> None:
    case = create_parallel_case()
    engine = valid.ValidEngine([FailingMetric()], opts=valid.ValidEngineOpts(workers=2))
    with pytest.raises(ValueError, match="Deliberate worker failure"):
        engine.run(case)
    engine.metrics = (valid.MetricAVM(),)
    assert np.isfinite(
        engine.run(case)
        .metrics["camera"][valid.EMetricType.AVM]
        .outputs[valid.EMetricOutput.DISTANCE]
    ).any()


@pytest.mark.parametrize("workers", [2])
def test_uneven_partitions_preserve_physical_offsets(workers: int) -> None:
    point_count, component_count, time_count = (5, 2, 3)
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
    model_shifts = np.stack(
        (np.zeros_like(model_lower), -model_lower, 2 * model_lower), axis=-1
    )
    exp_shifts = np.stack(
        (np.zeros_like(exp_lower), -exp_lower, 2 * exp_lower), axis=-1
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


def create_scalar_case(observations: np.ndarray) -> valid.ValidCase:
    observables = {
        "scalar": valid.ValidObservable(
            components=("scalar",), geometry=valid.EValidGeometry.GLOBAL, coords=None
        )
    }
    quantity = valid.ValidArray(observations, (valid.EValidAxis.ALEATORY,))
    return valid.ValidCase(observables, {"scalar": quantity}, {"scalar": quantity})


def test_engine_snapshots_and_all_nan_configuration() -> None:
    metric = valid.MetricMAVM(alpha=0.1)
    metrics = [metric]
    options = valid.ValidEngineOpts(workers=1)
    engine = valid.ValidEngine(metrics, opts=options)
    metric.alpha = 0.2
    metrics.clear()
    options.workers = 0
    result = engine.run(create_scalar_case(np.array([1.0, 2.0])))
    engine.metrics[0].alpha = 0.3
    assert result.metric_configs[valid.EMetricType.MAVM].alpha == 0.1
    assert engine.opts.workers == 1
    engine.metrics[0].tol = -1.0
    with pytest.raises(ValueError, match="tolerance"):
        engine.run(create_scalar_case(np.array([np.nan])))


class MetricSampleMeans(valid.IValidMetric):
    """Exercise custom output ordering independently of built-in dispatch."""

    def get_scale(self) -> valid.EMetricScale:
        return valid.EMetricScale.PHYSICAL

    def get_comparison(self) -> valid.EMetricComparison:
        return valid.EMetricComparison.DISTRIBUTIONAL

    def get_kind(self) -> valid.EMetricType:
        return valid.EMetricType.AVM

    def get_outputs(self) -> tuple[valid.EMetricOutput, ...]:
        return (valid.EMetricOutput.ERROR, valid.EMetricOutput.DISTANCE)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> valid.MetricResult:
        return valid.ScalarMetricResult(self.get_kind(), float(np.mean(sim)))

    def extract_outputs(self, result: valid.MetricResult) -> tuple[float, ...]:
        assert isinstance(result, valid.ScalarMetricResult)
        return (result.value, result.value + 1.0)


@pytest.mark.parametrize("workers", [1, 2])
def test_custom_multiple_outputs_preserve_order_axes_and_missing_cells(
    workers: int,
) -> None:
    observations = np.array([[1.0, 3.0], [np.nan, np.nan], [4.0, 6.0]])
    observable = valid.ValidObservable(
        components=("scalar",),
        geometry=valid.EValidGeometry.POINT,
        coords=np.zeros((3, 3)),
    )
    quantity = valid.ValidArray(
        observations, (valid.EValidAxis.POINT, valid.EValidAxis.ALEATORY)
    )
    case = valid.ValidCase(
        {"scalar": observable}, {"scalar": quantity}, {"scalar": quantity}
    )
    result = valid.ValidEngine(
        [MetricSampleMeans(), valid.MetricKS()],
        opts=valid.ValidEngineOpts(workers=workers),
    ).run(case)
    field = result.metrics["scalar"][valid.EMetricType.AVM]
    assert tuple(field.outputs) == (
        valid.EMetricOutput.ERROR,
        valid.EMetricOutput.DISTANCE,
    )
    for output, expected in (
        (valid.EMetricOutput.ERROR, [2.0, np.nan, 5.0]),
        (valid.EMetricOutput.DISTANCE, [3.0, np.nan, 6.0]),
    ):
        assert field.outputs[output].shape == (3, 1, 1)
        np.testing.assert_allclose(
            valid.extract_metric_values(field, output=output), expected
        )
        selected = valid.select_metric_output(field, output)
        assert np.shares_memory(selected, field.outputs[output])
    statistic = result.metrics["scalar"][valid.EMetricType.KS].outputs[
        valid.EMetricOutput.STATISTIC
    ]
    np.testing.assert_allclose(statistic[:, 0, 0], [0.0, np.nan, 0.0])
    maximum, selection = result.find_extreme(
        "scalar", valid.EMetricType.AVM, output=valid.EMetricOutput.ERROR
    )
    assert maximum == 5.0
    assert selection[valid.EValidResAxis.POINT] == 2


@dataclass(slots=True)
class MetricSampleDifference(valid.IValidMetric):
    """Custom implementation whose metadata deliberately differs from its KS tag."""

    comparison: valid.EMetricComparison
    scale: valid.EMetricScale = valid.EMetricScale.PHYSICAL

    def get_kind(self) -> valid.EMetricType:
        return valid.EMetricType.KS

    def get_scale(self) -> valid.EMetricScale:
        return self.scale

    def get_comparison(self) -> valid.EMetricComparison:
        return self.comparison

    def get_outputs(self) -> tuple[valid.EMetricOutput, ...]:
        return (valid.EMetricOutput.STATISTIC,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> valid.ScalarMetricResult:
        return valid.ScalarMetricResult(
            self.get_kind(), float(np.sum(sim) - np.sum(exp))
        )

    def extract_outputs(self, result: valid.MetricResult) -> tuple[float, ...]:
        assert isinstance(result, valid.ScalarMetricResult)
        return (result.value,)


def test_custom_paired_classification_uses_joint_missing_observation_mask() -> None:
    axes = (valid.EValidAxis.ALEATORY,)
    valid_case = valid.ValidCase(
        {
            "length": valid.ValidObservable(
                components=("length",),
                geometry=valid.EValidGeometry.GLOBAL,
                phys_units="m",
            )
        },
        {"length": valid.ValidArray(np.array([1.0, np.nan, 3.0]), axes)},
        {"length": valid.ValidArray(np.array([0.0, 20.0, np.nan]), axes)},
    )
    metric = MetricSampleDifference(valid.EMetricComparison.PAIRED)
    result = valid.ValidEngine([metric]).run(valid_case)
    field = result.metrics["length"][valid.EMetricType.KS]
    np.testing.assert_array_equal(
        field.outputs[valid.EMetricOutput.STATISTIC], [[[1.0]]]
    )
