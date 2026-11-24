"""Validation engine behaviour."""

from dataclasses import dataclass

import numpy as np
import pytest

import pyvale.validation as vl


def create_parallel_case() -> vl.ValidCase:
    generator = np.random.default_rng(42)
    observables = {}
    sims = {}
    experiments = {}
    for name, points in (("camera", 7), ("gauge", 1)):
        observables[name] = vl.ValidObservable(
            components=("temperature",),
            geometry=vl.EValidGeometry.POINT,
            coords=np.zeros((points, 3)),
            times=np.array([0.0, 1.0]),
        )
        samples = generator.normal(size=(points, 1, 2, 2, 20))[..., ::2]
        samples[0, 0, 0, 0] = np.nan
        sims[name] = vl.ValidArray(
            samples,
            (
                vl.EValidAxis.POINT,
                vl.EValidAxis.COMPONENT,
                vl.EValidAxis.TIME,
                vl.EValidAxis.EPISTEMIC_TRANS,
                vl.EValidAxis.ALEATORY,
            ),
            vl.EpistemicOffsets(0.05, 0.1),
        )
        experiments[name] = vl.ValidArray(
            generator.normal(size=(points, 1, 2, 10)),
            (
                vl.EValidAxis.POINT,
                vl.EValidAxis.COMPONENT,
                vl.EValidAxis.TIME,
                vl.EValidAxis.ALEATORY,
            ),
            vl.EpistemicOffsets(
                np.full(points, 0.1), np.full(points, 0.2), (vl.EValidAxis.POINT,)
            ),
        )
    return vl.ValidCase(observables, sims, experiments)


@pytest.mark.parametrize("workers", [2])
def test_parallel_metrics_match_serial(workers: int) -> None:
    test_case = create_parallel_case()
    metrics = [
        vl.MetricMAVM(),
        vl.MetricAVM(),
        vl.MetricKS(),
        vl.MetricCVM(),
    ]
    serial = vl.ValidEngine(metrics).run(test_case)
    parallel = vl.ValidEngine(
        metrics, opts=vl.ValidEngineOpts(workers=workers)
    ).run(test_case)
    assert parallel.observables is test_case.observables
    assert parallel.metric_configs.keys() == serial.metric_configs.keys()
    serial_mavm = serial.metric_configs[vl.EMetricType.MAVM]
    parallel_mavm = parallel.metric_configs[vl.EMetricType.MAVM]
    assert isinstance(serial_mavm, vl.MetricMAVM)
    assert isinstance(parallel_mavm, vl.MetricMAVM)
    assert (parallel_mavm.alpha, parallel_mavm.duplicates, parallel_mavm.tol) == (
        serial_mavm.alpha,
        serial_mavm.duplicates,
        serial_mavm.tol,
    )
    assert parallel_mavm is not serial_mavm
    for name in test_case.observables:
        for metric in metrics:
            expected = serial.metrics[name][metric.get_kind()]
            actual = parallel.metrics[name][metric.get_kind()]
            assert actual.axes == expected.axes
            assert actual.epis_offsets == expected.epis_offsets
            assert actual.metric_vals.keys() == expected.metric_vals.keys()
            for output in expected.metric_vals:
                np.testing.assert_allclose(
                    actual.metric_vals[output],
                    expected.metric_vals[output],
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
    assert np.isnan(test_case.sim["camera"].data[0, 0, 0, 0]).all()


class FailingMetric(vl.IValidMetric):
    def get_scale(self) -> vl.EMetricScale:
        return vl.EMetricScale.PHYSICAL

    def get_comparison(self) -> vl.EMetricCompare:
        return vl.EMetricCompare.DISTRIB

    def get_outputs(self) -> tuple[vl.EMetricOut, ...]:
        return (vl.EMetricOut.DISTANCE,)

    def get_kind(self) -> vl.EMetricType:
        return vl.EMetricType.AVM

    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> vl.MetricResult:
        return self.calc(sim, exp)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> vl.MetricResult:
        raise ValueError("Deliberate worker failure.")


def test_worker_failure_propagates_and_engine_can_run_again() -> None:
    test_case = create_parallel_case()
    engine = vl.ValidEngine([FailingMetric()], opts=vl.ValidEngineOpts(workers=2))
    with pytest.raises(ValueError, match="Deliberate worker failure"):
        engine.run(test_case)
    engine.metrics = (vl.MetricAVM(),)
    assert np.isfinite(
        engine.run(test_case)
        .metrics["camera"][vl.EMetricType.AVM]
        .metric_vals[vl.EMetricOut.DISTANCE]
    ).any()


@pytest.mark.parametrize("workers", [2])
def test_uneven_partitions_preserve_physical_offsets(workers: int) -> None:
    point_count, component_count, time_count = (5, 2, 3)
    physical = (
        np.arange(point_count)[:, None, None] * 10.0
        + np.arange(component_count)[None, :, None] * 2.0
        + np.arange(time_count)[None, None, :] * 0.5
    )
    sim_trans_shifts = np.array([0.0, 2.0])
    exp_trans_shifts = np.array([0.25, 0.75])
    samples = np.arange(3.0)
    model = physical[..., None, None] + sim_trans_shifts[:, None] + samples
    exp = physical[..., None, None] + exp_trans_shifts[:, None] + samples
    model_lower = np.arange(point_count)[:, None] * 0.1 + np.arange(time_count) * 0.05
    exp_lower = np.full(physical.shape, 0.3)
    observable = vl.ValidObservable(
        components=("x", "y"),
        geometry=vl.EValidGeometry.POINT,
        coords=np.zeros((point_count, 3)),
        times=np.arange(float(time_count)),
    )
    physical_axes = (
        vl.EValidAxis.POINT,
        vl.EValidAxis.COMPONENT,
        vl.EValidAxis.TIME,
    )
    test_case = vl.ValidCase(
        {"response": observable},
        {
            "response": vl.ValidArray(
                model,
                (
                    *physical_axes,
                    vl.EValidAxis.EPISTEMIC_TRANS,
                    vl.EValidAxis.ALEATORY,
                ),
                vl.EpistemicOffsets(
                    model_lower,
                    2 * model_lower,
                    (vl.EValidAxis.POINT, vl.EValidAxis.TIME),
                ),
            )
        },
        {
            "response": vl.ValidArray(
                exp,
                (
                    *physical_axes,
                    vl.EValidAxis.EPISTEMIC_NUIS,
                    vl.EValidAxis.ALEATORY,
                ),
                vl.EpistemicOffsets(exp_lower, 2 * exp_lower, physical_axes),
            )
        },
    )
    result = vl.ValidEngine(
        [vl.MetricAVM(), vl.MetricRMSE()],
        opts=vl.ValidEngineOpts(workers=workers),
    ).run(test_case)
    model_shifts = np.stack(
        (np.zeros_like(model_lower), -model_lower, 2 * model_lower), axis=-1
    )
    exp_shifts = np.stack(
        (np.zeros_like(exp_lower), -exp_lower, 2 * exp_lower), axis=-1
    )
    expected = np.abs(
        sim_trans_shifts[None, None, None, :, None, None, None]
        - exp_trans_shifts[None, None, None, None, :, None, None]
        + model_shifts[:, None, :, None, None, :, None]
        - exp_shifts[:, :, :, None, None, None, :]
    )
    for metric, output in (
        (vl.EMetricType.AVM, vl.EMetricOut.DISTANCE),
        (vl.EMetricType.RMSE, vl.EMetricOut.ERROR),
    ):
        np.testing.assert_allclose(
            result.metrics["response"][metric].metric_vals[output],
            expected,
            rtol=1e-13,
            atol=1e-12,
        )


def test_small_workload_with_worker_cap_matches_analytic_shift() -> None:
    observables = {
        "response": vl.ValidObservable(
            components=("response",),
            geometry=vl.EValidGeometry.POINT,
            coords=np.zeros((1, 3)),
        )
    }
    axes = (
        vl.EValidAxis.POINT,
        vl.EValidAxis.COMPONENT,
        vl.EValidAxis.ALEATORY,
    )
    sim_data = np.array([[[0.0, 1.0]]])
    test_case = vl.ValidCase(
        observables,
        {"response": vl.ValidArray(sim_data, axes)},
        {"response": vl.ValidArray(sim_data + 0.125, axes)},
    )
    metrics = [
        vl.MetricAVM(),
        vl.MetricRMSE(),
        vl.MetricRelErr(normalisation=vl.ERelErrNorm.CONSTANT, constant=1.0),
    ]
    for workers in (1, 4):
        result = vl.ValidEngine(
            metrics, opts=vl.ValidEngineOpts(workers=workers)
        ).run(test_case)
        for metric in metrics:
            assert (
                result.metrics["response"][metric.get_kind()]
                .metric_vals[metric.get_outputs()[0]]
                .item()
                == 0.125
            )


def create_scalar_case(observations: np.ndarray) -> vl.ValidCase:
    observables = {
        "scalar": vl.ValidObservable(
            components=("scalar",), geometry=vl.EValidGeometry.GLOBAL, coords=None
        )
    }
    quantity = vl.ValidArray(observations, (vl.EValidAxis.ALEATORY,))
    return vl.ValidCase(observables, {"scalar": quantity}, {"scalar": quantity})


def test_engine_snapshots_and_all_nan_configuration() -> None:
    metric = vl.MetricMAVM(alpha=0.1)
    metrics = [metric]
    options = vl.ValidEngineOpts(workers=1)
    engine = vl.ValidEngine(metrics, opts=options)
    metric.alpha = 0.2
    metrics.clear()
    options.workers = 0
    result = engine.run(create_scalar_case(np.array([1.0, 2.0])))
    engine.metrics[0].alpha = 0.3
    assert result.metric_configs[vl.EMetricType.MAVM].alpha == 0.1
    assert engine.opts.workers == 1
    engine.metrics[0].tol = -1.0
    with pytest.raises(ValueError, match="tolerance"):
        engine.run(create_scalar_case(np.array([np.nan])))


class MetricSampleMeans(vl.IValidMetric):
    """Exercise custom output ordering independently of built-in dispatch."""

    def get_scale(self) -> vl.EMetricScale:
        return vl.EMetricScale.PHYSICAL

    def get_comparison(self) -> vl.EMetricCompare:
        return vl.EMetricCompare.DISTRIB

    def get_kind(self) -> vl.EMetricType:
        return vl.EMetricType.AVM

    def get_outputs(self) -> tuple[vl.EMetricOut, ...]:
        return (vl.EMetricOut.ERROR, vl.EMetricOut.DISTANCE)

    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> vl.MetricResult:
        return self.calc(sim, exp)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> vl.MetricResult:
        mean = float(np.mean(sim))
        return vl.MultiMetricResult(self.get_kind(), (mean, mean + 1.0))


@pytest.mark.parametrize("workers", [1, 2])
def test_custom_multiple_outputs_preserve_order_axes_and_missing_cells(
    workers: int,
) -> None:
    observations = np.array([[1.0, 3.0], [np.nan, np.nan], [4.0, 6.0]])
    observable = vl.ValidObservable(
        components=("scalar",),
        geometry=vl.EValidGeometry.POINT,
        coords=np.zeros((3, 3)),
    )
    quantity = vl.ValidArray(
        observations, (vl.EValidAxis.POINT, vl.EValidAxis.ALEATORY)
    )
    test_case = vl.ValidCase(
        {"scalar": observable}, {"scalar": quantity}, {"scalar": quantity}
    )
    result = vl.ValidEngine(
        [MetricSampleMeans(), vl.MetricKS()],
        opts=vl.ValidEngineOpts(workers=workers),
    ).run(test_case)
    field = result.metrics["scalar"][vl.EMetricType.AVM]
    assert tuple(field.metric_vals) == (
        vl.EMetricOut.ERROR,
        vl.EMetricOut.DISTANCE,
    )
    for output, expected in (
        (vl.EMetricOut.ERROR, [2.0, np.nan, 5.0]),
        (vl.EMetricOut.DISTANCE, [3.0, np.nan, 6.0]),
    ):
        assert field.metric_vals[output].shape == (3, 1, 1)
        np.testing.assert_allclose(
            vl.extract_metric_values(field, output=output), expected
        )
        selected = vl.select_metric_output(field, output)
        assert np.shares_memory(selected, field.metric_vals[output])
    statistic = result.metrics["scalar"][vl.EMetricType.KS].metric_vals[
        vl.EMetricOut.STATISTIC
    ]
    np.testing.assert_allclose(statistic[:, 0, 0], [0.0, np.nan, 0.0])
    maximum, selection = result.find_extreme(
        "scalar", vl.EMetricType.AVM, output=vl.EMetricOut.ERROR
    )
    assert maximum == 5.0
    assert selection[vl.EValidResAxis.POINT] == 2


@dataclass(slots=True)
class MetricSampleDifference(vl.IValidMetric):
    """Custom implementation whose metadata deliberately differs from its KS tag."""

    comparison: vl.EMetricCompare
    scale: vl.EMetricScale = vl.EMetricScale.PHYSICAL

    def get_kind(self) -> vl.EMetricType:
        return vl.EMetricType.KS

    def get_scale(self) -> vl.EMetricScale:
        return self.scale

    def get_comparison(self) -> vl.EMetricCompare:
        return self.comparison

    def get_outputs(self) -> tuple[vl.EMetricOut, ...]:
        return (vl.EMetricOut.STATISTIC,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> vl.ScalarMetricResult:
        return self._calc_trusted(sim, exp)

    def _calc_trusted(self, sim: np.ndarray, exp: np.ndarray) -> vl.ScalarMetricResult:
        return vl.ScalarMetricResult(
            self.get_kind(), float(np.sum(sim) - np.sum(exp))
        )


def test_custom_paired_classification_uses_joint_missing_observation_mask() -> None:
    axes = (vl.EValidAxis.ALEATORY,)
    valid_case = vl.ValidCase(
        {
            "length": vl.ValidObservable(
                components=("length",),
                geometry=vl.EValidGeometry.GLOBAL,
                phys_units="m",
            )
        },
        {"length": vl.ValidArray(np.array([1.0, np.nan, 3.0]), axes)},
        {"length": vl.ValidArray(np.array([0.0, 20.0, np.nan]), axes)},
    )
    metric = MetricSampleDifference(vl.EMetricCompare.PAIRED)
    result = vl.ValidEngine([metric]).run(valid_case)
    field = result.metrics["length"][vl.EMetricType.KS]
    np.testing.assert_array_equal(
        field.metric_vals[vl.EMetricOut.STATISTIC], [[[1.0]]]
    )


@pytest.mark.parametrize(
    ("sim_count", "include_missing"), [(1, True), (3, False), (3, True)]
)
def test_batched_avm_matches_scalar_comparisons(
    sim_count: int, include_missing: bool
) -> None:
    tag = "temperature"
    simulation = np.arange(6, dtype=np.float32).reshape(3, 2, 1) + np.array(
        [0.4, -0.1, 0.2], dtype=np.float32
    )[:sim_count]
    experiment = np.arange(27.0).reshape(3, 3, 3)[..., ::-1] / 10.0
    if include_missing:
        simulation[1, 1, 0] = np.nan
        experiment[0, 0, 0] = np.nan
        experiment[2, 1] = np.nan
    axes = (
        vl.EValidAxis.POINT,
        vl.EValidAxis.EPISTEMIC_TRANS,
        vl.EValidAxis.ALEATORY,
    )
    test_case = vl.ValidCase(
        {
            tag: vl.ValidObservable(
                components=(tag,),
                geometry=vl.EValidGeometry.POINT,
                coords=np.zeros((3, 3)),
            )
        },
        {tag: vl.ValidArray(simulation, axes)},
        {tag: vl.ValidArray(experiment, axes)},
    )
    expected = np.full((3, 2, 3), np.nan)
    for point, sim_trans_idx, exp_trans_idx in np.ndindex(expected.shape):
        predicted_samples = simulation[point, sim_trans_idx]
        observations = experiment[point, exp_trans_idx]
        if np.any(np.isfinite(predicted_samples)) and np.any(np.isfinite(observations)):
            expected[point, sim_trans_idx, exp_trans_idx] = vl.calc_avm_1d(
                predicted_samples, observations
            )

    for workers in (1, 4):
        result = vl.ValidEngine(
            [vl.MetricAVM()], opts=vl.ValidEngineOpts(workers=workers)
        ).run(test_case)
        field = result.metrics[tag][vl.EMetricType.AVM]
        assert field.axes[-2:] == (vl.EValidResAxis.SIM_TRANS, vl.EValidResAxis.EXP_TRANS)
        np.testing.assert_allclose(
            field.metric_vals[vl.EMetricOut.DISTANCE][:, 0, 0],
            expected,
            rtol=0,
            atol=1e-14,
        )
