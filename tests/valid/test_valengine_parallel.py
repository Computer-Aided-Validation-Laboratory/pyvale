"""Serial and spawned shared-memory metric evaluation parity."""

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
            val.ValGeometry(val.EValGeometry.POINT, np.zeros((points, 3))),
            val.EValOperator.SAMPLE, ("temperature",), np.array([0.0, 1.0]),
        )
        # Strided input exercises the shared-memory copy's layout handling.
        samples = generator.normal(size=(points, 1, 2, 2, 20))[..., ::2]
        samples[0, 0, 0, 0] = np.nan
        simulations[name] = val.ValArray(
            samples,
            (val.EValAxis.POINT, val.EValAxis.COMPONENT, val.EValAxis.TIME,
             val.EValAxis.EPISTEMIC_TRANS, val.EValAxis.ALEATORY),
            val.EpistemicOffsets(0.05, 0.1),
        )
        experiments[name] = val.ValArray(
            generator.normal(size=(points, 1, 2, 10)),
            (val.EValAxis.POINT, val.EValAxis.COMPONENT, val.EValAxis.TIME,
             val.EValAxis.ALEATORY),
            val.EpistemicOffsets(
                np.full(points, 0.1), np.full(points, 0.2),
                (val.EValAxis.POINT,),
            ),
        )
    return val.ValCase(val.ValDomain(observables), simulations, experiments)


@pytest.mark.parametrize("chunk_size", [None, 5])
def test_spawned_metrics_match_serial(chunk_size: int | None) -> None:
    case = create_case()
    metrics = [val.MetricMAVM(), val.MetricAVM(), val.MetricKS(), val.MetricCVM()]
    serial = val.ValidationEngine(metrics).run(case)
    parallel = val.ValidationEngine(
        metrics, opts=val.ValEngineOpts(workers=2, chunk_size=chunk_size)
    ).run(case)
    assert parallel.domain is case.domain
    assert parallel.metric_configs == serial.metric_configs
    for name in case.domain.observables:
        for metric in metrics:
            expected = serial.get(name, metric.get_kind())
            actual = parallel.get(name, metric.get_kind())
            assert actual.axes == expected.axes
            assert actual.scenarios == expected.scenarios
            for attribute in ("metric", "d_plus", "d_minus"):
                expected_array = getattr(expected, attribute)
                actual_array = getattr(actual, attribute)
                if expected_array is None:
                    assert actual_array is None
                else:
                    np.testing.assert_allclose(
                        actual_array, expected_array, rtol=0, atol=0, equal_nan=True
                    )
    assert np.isnan(case.sim["camera"].data[0, 0, 0, 0]).all()


def test_spawned_failure_propagates_and_engine_can_run_again() -> None:
    case = create_case()
    engine = val.ValidationEngine([val.MetricAVM()], opts=val.ValEngineOpts(workers=2))
    case.sim["camera"].data[0, 0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        engine.run(case)
    case.sim["camera"].data[0, 0, 0, 0, 0] = 0.0
    assert np.isfinite(engine.run(case).get("camera", val.EMetricType.AVM).metric).any()


@pytest.mark.parametrize("name", ["workers", "chunk_size"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_options(name: str, value: int | float) -> None:
    with pytest.raises((TypeError, ValueError)):
        val.ValEngineOpts(**{name: value})
