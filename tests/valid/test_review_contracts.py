"""Analytic scientific contracts and public boundary regressions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

from pyvale import valid
from pyvale.valid import validplot
from pyvale.valid.cython.mavm import cyth_calc_mavm_1d, cyth_calc_mavm_pbox_1d
from pyvale.valid.metrics import ERelErrNorm as Normalisation
from pyvale.valid.validtools import build_metric_selection


@pytest.mark.parametrize(
    "normalisation,constant,expected",
    [
        (Normalisation.EXP_MAX, None, 0.25),
        (Normalisation.SIM_MAX, None, 0.2),
        (Normalisation.CONSTANT, 2.0, 0.5),
    ],
)
def test_relative_error_normalisation_parity(
    normalisation: Normalisation, constant: float | None, expected: float
) -> None:
    sim_data = np.array([2.0, 5.0, np.nan])
    exp = np.array([1.0, 4.0, 100.0])
    metric = valid.MetricRelErr(normalisation=normalisation, constant=constant)
    summary = valid.calc_deterministic_metrics_1d(
        sim_data,
        exp,
        normalisation=normalisation,
        constant=constant,
    )
    assert summary.relative_error == pytest.approx(expected)
    assert metric.calc(sim_data, exp).value == summary.relative_error


@pytest.mark.parametrize("reference", [0.0, 1e-15, -1e-15])
def test_relative_error_zero_and_near_zero(reference: float) -> None:
    sim_data, exp = np.array([1.0]), np.array([reference])
    standalone = valid.calc_deterministic_metrics_1d(sim_data, exp)
    strategy = valid.MetricRelErr().calc(sim_data, exp).value
    if reference == 0:
        assert np.isnan(standalone.relative_error) and np.isnan(strategy)
    else:
        assert standalone.relative_error == pytest.approx(
            abs(1 - reference) / abs(reference)
        )
        assert strategy == standalone.relative_error
    assert np.isnan(standalone.nmse)


@pytest.mark.parametrize(
    "normalisation,constant",
    [
        ("experiment", None),
        (Normalisation.CONSTANT, None),
        (Normalisation.CONSTANT, 0.0),
        (Normalisation.CONSTANT, -1.0),
        (Normalisation.CONSTANT, np.inf),
        (Normalisation.EXP_MAX, 1.0),
    ],
)
def test_invalid_relative_error_configuration(
    normalisation: Normalisation | str, constant: float | None
) -> None:
    with pytest.raises((ValueError, TypeError)):
        valid.MetricRelErr(normalisation=normalisation, constant=constant)


@pytest.mark.parametrize(
    "observation,expected", [(-1.0, 0.5), (0.0, 0.25), (0.5, 0.25), (1.0, 0.5)]
)
def test_u_pooling_continuous_uniform_oracle(
    observation: float, expected: float
) -> None:
    cdf = valid.ValidECDF(np.array([0.0, 1.0]), np.array([0.5, 1.0]))
    # W1(delta_u, U) = (u**2 + (1-u)**2)/2, with right-continuous ECDF.
    assert valid.calc_u_pooling_1d([cdf], np.array([observation])) == pytest.approx(
        expected
    )


def test_u_pooling_multiple_sample_oracle() -> None:
    cdf = valid.ValidECDF(np.array([0.0]), np.array([1.0]))
    # Half mass at either endpoint: integrate p on [0,.5], 1-p on [.5,1].
    assert valid.calc_u_pooling_1d([cdf, cdf], np.array([-1.0, 1.0])) == pytest.approx(
        0.25
    )
    with pytest.raises(ValueError):
        valid.calc_u_pooling_1d([], np.array([]))
    with pytest.raises(ValueError):
        valid.calc_u_pooling_1d([], np.array([1.0]))


@pytest.mark.parametrize(
    "support,probabilities",
    [
        ([], []),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [0.5, 1.0]),
        ([0.0, 0.0], [0.5, 1.0]),
        ([0.0, np.inf], [0.5, 1.0]),
        ([0.0, 1.0], [0.5, 0.9]),
        ([0.0, 1.0], [-0.1, 1.0]),
        ([0.0, 1.0, 2.0], [0.8, 0.5, 1.0]),
    ],
)
def test_invalid_ecdf(support: list[float], probabilities: list[float]) -> None:
    cdf = valid.ValidECDF(np.array(support), np.array(probabilities))
    with pytest.raises(ValueError):
        valid.evaluate_ecdf_quantiles(cdf, np.array([0.5]))
    with pytest.raises(ValueError):
        valid.calc_u_pooling_1d([cdf], np.array([0.5]))


@pytest.mark.parametrize(
    "configuration",
    [
        {"duplicates": "include"},
        {"tol": -1.0},
        {"tol": np.nan},
        {"tol": np.inf},
        {"alpha": np.nan},
    ],
)
def test_mavm_configuration_rejected_before_observations(
    configuration: dict[str, float | str],
) -> None:
    message = (
        "duplicates"
        if "duplicates" in configuration
        else "tolerance"
        if "tol" in configuration
        else "alpha"
    )
    with pytest.raises((ValueError, TypeError), match=message):
        valid.MetricMAVM(**configuration)
    for calculate in (valid.calc_mavm_1d, cyth_calc_mavm_1d):
        with pytest.raises((ValueError, TypeError), match=message):
            calculate(np.array([np.nan]), np.array([np.nan]), **configuration)
    with pytest.raises((ValueError, TypeError), match=message):
        cyth_calc_mavm_pbox_1d(
            np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), **configuration
        )


def test_mavm_frequency_sensitive_legacy_confidence() -> None:
    model = np.array([0.0, 2.0])
    unique = np.array([0.0, 2.0])
    repeated = np.array([0.0, 0.0, 0.0, 2.0])
    first = valid.calc_mavm_1d(model, unique, duplicates=valid.EMAVMDuplicates.INCLUDE)
    second = valid.calc_mavm_1d(
        model, repeated, duplicates=valid.EMAVMDuplicates.INCLUDE
    )
    expected_half_width = stats.t.ppf(0.95, 1) / np.sqrt(2)
    np.testing.assert_allclose(first.exp_conf_upper - unique, expected_half_width)
    np.testing.assert_array_equal(first.exp_conf_upper, second.exp_conf_upper)
    np.testing.assert_allclose(second.exp_probs, [0.75, 1.0])
    assert second.d_plus == pytest.approx(first.d_plus - 0.5)
    assert second.d_minus == pytest.approx(first.d_minus + 0.5)


def create_case(observations: np.ndarray) -> valid.ValidCase:
    observables = {
        "scalar": valid.ValidObservable(
            components=("scalar",), geometry=valid.EValidGeometry.GLOBAL, coords=None
        )
    }
    quantity = valid.ValidArray(observations, (valid.EValidAxis.ALEATORY,))
    return valid.ValidCase(observables, {"scalar": quantity}, {"scalar": quantity})


@pytest.mark.parametrize(
    "observations",
    [np.array([1j]), np.array([True]), np.array([1], dtype=object), np.array([np.inf])],
)
def test_unsupported_observations(observations: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        create_case(observations)
    for calculate in (
        valid.calc_avm_1d,
        valid.calc_ks_1d,
        valid.calc_cvm_1d,
        valid.calc_mavm_1d,
        valid.calc_deterministic_metrics_1d,
    ):
        with pytest.raises((TypeError, ValueError)):
            calculate(observations, observations)
    for metric in (valid.MetricRMSE(), valid.MetricRelErr()):
        with pytest.raises((TypeError, ValueError)):
            metric.calc(observations, observations)


def test_engine_snapshots_and_all_nan_configuration() -> None:
    metric = valid.MetricMAVM(alpha=0.1)
    metrics = [metric]
    options = valid.ValidEngineOpts(workers=1)
    engine = valid.ValidEngine(metrics, opts=options)
    metric.alpha = 0.2
    metrics.clear()
    options.workers = 0
    result = engine.run(create_case(np.array([1.0, 2.0])))
    engine.metrics[0].alpha = 0.3
    assert result.metric_configs[valid.EMetricType.MAVM].alpha == 0.1
    assert engine.opts.workers == 1
    engine.metrics[0].tol = -1.0
    with pytest.raises(ValueError, match="tolerance"):
        engine.run(create_case(np.array([np.nan])))


def test_offset_and_borrowing_contract() -> None:
    observations = np.array([1.0, 2.0])
    quantity = valid.ValidArray(
        observations, (valid.EValidAxis.ALEATORY,), valid.EpistemicOffsets(1.0, 1.0)
    )
    assert np.shares_memory(quantity.select_aleatory(0, 0, 0), observations)
    assert not np.shares_memory(
        quantity.select_aleatory(0, 0, 0, offset=valid.EEpisOffset.LOWER), observations
    )
    for invalid in (0, 0.0, True):
        with pytest.raises(TypeError):
            quantity.select_aleatory(0, 0, 0, offset=invalid)


def test_metric_field_selectors() -> None:
    axes = (
        valid.EValidResAxis.POINT,
        valid.EValidResAxis.COMPONENT,
        valid.EValidResAxis.TIME,
    )
    field = valid.ValidMetricField(
        axes, {valid.EMetricOutput.ERROR: np.zeros((2, 1, 1))}
    )
    for index in (-1, 2, True, 0.0):
        with pytest.raises((TypeError, IndexError)):
            build_metric_selection(field, {valid.EValidResAxis.POINT: index})
    with pytest.raises(ValueError):
        build_metric_selection(field, {valid.EValidResAxis.SIM_NUIS: 0})


@pytest.mark.parametrize(
    "coordinates", [np.array([[np.nan, 0.0, 0.0]]), np.array([[1j, 0.0, 0.0]])]
)
def test_geometry_rejects_nonfinite_and_complex(coordinates: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        valid.ValidObservable(
            components=("scalar",),
            coords=coordinates,
            geometry=valid.EValidGeometry.POINT,
        )


def test_unsigned_support_must_be_ordered() -> None:
    cdf = valid.ValidECDF(np.array([1, 0], dtype=np.uint64), np.array([0.5, 1.0]))
    with pytest.raises(ValueError, match="increase"):
        valid.evaluate_ecdf_quantiles(cdf, np.array([0.5]))
    with pytest.raises(ValueError, match="increase"):
        valid.calc_u_pooling_1d([cdf], np.array([0.5]))


@pytest.mark.parametrize("selector", [-1, 1, True, 0.0])
def test_plot_selectors_fail_without_creating_figures(selector: float) -> None:

    case = create_case(np.array([0.0, 1.0]))
    result = valid.ValidEngine([valid.MetricAVM()]).run(case)
    figures = plt.get_fignums()
    with pytest.raises((TypeError, IndexError)):
        validplot.plot_valid_metric_distribution(
            result,
            "scalar",
            valid.EMetricType.AVM,
            selectors={valid.EValidResAxis.POINT: selector},
            output=valid.EMetricOutput.DISTANCE,
        )
    with pytest.raises((TypeError, IndexError)):
        validplot.plot_valid_cdfs_points(case, result, "scalar", point_idxs=(selector,))
    assert plt.get_fignums() == figures


def test_unknown_histogram_axis_fails_without_creating_figure() -> None:

    result = valid.ValidEngine([valid.MetricAVM()]).run(create_case(np.array([1.0])))
    figures = plt.get_fignums()
    with pytest.raises(ValueError):
        validplot.plot_valid_metric_distribution(
            result,
            "scalar",
            valid.EMetricType.AVM,
            selectors={valid.EValidResAxis.SIM_TRANS: 0},
            output=valid.EMetricOutput.DISTANCE,
        )
    assert plt.get_fignums() == figures


class MetricSampleMeans(valid.IValidMetric):
    """Exercise custom output ordering independently of built-in dispatch."""

    def get_kind(self) -> valid.EMetricType:
        return valid.EMetricType.AVM

    def get_outputs(self) -> tuple[valid.EMetricOutput, ...]:
        return (valid.EMetricOutput.ERROR, valid.EMetricOutput.DISTANCE)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> valid.MetricResult:
        return valid.ScalarMetricResult(self.get_kind(), float(np.mean(sim)))

    def extract_outputs(self, result: valid.MetricResult) -> tuple[float, ...]:
        assert isinstance(result, valid.ScalarMetricResult)
        return result.value, result.value + 1.0


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


@pytest.mark.parametrize(
    ("geometry", "coords"),
    [
        (valid.EValidGeometry.GLOBAL, np.zeros((1, 3))),
        (valid.EValidGeometry.POINT, None),
        (valid.EValidGeometry.POINT, np.zeros(3)),
        (valid.EValidGeometry.POINT, np.zeros((1, 2))),
        (valid.EValidGeometry.POINT, np.zeros((0, 3))),
    ],
)
def test_observable_rejects_incompatible_geometry_coordinates(
    geometry: valid.EValidGeometry, coords: np.ndarray | None
) -> None:
    with pytest.raises(ValueError):
        valid.ValidObservable(components=("scalar",), geometry=geometry, coords=coords)


@pytest.mark.parametrize(
    ("metric", "output"),
    [
        (valid.MetricAVM(), valid.EMetricOutput.DISTANCE),
        (valid.MetricKS(), valid.EMetricOutput.STATISTIC),
        (valid.MetricCVM(), valid.EMetricOutput.STATISTIC),
        (valid.MetricRMSE(), valid.EMetricOutput.ERROR),
        (valid.MetricRelErr(), valid.EMetricOutput.ERROR),
    ],
)
def test_single_output_defaults_match_explicit_selection(
    metric: valid.IValidMetric, output: valid.EMetricOutput
) -> None:
    result = valid.ValidEngine([metric]).run(create_case(np.array([1.0, 2.0, 3.0])))
    field = result.metrics["scalar"][metric.get_kind()]
    assert tuple(field.outputs) == (output,)
    np.testing.assert_array_equal(
        valid.extract_metric_values(field),
        valid.extract_metric_values(field, output=output),
    )
    assert np.shares_memory(valid.select_metric_output(field), field.outputs[output])
    assert result.find_extreme("scalar", metric.get_kind()) == result.find_extreme(
        "scalar", metric.get_kind(), output=output
    )


@pytest.mark.parametrize("metric", [valid.MetricMAVM(), MetricSampleMeans()])
def test_multiple_outputs_require_explicit_selection(
    metric: valid.IValidMetric,
) -> None:

    result = valid.ValidEngine([metric]).run(create_case(np.array([1.0, 2.0, 3.0])))
    field = result.metrics["scalar"][metric.get_kind()]
    with pytest.raises(ValueError, match="output explicitly"):
        valid.select_metric_output(field)
    with pytest.raises(ValueError, match="output explicitly"):
        valid.extract_metric_values(field)
    with pytest.raises(ValueError, match="output explicitly"):
        result.find_extreme("scalar", metric.get_kind())
    figures = plt.get_fignums()
    with pytest.raises(ValueError, match="output explicitly"):
        validplot.plot_valid_metric_distribution(result, "scalar", metric.get_kind())
    assert plt.get_fignums() == figures


class MetricInvalidOutputs(valid.IValidMetric):
    def get_kind(self) -> valid.EMetricType:
        return valid.EMetricType.AVM

    def get_outputs(self) -> tuple[valid.EMetricOutput, ...]:
        return (valid.EMetricOutput.DISTANCE, valid.EMetricOutput.ERROR)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> valid.MetricResult:
        return valid.ScalarMetricResult(self.get_kind(), 1.0)

    def extract_outputs(self, result: valid.MetricResult) -> tuple[float, ...]:
        return (1.0,)


@pytest.mark.parametrize("workers", [1, 2])
def test_custom_metric_output_count_is_checked(workers: int) -> None:
    observations = valid.ValidArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        (valid.EValidAxis.EPISTEMIC_TRANS, valid.EValidAxis.ALEATORY),
    )
    case = valid.ValidCase(
        create_case(np.array([1.0, 2.0])).observables,
        {"scalar": observations},
        {"scalar": observations},
    )
    with pytest.raises(ValueError, match="Calculated outputs must match"):
        valid.ValidEngine(
            [MetricInvalidOutputs()], opts=valid.ValidEngineOpts(workers=workers)
        ).run(case)
