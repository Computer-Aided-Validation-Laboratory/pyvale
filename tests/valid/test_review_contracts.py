"""Analytic scientific contracts and public boundary regressions."""

import numpy as np
import pytest
from scipy import stats

import pyvale.valid as val
from pyvale.valid.cython.mavm import cyth_calc_mavm_1d, cyth_calc_mavm_pbox_1d
from pyvale.valid.metrics import ERelativeErrorNormalization as Normalization
from pyvale.valid.valtools import build_metric_selection


@pytest.mark.parametrize(
    "normalization,constant,expected",
    [
        (Normalization.EXPERIMENT_MAX, None, 0.25),
        (Normalization.SIMULATION_MAX, None, 0.2),
        (Normalization.CONSTANT, 2.0, 0.5),
    ],
)
def test_relative_error_normalization_parity(
    normalization: Normalization, constant: float | None, expected: float
) -> None:
    simulation = np.array([2.0, 5.0, np.nan])
    experiment = np.array([1.0, 4.0, 100.0])
    metric = val.MetricRelErr(normalization=normalization, constant=constant)
    summary = val.calc_deterministic_metrics_1d(
        simulation,
        experiment,
        normalization=normalization,
        constant=constant,
    )
    assert summary.relative_error == pytest.approx(expected)
    assert metric.calc(simulation, experiment).value == summary.relative_error


@pytest.mark.parametrize("reference", [0.0, 1e-15, -1e-15])
def test_relative_error_zero_and_near_zero(reference: float) -> None:
    simulation, experiment = np.array([1.0]), np.array([reference])
    standalone = val.calc_deterministic_metrics_1d(simulation, experiment)
    strategy = val.MetricRelErr().calc(simulation, experiment).value
    if reference == 0:
        assert np.isnan(standalone.relative_error) and np.isnan(strategy)
    else:
        assert standalone.relative_error == pytest.approx(
            abs(1 - reference) / abs(reference)
        )
        assert strategy == standalone.relative_error
    assert np.isnan(standalone.nmse)


@pytest.mark.parametrize(
    "normalization,constant",
    [
        ("experiment", None),
        (Normalization.CONSTANT, None),
        (Normalization.CONSTANT, 0.0),
        (Normalization.CONSTANT, -1.0),
        (Normalization.CONSTANT, np.inf),
        (Normalization.EXPERIMENT_MAX, 1.0),
    ],
)
def test_invalid_relative_error_configuration(
    normalization: Normalization | str, constant: float | None
) -> None:
    with pytest.raises((ValueError, TypeError)):
        val.MetricRelErr(normalization=normalization, constant=constant)


@pytest.mark.parametrize(
    "observation,expected", [(-1.0, 0.5), (0.0, 0.25), (0.5, 0.25), (1.0, 0.5)]
)
def test_u_pooling_continuous_uniform_oracle(
    observation: float, expected: float
) -> None:
    cdf = val.ValECDF(np.array([0.0, 1.0]), np.array([0.5, 1.0]))
    # W1(delta_u, U) = (u**2 + (1-u)**2)/2, with right-continuous ECDF.
    assert val.calc_u_pooling_1d([cdf], np.array([observation])) == pytest.approx(
        expected
    )


def test_u_pooling_multiple_sample_oracle() -> None:
    cdf = val.ValECDF(np.array([0.0]), np.array([1.0]))
    # Half mass at either endpoint: integrate p on [0,.5], 1-p on [.5,1].
    assert val.calc_u_pooling_1d([cdf, cdf], np.array([-1.0, 1.0])) == pytest.approx(
        0.25
    )
    with pytest.raises(ValueError):
        val.calc_u_pooling_1d([], np.array([]))
    with pytest.raises(ValueError):
        val.calc_u_pooling_1d([], np.array([1.0]))


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
    with pytest.raises(ValueError):
        val.ValECDF(np.array(support), np.array(probabilities))


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
        val.MetricMAVM(**configuration)
    for calculate in (val.calc_mavm_1d, cyth_calc_mavm_1d):
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
    first = val.calc_mavm_1d(model, unique, duplicates=val.EMAVMDuplicates.INCLUDE)
    second = val.calc_mavm_1d(model, repeated, duplicates=val.EMAVMDuplicates.INCLUDE)
    expected_half_width = stats.t.ppf(0.95, 1) / np.sqrt(2)
    np.testing.assert_allclose(first.exp_conf_upper - unique, expected_half_width)
    np.testing.assert_array_equal(first.exp_conf_upper, second.exp_conf_upper)
    np.testing.assert_allclose(second.exp_probs, [0.75, 1.0])
    assert second.d_plus == pytest.approx(first.d_plus - 0.5)
    assert second.d_minus == pytest.approx(first.d_minus + 0.5)


def create_case(observations: np.ndarray) -> val.ValCase:
    domain = val.ValDomain(
        {
            "scalar": val.ValObservable(
                val.ValGeometry(val.EValGeometry.GLOBAL, None),
                val.EValOperator.MEAN,
                ("scalar",),
            )
        }
    )
    quantity = val.ValArray(observations, (val.EValAxis.ALEATORY,))
    return val.ValCase(domain, {"scalar": quantity}, {"scalar": quantity})


@pytest.mark.parametrize(
    "observations",
    [np.array([1j]), np.array([True]), np.array([1], dtype=object), np.array([np.inf])],
)
def test_unsupported_observations(observations: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        create_case(observations)
    for calculate in (
        val.calc_avm_1d,
        val.calc_ks_1d,
        val.calc_cvm_1d,
        val.calc_mavm_1d,
        val.calc_deterministic_metrics_1d,
    ):
        with pytest.raises((TypeError, ValueError)):
            calculate(observations, observations)
    for metric in (val.MetricRMSE(), val.MetricRelErr()):
        with pytest.raises((TypeError, ValueError)):
            metric.calc(observations, observations)


def test_engine_snapshots_and_all_nan_configuration() -> None:
    metric = val.MetricMAVM(alpha=0.1)
    metrics = [metric]
    options = val.ValEngineOpts(workers=1)
    engine = val.ValidationEngine(metrics, opts=options)
    metric.alpha = 0.2
    metrics.clear()
    options.workers = 0
    result = engine.run(create_case(np.array([1.0, 2.0])))
    engine.metrics[0].alpha = 0.3
    assert result.metric_configs[val.EMetricType.MAVM].alpha == 0.1
    assert engine.opts.workers == 1
    engine.metrics[0].tol = -1.0
    with pytest.raises(ValueError, match="tolerance"):
        engine.run(create_case(np.array([np.nan])))


def test_offset_and_borrowing_contract() -> None:
    observations = np.array([1.0, 2.0])
    quantity = val.ValArray(
        observations, (val.EValAxis.ALEATORY,), val.EpistemicOffsets(1.0, 1.0)
    )
    assert np.shares_memory(quantity.select_aleatory(0, 0, 0), observations)
    assert not np.shares_memory(
        quantity.select_aleatory(0, 0, 0, offset=val.EEpisOffset.LOWER), observations
    )
    for invalid in (0, 0.0, True):
        with pytest.raises(TypeError):
            quantity.select_aleatory(0, 0, 0, offset=invalid)


def test_metric_field_structure_and_selectors() -> None:
    axes = (val.EValResAxis.POINT, val.EValResAxis.COMPONENT, val.EValResAxis.TIME)
    with pytest.raises(ValueError):
        val.ValMetricField(axes, np.zeros((2, 1)))
    with pytest.raises(ValueError):
        val.ValMetricField(axes, np.zeros((2, 1, 1)), d_plus=np.zeros((2, 1, 1)))
    field = val.ValMetricField(axes, np.zeros((2, 1, 1)))
    for index in (-1, 2, True, 0.0):
        with pytest.raises((TypeError, IndexError)):
            build_metric_selection(field, {val.EValResAxis.POINT: index})
    with pytest.raises(ValueError):
        build_metric_selection(field, {val.EValResAxis.SIM_NUIS: 0})


@pytest.mark.parametrize(
    "coordinates", [np.array([[np.nan, 0.0, 0.0]]), np.array([[1j, 0.0, 0.0]])]
)
def test_geometry_rejects_nonfinite_and_complex(coordinates: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        val.ValGeometry(val.EValGeometry.POINT, coordinates)


def test_unsigned_support_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="increase"):
        val.ValECDF(np.array([1, 0], dtype=np.uint64), np.array([0.5, 1.0]))


@pytest.mark.parametrize("selector", [-1, 1, True, 0.0])
def test_plot_selectors_fail_without_creating_figures(selector: float) -> None:
    import matplotlib.pyplot as plt

    from pyvale.valid import valplot

    case = create_case(np.array([0.0, 1.0]))
    result = val.ValidationEngine([val.MetricAVM()]).run(case)
    figures = plt.get_fignums()
    with pytest.raises((TypeError, IndexError)):
        valplot.plot_val_metric_distribution(
            result,
            "scalar",
            val.EMetricType.AVM,
            selectors={val.EValResAxis.POINT: selector},
        )
    with pytest.raises((TypeError, IndexError)):
        valplot.plot_val_cdfs_points(case, result, "scalar", point_idxs=(selector,))
    assert plt.get_fignums() == figures


def test_unknown_histogram_axis_fails_without_creating_figure() -> None:
    import matplotlib.pyplot as plt

    from pyvale.valid import valplot

    result = val.ValidationEngine([val.MetricAVM()]).run(create_case(np.array([1.0])))
    figures = plt.get_fignums()
    with pytest.raises(ValueError):
        valplot.plot_val_metric_distribution(
            result,
            "scalar",
            val.EMetricType.AVM,
            selectors={val.EValResAxis.SIM_TRANS: 0},
        )
    assert plt.get_fignums() == figures
