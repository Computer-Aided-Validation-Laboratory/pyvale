"""Validation input errors behaviour."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pytest
from plate_fixture import load_plate_case

import pyvale.dataio as io
import pyvale.sensorsim as sens
import pyvale.validation as vl
from pyvale.dataio.simdata import SimData
from pyvale.validation import validplot, validvis
from pyvale.validation.constants import MAVM_DUPLICATE_TOLERANCE
from pyvale.validation.metrics import (
    EMAVMDuplicates,
    calc_mavm_1d,
    calc_mavm_pbox_1d,
)
from pyvale.validation.metrics import ERelErrNorm as Normalisation
from pyvale.validation.validtools import _build_metric_selection


def test_invalid_observations_fail_when_case_is_initialized() -> None:
    test_case = create_scalar_case(np.array([1.0, 2.0]))
    test_case.sim["scalar"].data[0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        vl.ValidCase(test_case.observables, test_case.sim, test_case.exp)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_options(value: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        vl.ValidEngineOpts(workers=value)


@pytest.mark.parametrize(
    "probabilities",
    [np.array([-0.1]), np.array([1.1]), np.array([np.nan]), np.ones((2, 2))],
)
def test_invalid_quantile_probabilities_fail(probabilities: np.ndarray) -> None:
    with pytest.raises(ValueError):
        vl.evaluate_ecdf_quantiles(
            vl.calc_ecdf(np.array([1.0, 2.0])), probabilities
        )


def test_missing_diagnostic_inputs_fail_before_plotting() -> None:
    with pytest.raises(ValueError):
        vl.select_ecdf_support_cases(())
    quantity = vl.ValidArray(np.array([1.0, 2.0]), (vl.EValidAxis.ALEATORY,))
    curves = vl.calc_ecdf_view(quantity)
    with pytest.raises(ValueError, match="bound pairs"):
        validplot.plot_valid_mavm_diagnostic(
            curves, view=vl.EValidMAVMDiagnostic.CORRECTION
        )


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
        vl.MetricRelErr(normalisation=normalisation, constant=constant)


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
    cdf = vl.ECDF(np.array(support), np.array(probabilities))
    with pytest.raises(ValueError):
        vl.evaluate_ecdf_quantiles(cdf, np.array([0.5]))


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
        vl.MetricMAVM(**configuration)
    with pytest.raises((ValueError, TypeError), match=message):
        vl.calc_mavm_1d(np.array([np.nan]), np.array([np.nan]), **configuration)
    with pytest.raises((ValueError, TypeError), match=message):
        vl.calc_mavm_pbox_1d(
            np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), **configuration
        )


def create_scalar_case(observations: np.ndarray) -> vl.ValidCase:
    observables = {
        "scalar": vl.ValidObservable(
            components=("scalar",), geometry=vl.EValidGeometry.GLOBAL, coords=None
        )
    }
    quantity = vl.ValidArray(observations, (vl.EValidAxis.ALEATORY,))
    return vl.ValidCase(observables, {"scalar": quantity}, {"scalar": quantity})


@pytest.mark.parametrize(
    "observations",
    [np.array([1j]), np.array([True]), np.array([1], dtype=object), np.array([np.inf])],
)
def test_unsupported_observations(observations: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        create_scalar_case(observations)
    for calculate in (
        vl.calc_avm_1d,
        vl.calc_ks_1d,
        vl.calc_cvm_1d,
        vl.calc_mavm_1d,
        vl.calc_deterministic_metrics_1d,
    ):
        with pytest.raises((TypeError, ValueError)):
            calculate(observations, observations)
    for metric in (vl.MetricRMSE(), vl.MetricRelErr()):
        with pytest.raises((TypeError, ValueError)):
            metric.calc(observations, observations)


@pytest.mark.parametrize(
    "coordinates", [np.array([[np.nan, 0.0, 0.0]]), np.array([[1j, 0.0, 0.0]])]
)
def test_geometry_rejects_nonfinite_and_complex(coordinates: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        vl.ValidObservable(
            components=("scalar",),
            coords=coordinates,
            geometry=vl.EValidGeometry.POINT,
        )


def test_unsigned_support_must_be_ordered() -> None:
    cdf = vl.ECDF(np.array([1, 0], dtype=np.uint64), np.array([0.5, 1.0]))
    with pytest.raises(ValueError, match="increase"):
        vl.evaluate_ecdf_quantiles(cdf, np.array([0.5]))


@pytest.mark.parametrize("selector", [-1, 1, True, 0.0])
def test_plot_selectors_fail_without_creating_figures(selector: float) -> None:
    test_case = create_scalar_case(np.array([0.0, 1.0]))
    result = vl.ValidEngine([vl.MetricAVM()]).run(test_case)
    figures = plt.get_fignums()
    with pytest.raises((TypeError, IndexError)):
        validplot.plot_metric_dist(
            result,
            "scalar",
            vl.EMetricType.AVM,
            selectors={vl.EValidResAxis.POINT: selector},
            output=vl.EMetricOut.DISTANCE,
        )
    with pytest.raises((TypeError, IndexError)):
        validplot.plot_sim_exp_cdfs(test_case, "scalar", point_idxs=(selector,))
    assert plt.get_fignums() == figures


def test_unknown_histogram_axis_fails_without_creating_figure() -> None:
    result = vl.ValidEngine([vl.MetricAVM()]).run(
        create_scalar_case(np.array([1.0]))
    )
    figures = plt.get_fignums()
    with pytest.raises(ValueError):
        validplot.plot_metric_dist(
            result,
            "scalar",
            vl.EMetricType.AVM,
            selectors={vl.EValidResAxis.SIM_TRANS: 0},
            output=vl.EMetricOut.DISTANCE,
        )
    assert plt.get_fignums() == figures


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

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> vl.MetricResult:
        mean = float(np.mean(sim))
        return vl.MultiMetricResult(self.get_kind(), (mean, mean + 1.0))


@pytest.mark.parametrize(
    ("geometry", "coords"),
    [
        (vl.EValidGeometry.GLOBAL, np.zeros((1, 3))),
        (vl.EValidGeometry.SURFACE, None),
        (vl.EValidGeometry.POINT, np.zeros(3)),
        (vl.EValidGeometry.POINT, np.zeros((1, 2))),
        (vl.EValidGeometry.POINT, np.zeros((0, 3))),
    ],
)
def test_observable_rejects_incompatible_geometry_coordinates(
    geometry: vl.EValidGeometry, coords: np.ndarray | None
) -> None:
    with pytest.raises(ValueError):
        vl.ValidObservable(components=("scalar",), geometry=geometry, coords=coords)


@pytest.mark.parametrize("metric", [vl.MetricMAVM(), MetricSampleMeans()])
def test_multiple_outputs_require_explicit_selection(
    metric: vl.IValidMetric,
) -> None:
    result = vl.ValidEngine([metric]).run(
        create_scalar_case(np.array([1.0, 2.0, 3.0]))
    )
    field = result.metrics["scalar"][metric.get_kind()]
    with pytest.raises(ValueError, match="output explicitly"):
        vl.select_metric_output(field)
    with pytest.raises(ValueError, match="output explicitly"):
        vl.extract_metric_values(field)
    with pytest.raises(ValueError, match="output explicitly"):
        result.find_extreme("scalar", metric.get_kind())
    figures = plt.get_fignums()
    with pytest.raises(ValueError, match="output explicitly"):
        validplot.plot_metric_dist(result, "scalar", metric.get_kind())
    assert plt.get_fignums() == figures


class MetricInvalidOutputs(vl.IValidMetric):
    def get_scale(self) -> vl.EMetricScale:
        return vl.EMetricScale.PHYSICAL

    def get_comparison(self) -> vl.EMetricCompare:
        return vl.EMetricCompare.DISTRIB

    def get_kind(self) -> vl.EMetricType:
        return vl.EMetricType.AVM

    def get_outputs(self) -> tuple[vl.EMetricOut, ...]:
        return (vl.EMetricOut.DISTANCE, vl.EMetricOut.ERROR)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> vl.MetricResult:
        return vl.ScalarMetricResult(self.get_kind(), 1.0)


@pytest.mark.parametrize("workers", [1, 2])
def test_custom_metric_output_count_is_checked(workers: int) -> None:
    observations = vl.ValidArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        (vl.EValidAxis.EPISTEMIC_TRANS, vl.EValidAxis.ALEATORY),
    )
    test_case = vl.ValidCase(
        create_scalar_case(np.array([1.0, 2.0])).observables,
        {"scalar": observations},
        {"scalar": observations},
    )
    with pytest.raises(ValueError, match="Calculated outputs must match"):
        vl.ValidEngine(
            [MetricInvalidOutputs()],
            opts=vl.ValidEngineOpts(workers=workers),
        ).run(test_case)


def create_ecdf_case(sim: np.ndarray, exp: np.ndarray) -> vl.ValidCase:
    sim_axes = (vl.EValidAxis.ALEATORY,)
    if sim.ndim == 2:
        sim_axes = (vl.EValidAxis.EPISTEMIC_TRANS, vl.EValidAxis.ALEATORY)
    exp_axes = (vl.EValidAxis.ALEATORY,)
    if exp.ndim == 2:
        exp_axes = (vl.EValidAxis.EPISTEMIC_TRANS, vl.EValidAxis.ALEATORY)
    observable = vl.ValidObservable(
        components=("length",), geometry=vl.EValidGeometry.GLOBAL, phys_units="m"
    )
    return vl.ValidCase(
        {"length": observable},
        {"length": vl.ValidArray(sim, sim_axes)},
        {"length": vl.ValidArray(exp, exp_axes)},
    )


@pytest.mark.parametrize(
    "reference",
    [
        0.0,
        np.inf,
        np.nan,
        validplot.ECDFReference(probability=0.0),
        validplot.ECDFReference(probability=1.1),
    ],
)
def test_invalid_normalisation_does_not_add_partial_axes(
    reference: float | validplot.ECDFReference,
) -> None:
    plot = validplot.plot_sim_exp_cdfs(
        create_ecdf_case(np.array([0.0, 2.0]), np.array([1.0, 3.0])), "length"
    )
    with pytest.raises(ValueError):
        validplot.add_normalised_ecdf_axis(plot, reference)
    assert plot.normalisation is None
    assert not plot.axes[0, 0].child_axes
    plt.close(plot.figure)


def test_zero_reference_quantile_is_rejected() -> None:
    plot = validplot.plot_sim_exp_cdfs(
        create_ecdf_case(np.array([0.0, 2.0]), np.array([1.0, 3.0])), "length"
    )
    with pytest.raises(ValueError, match="nonzero"):
        validplot.add_normalised_ecdf_axis(plot, validplot.ECDFReference())
    plt.close(plot.figure)


def create_point_observables(points: int = 2) -> dict[str, vl.ValidObservable]:
    coords = np.column_stack(
        (np.arange(points), np.zeros(points), np.zeros(points))
    ).astype(float)
    return {
        "temperature": vl.ValidObservable(
            components=("temperature",),
            geometry=vl.EValidGeometry.POINT,
            coords=coords,
            times=None,
            phys_units="degC",
        )
    }


def test_paired_rmse_rejects_unrelated_sample_lengths() -> None:
    metric = vl.MetricRMSE()
    assert metric.calc(
        np.array([0.0, 2.0]), np.array([1.0, 0.0])
    ).value == pytest.approx(np.sqrt(2.5))
    with pytest.raises(ValueError, match="paired"):
        metric.calc(np.array([0.0, 1.0]), np.array([1.0]))


@pytest.mark.parametrize(
    "metric",
    (vl.calc_avm_1d, vl.calc_ks_1d, vl.calc_cvm_1d, vl.calc_mavm_1d),
)
def test_distribution_kernels_reject_implicit_axis_pooling(
    metric: Callable[[np.ndarray, np.ndarray], object],
) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        metric(np.ones((2, 3)), np.ones(3))


def create_linear_sim(shift: float = 0.0) -> SimData:
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    times = np.array([0.0, 1.0])
    temperature = coords[:, :1] + 2 * coords[:, 1:2] + 3 * times + shift
    return SimData(
        num_spat_dims=2,
        coords=coords,
        time=times,
        node_vars={"temperature": temperature},
        glob_vars={"mean_temperature": 1.5 + 3 * times + shift},
    )


@pytest.mark.parametrize(
    "indices, message",
    [(((0,), (0,)), "Duplicate"), (((0,),), "incomplete"), (((2,),), "outside")],
)
def test_sample_sweep_rejects_invalid_run_tensor(
    indices: tuple[tuple[int, ...], ...], message: str
) -> None:
    field = sens.FieldScalar(create_linear_sim(), "temperature", sens.EDim.TWOD)
    observable = vl.ValidObservable(
        components=("temperature",),
        geometry=vl.EValidGeometry.POINT,
        coords=np.array([[0.25, 0.5, 0.0]]),
        times=np.array([0.5]),
    )
    with pytest.raises(ValueError, match=message):
        vl.sample_sim_sweep_for_valid(
            {"temperature": field},
            {"temperature": observable},
            ((index, create_linear_sim()) for index in indices),
            sample_axes=(vl.EValidAxis.ALEATORY,),
            sample_shape=(2,),
        )


@pytest.mark.parametrize(
    "samples", [np.ones((2, 2)), np.array([np.inf]), np.array([np.nan]), np.array([])]
)
def test_calc_ecdf_rejects_invalid_samples(samples: np.ndarray) -> None:
    with pytest.raises(ValueError):
        vl.calc_ecdf(samples)


@pytest.mark.parametrize("index", [-1, 2])
def test_extraction_rejects_invalid_epistemic_indices(
    ecdf_quantity: vl.ValidArray, index: int
) -> None:
    with pytest.raises(IndexError, match="EPISTEMIC_TRANS"):
        vl.extract_ecdf(ecdf_quantity, epis_trans_idx=index)
    with pytest.raises(IndexError, match="EPISTEMIC_TRANS"):
        vl.extract_ecdf_envelope(ecdf_quantity, epis_trans_idx=index)


def test_extraction_rejects_nonzero_index_on_absent_axis() -> None:
    ecdf_quantity = vl.ValidArray(np.array([0.0, 1.0]), (vl.EValidAxis.ALEATORY,))
    with pytest.raises(IndexError, match="EPISTEMIC_NUIS"):
        vl.extract_ecdf(ecdf_quantity, epis_nuis_idx=1)


def create_plot_epistemic_case() -> vl.ValidCase:
    shifts = np.array([[1.0, 9.0, 8.0], [2.0, 3.0, 7.0], [4.0, 6.0, 5.0]])
    samples = shifts[..., None] + np.array([0.0, 2.0])
    quantity = vl.ValidArray(
        samples,
        (
            vl.EValidAxis.EPISTEMIC_TRANS,
            vl.EValidAxis.EPISTEMIC_NUIS,
            vl.EValidAxis.ALEATORY,
        ),
    )
    observable = vl.ValidObservable(
        components=("length",), geometry=vl.EValidGeometry.GLOBAL, phys_units="m"
    )
    exp = vl.ValidArray(np.array([2.0, 5.0, 8.0]), (vl.EValidAxis.ALEATORY,))
    return vl.ValidCase(
        {"length": observable}, {"length": quantity}, {"length": exp}
    )


@pytest.mark.parametrize(
    "points, exception",
    [
        ((), ValueError),
        ((-1,), IndexError),
        ((12,), IndexError),
        ((True,), TypeError),
        ((0.5,), TypeError),
    ],
)
def test_invalid_points_do_not_create_figures(
    points: tuple[int, ...], exception: type[Exception]
) -> None:
    figures = plt.get_fignums()
    with pytest.raises(exception):
        validplot.plot_sim_ecdfs(load_plate_case(), "temperature", point_idxs=points)
    assert plt.get_fignums() == figures


@pytest.mark.parametrize(
    "view, exception",
    [
        (vl.ECDFViewOpts(trans_idxs=()), ValueError),
        (vl.ECDFViewOpts(trans_idxs=(3,)), IndexError),
        (vl.ECDFViewOpts(trans_idxs=(0, 0)), ValueError),
        (vl.ECDFViewOpts(trans_idxs=(True,)), TypeError),
        (vl.ECDFViewOpts(scenario_idxs=((0, 3),)), IndexError),
        (vl.ECDFViewOpts(scenario_idxs=((0, 0), (0, 0))), ValueError),
        (vl.ECDFViewOpts(bounds=False), ValueError),
    ],
)
def test_invalid_views_fail_before_plotting(
    view: vl.ECDFViewOpts, exception: type[Exception]
) -> None:
    figures = plt.get_fignums()
    with pytest.raises(exception):
        validplot.plot_sim_ecdfs(create_plot_epistemic_case(), "length", view=view)
    assert plt.get_fignums() == figures


@pytest.mark.parametrize("sample_index", [0, 1, 2])
@pytest.mark.parametrize("separation", [0.0, MAVM_DUPLICATE_TOLERANCE / 2.0])
def test_mavm_pbox_rejects_duplicate_samples(
    sample_index: int, separation: float
) -> None:
    samples = [np.array([0.0, 1.0, 2.0]) for _ in range(3)]
    samples[sample_index] = np.array([0.0, separation, 2.0])
    with pytest.raises(ValueError, match="REJECT MAVM duplicates"):
        calc_mavm_pbox_1d(*samples, duplicates=EMAVMDuplicates.REJECT)


@pytest.mark.parametrize("separation", [0.0, MAVM_DUPLICATE_TOLERANCE / 2.0])
def test_mavm_default_rejects_near_duplicate_model_data(separation: float) -> None:
    """REJECT directs near-duplicate input to the robust implementation."""
    model_data = np.array([0.0, separation, 2.0], dtype=np.float64)
    exp_data = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    with pytest.raises(
        ValueError,
        match="REJECT MAVM duplicates requires unique sim observations.*INCLUDE",
    ):
        calc_mavm_1d(model_data, exp_data, duplicates=EMAVMDuplicates.REJECT)


def test_mavm_default_rejects_near_duplicate_experimental_data() -> None:
    """REJECT names experimental data when it contains near duplicates."""
    model_data = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    exp_data = np.array(
        [0.5, 0.5 + MAVM_DUPLICATE_TOLERANCE / 2.0, 2.5], dtype=np.float64
    )
    with pytest.raises(
        ValueError,
        match="REJECT MAVM duplicates requires unique experimental observations.*INCLUDE",
    ):
        calc_mavm_1d(model_data, exp_data, duplicates=EMAVMDuplicates.REJECT)


@pytest.mark.parametrize(
    "observations, message",
    [
        (np.ones((2, 2)), "one-dimensional"),
        (np.array([0.0, np.inf]), "infinite"),
        (np.array([np.nan]), "finite observation"),
        (np.array([]), "finite observation"),
    ],
)
@pytest.mark.parametrize("invalid_model", [True, False])
def test_mavm_rejects_invalid_samples(
    observations: np.ndarray, message: str, invalid_model: bool
) -> None:
    valid_observations = np.array([0.0, 1.0])
    model = observations if invalid_model else valid_observations
    experimental = valid_observations if invalid_model else observations
    with pytest.raises(ValueError, match=message):
        calc_mavm_1d(model, experimental)


@pytest.mark.parametrize("alpha", [0.0, 1.0, np.nan])
def test_mavm_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        calc_mavm_1d(np.array([0.0, 1.0]), np.array([0.0, 1.0]), alpha=alpha)


def test_selection_requires_all_scenarios_and_preserves_output(
    metric_scene: tuple[vl.ValidResult, io.SimData],
) -> None:
    result, _ = metric_scene
    field = result.metrics["temperature"][vl.EMetricType.MAVM]
    with pytest.raises(ValueError, match="Select"):
        vl.extract_metric_values(field, output=vl.EMetricOut.D_MINUS)
    selected = vl.extract_metric_values(
        field,
        selectors={vl.EValidResAxis.TIME: 0, vl.EValidResAxis.SIM_TRANS: 0},
        output=vl.EMetricOut.D_MINUS,
    )
    np.testing.assert_allclose(selected, [0.0, 3.0, 1.0, np.nan])
    assert field.metric_vals[vl.EMetricOut.D_MINUS][2, 0, 0, 0] == 1.0
    with pytest.raises(ValueError, match="output"):
        vl.select_metric_output(
            vl.MetricField(
                field.axes,
                {
                    vl.EMetricOut.D_MINUS: field.metric_vals[
                        vl.EMetricOut.D_MINUS
                    ]
                },
            ),
            vl.EMetricOut.D_PLUS,
        )


@pytest.mark.parametrize(
    "time_idx, units, coord_frame, values, message",
    [
        (0, "mm", "cube", np.ones(4), "times must match"),
        (1, "m", "cube", np.ones(4), "units/coord_frame"),
        (1, "mm", "other", np.ones(4), "units/coord_frame"),
        (1, "mm", "cube", np.ones(3), "n_nodes"),
    ],
)
def test_overlay_rejects_misaligned_background(
    metric_scene: tuple[vl.ValidResult, io.SimData],
    time_idx: int,
    units: str,
    coord_frame: str,
    values: np.ndarray,
    message: str,
) -> None:
    result, sim_data = metric_scene
    with pytest.raises(ValueError, match=message):
        validvis.plot_metric_on_sim(
            result,
            "temperature",
            vl.EMetricType.MAVM,
            selectors={vl.EValidResAxis.TIME: 1, vl.EValidResAxis.SIM_TRANS: 0},
            sim_data=sim_data,
            sim_field=values,
            sim_time_idx=time_idx,
            sim_coord_units=units,
            sim_coord_frame=coord_frame,
            output=vl.EMetricOut.D_PLUS,
        )


def test_output_limits_must_increase(
    metric_scene: tuple[vl.ValidResult, io.SimData],
) -> None:
    result, _ = metric_scene
    with pytest.raises(ValueError, match="increasing"):
        validvis.plot_metric_on_sim(
            result,
            "temperature",
            vl.EMetricType.MAVM,
            selectors={vl.EValidResAxis.TIME: 0, vl.EValidResAxis.SIM_TRANS: 0},
            output=vl.EMetricOut.D_MINUS,
            metric_opts=validvis.MetricVisOpts(metric_colour_lims=(2.0, 1.0)),
        )


@pytest.mark.parametrize(
    ("sim_field", "sim_time_idx", "message"),
    [
        ("missing", 1, "no nodal field"),
        ("temperature", None, "time index"),
        ("temperature", -1, "time index"),
        ("temperature", 2, "time index"),
        (np.ones((4, 2)), 1, "n_nodes"),
        (np.array([1.0, 2.0, 3.0, np.inf]), 1, "infinite"),
    ],
)
def test_background_rejects_invalid_fields_and_steps(
    metric_scene: tuple[vl.ValidResult, io.SimData],
    sim_field: str | np.ndarray,
    sim_time_idx: int | None,
    message: str,
) -> None:
    result, sim_data = metric_scene
    with pytest.raises(ValueError, match=message):
        validvis.plot_metric_on_sim(
            result,
            "temperature",
            vl.EMetricType.MAVM,
            output=vl.EMetricOut.D_PLUS,
            selectors={vl.EValidResAxis.TIME: 1, vl.EValidResAxis.SIM_TRANS: 0},
            sim_data=sim_data,
            sim_field=sim_field,
            sim_time_idx=sim_time_idx,
            sim_coord_units="mm",
            sim_coord_frame="cube",
        )


def test_metric_field_selectors() -> None:
    axes = (
        vl.EValidResAxis.POINT,
        vl.EValidResAxis.COMPONENT,
        vl.EValidResAxis.TIME,
    )
    field = vl.MetricField(
        axes, {vl.EMetricOut.ERROR: np.zeros((2, 1, 1))}
    )
    for index in (-1, 2, True, 0.0):
        with pytest.raises((TypeError, IndexError)):
            _build_metric_selection(field, {vl.EValidResAxis.POINT: index})
    with pytest.raises(ValueError):
        _build_metric_selection(field, {vl.EValidResAxis.SIM_NUIS: 0})


def test_case_validates_sample_input_dimensions_for_each_source() -> None:
    """Both provenance fields must describe the source's uncertainty axes."""
    observables = create_point_observables(points=1)
    axes = (
        vl.EValidAxis.POINT,
        vl.EValidAxis.COMPONENT,
        vl.EValidAxis.ALEATORY,
    )
    quantities = {"temperature": vl.ValidArray(np.ones((1, 1, 3)), axes)}
    mismatched_inputs = vl.ValidSampleInputs(
        run_ids=vl.ValidArray(
            np.array(["run_0", "run_1"]), (vl.EValidAxis.ALEATORY,)
        )
    )
    with pytest.raises(ValueError, match="sample input ALEATORY mismatch"):
        vl.ValidCase(
            observables, quantities, quantities, sim_inputs=mismatched_inputs
        )
    with pytest.raises(ValueError, match="sample input ALEATORY mismatch"):
        vl.ValidCase(
            observables, quantities, quantities, exp_inputs=mismatched_inputs
        )


def test_multi_scenario_integration_requires_an_explicit_original_comparison() -> None:
    test_case = create_ecdf_case(
        np.array([[0.0, 2.0], [10.0, 12.0]]), np.array([1.0, 3.0])
    )
    plot = validplot.plot_sim_exp_cdfs(test_case, "length")
    with pytest.raises(ValueError, match="explicit original"):
        validplot.add_metric_overlay(
            plot,
            vl.MetricAVM(),
            display=validplot.EMetricDisp.INTEGRATION_AREA,
        )
    validplot.add_metric_overlay(
        plot,
        vl.MetricAVM(),
        display=validplot.EMetricDisp.INTEGRATION_AREA,
        comparison=validplot.ECDFComparison(),
    )
    plt.close(plot.figure)
