"""Validation integration behaviour."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pyvista as pv
from plate_fixture import load_plate_case

import pyvale.dataio as io
import pyvale.sensorsim as sens
import pyvale.valid.validpreparation as validprep
from pyvale import valid
from pyvale.valid import validplot, validvis


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


@pytest.mark.parametrize("scale", tuple(valid.EMetricScale))
def test_custom_calibration_classification_allows_unequal_counts_and_controls_units(
    scale: valid.EMetricScale,
) -> None:
    axes = (valid.EValidAxis.ALEATORY,)
    valid_case = valid.ValidCase(
        {
            "length": valid.ValidObservable(
                components=("length",),
                geometry=valid.EValidGeometry.GLOBAL,
                phys_units="m",
            )
        },
        {"length": valid.ValidArray(np.array([0.1, 0.3]), axes)},
        {"length": valid.ValidArray(np.array([0.0]), axes)},
    )
    metric = MetricSampleDifference(valid.EMetricComparison.CALIBRATION, scale)
    result = valid.ValidEngine([metric]).run(valid_case)
    field = result.metrics["length"][valid.EMetricType.KS]
    np.testing.assert_allclose(field.outputs[valid.EMetricOutput.STATISTIC], [[[0.4]]])
    fig, ax = validplot.plot_valid_metric_distribution(
        result, "length", valid.EMetricType.KS
    )
    try:
        units = "m" if scale is valid.EMetricScale.PHYSICAL else "1"
        assert ax.get_xlabel().endswith(f"[{units}]")
    finally:
        plt.close(fig)
    metric.comparison = valid.EMetricComparison.PAIRED
    with pytest.raises(ValueError, match="equal aleatory counts"):
        valid.ValidEngine([metric]).run(valid_case)


def create_point_observables(points: int = 2) -> dict[str, valid.ValidObservable]:
    coords = np.column_stack(
        (np.arange(points), np.zeros(points), np.zeros(points))
    ).astype(float)
    return {
        "temperature": valid.ValidObservable(
            components=("temperature",),
            geometry=valid.EValidGeometry.POINT,
            coords=coords,
            times=None,
            phys_units="degC",
        )
    }


def test_arrays_are_zero_copy_and_axis_order_is_checked() -> None:
    samples = np.arange(12.0).reshape(2, 1, 2, 3)
    axes = (
        valid.EValidAxis.POINT,
        valid.EValidAxis.COMPONENT,
        valid.EValidAxis.EPISTEMIC_TRANS,
        valid.EValidAxis.ALEATORY,
    )
    wrapped = {"temperature": valid.ValidArray(samples, axes)}
    valid_case = valid.ValidCase(create_point_observables(), wrapped, wrapped)
    assert valid_case.sim["temperature"].data is samples
    with pytest.raises(ValueError, match="canonical"):
        valid.ValidArray(samples, axes[::-1])
    with pytest.raises(ValueError, match="unique"):
        valid.ValidArray(
            np.ones((2, 2)), (valid.EValidAxis.POINT, valid.EValidAxis.POINT)
        )


def test_combine_cases_preserves_transient_run_pairing() -> None:
    coords = np.array([[0.0, 0.0, 0.0]])
    times = np.array([0.0, 1.0])
    observables = {
        "temperature": valid.ValidObservable(
            components=("temperature",),
            geometry=valid.EValidGeometry.POINT,
            coords=coords,
            times=times,
            phys_units="degC",
        )
    }
    axes = (valid.EValidAxis.POINT, valid.EValidAxis.COMPONENT, valid.EValidAxis.TIME)
    sim = {"temperature": valid.ValidArray(np.zeros((1, 1, 2)), axes)}
    cases = [
        valid.ValidCase(
            observables,
            sim,
            {"temperature": valid.ValidArray(np.array([[[i, i + 10.0]]]), axes)},
        )
        for i in (1.0, 2.0)
    ]
    joined = validprep.concatenate_valid_cases(*cases)
    assert joined.sim is cases[0].sim
    for case in cases:
        assert case.exp["temperature"].axes == axes
        assert case.exp["temperature"].data.shape == (1, 1, 2)
    np.testing.assert_array_equal(
        joined.exp["temperature"].data, [[[[1.0, 2.0], [11.0, 12.0]]]]
    )
    np.testing.assert_array_equal(joined.exp_inputs.run_ids.data, ["case_0", "case_1"])


def test_combine_cases_preserves_parameter_provenance() -> None:
    observables = create_point_observables(points=1)
    axes = (valid.EValidAxis.POINT, valid.EValidAxis.COMPONENT)
    sim = {"temperature": valid.ValidArray(np.array([[0.0]]), axes)}
    cases = []
    for index, parameter in enumerate((10.0, 20.0)):
        inputs = valid.ValidSampleInputs(
            parameters={"loading": valid.ValidArray(np.array(parameter), ())}
        )
        exp = {"temperature": valid.ValidArray(np.array([[parameter]]), axes)}
        cases.append(valid.ValidCase(observables, sim, exp, exp_inputs=inputs))
    joined = validprep.concatenate_valid_cases(*cases)
    for case in cases:
        assert case.exp_inputs.parameters["loading"].axes == ()
        assert case.exp_inputs.parameters["loading"].data.shape == ()
    np.testing.assert_array_equal(
        joined.exp_inputs.parameters["loading"].data, [10.0, 20.0]
    )


def test_sensor_sim_bridge_retains_component_and_run_values() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    values = np.arange(16.0).reshape(2, 2, 2, 2)
    output = {
        ("plate", "strain", "meas"): values,
        ("plate", "strain", "sens_times"): np.array([0.0, 1.0]),
    }
    converted = validprep.extract_exp_sim_data(
        output,
        sim_key="plate",
        selections={"strain_xy": validprep.SensorOutputSelection("strain", 1, coords)},
    )
    assert len(converted) == 2
    np.testing.assert_array_equal(converted[1]["strain_xy"].fields, values[1, :, 1, :])


def test_combine_cases_preserves_sim_provenance() -> None:
    """Combining acquisitions retains the original model sample identifiers."""
    axes = (
        valid.EValidAxis.POINT,
        valid.EValidAxis.COMPONENT,
        valid.EValidAxis.ALEATORY,
    )
    quantities = {"temperature": valid.ValidArray(np.ones((1, 1, 2)), axes)}
    sim_inputs = valid.ValidSampleInputs(
        run_ids=valid.ValidArray(
            np.array(["sim_0", "sim_1"]), (valid.EValidAxis.ALEATORY,)
        )
    )
    valid_case = valid.ValidCase(
        create_point_observables(points=1),
        quantities,
        quantities,
        sim_inputs=sim_inputs,
    )
    combined = validprep.concatenate_valid_cases(valid_case, valid_case)
    assert combined.sim_inputs is sim_inputs
    np.testing.assert_array_equal(combined.sim_inputs.run_ids.data, ["sim_0", "sim_1"])
    assert combined.exp["temperature"].data.shape == (1, 1, 4)


def test_extract_ecdf_selects_epistemic_cell_and_retains_duplicate_mass(
    ecdf_quantity: valid.ValidArray,
) -> None:
    cdf = valid.extract_ecdf(
        ecdf_quantity,
        epis_trans_idx=1,
        epis_nuis_idx=0,
        offset=valid.EEpisOffset.UPPER,
    )
    np.testing.assert_array_equal(cdf.quantiles, [12.0, 14.0])
    np.testing.assert_allclose(cdf.probabilities, [2.0 / 3.0, 1.0])
    assert ecdf_quantity.data[1, 0, 0] == 10.0


def test_envelope_fixes_transferable_axis_and_spans_nuisance_axis(
    ecdf_quantity: valid.ValidArray,
) -> None:
    support, lower, upper = valid.extract_ecdf_envelope(
        ecdf_quantity, epis_trans_idx=1, offset=valid.EEpisOffset.NOMINAL
    )
    np.testing.assert_array_equal(support, [10.0, 12.0, 14.0])
    np.testing.assert_allclose(lower, [0.0, 2.0 / 3.0, 1.0])
    np.testing.assert_allclose(upper, [2.0 / 3.0, 1.0, 1.0])


def test_envelope_spans_coherent_offsets_of_one_epistemic_cell(
    ecdf_quantity: valid.ValidArray,
) -> None:
    support, lower, upper = valid.extract_ecdf_envelope(
        ecdf_quantity, epis_trans_idx=1, epis_nuis_idx=0
    )
    np.testing.assert_array_equal(support, [9.0, 10.0, 11.0, 12.0, 14.0])
    np.testing.assert_allclose(lower, [0.0, 0.0, 0.0, 2.0 / 3.0, 1.0])
    np.testing.assert_allclose(upper, [2.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0])


def test_metric_panels_select_direction_and_leave_selectors_unchanged() -> None:
    valid_case = load_plate_case(transient=True, epistemic=True, offsets=True)
    result = valid.ValidEngine([valid.MetricMAVM()]).run(valid_case)
    selectors = {
        valid.EValidResAxis.SIM_TRANS: 1,
        valid.EValidResAxis.EXP_OFFSET: valid.EEpisOffset.UPPER,
    }
    fig, axes = validplot.plot_valid_metric_time_points(
        result,
        "temperature",
        valid.EMetricType.MAVM,
        point_idxs=(0, 5),
        selectors=selectors,
        output=valid.EMetricOutput.D_MINUS,
    )
    field = result.metrics["temperature"][valid.EMetricType.MAVM]
    for point, ax in zip((0, 5), axes.flat, strict=True):
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(),
            field.outputs[valid.EMetricOutput.D_MINUS][point, 0, :, 1, 2],
        )
        assert ax.get_ylabel() == "MAVM d− [degC]"
    assert valid.EValidResAxis.POINT not in selectors
    plt.close(fig)


def test_directional_outputs_can_share_time_axes() -> None:
    case = load_plate_case(transient=True)
    result = valid.ValidEngine([valid.MetricMAVM()]).run(case)
    fig, ax = validplot.plot_valid_metric_time(
        result,
        "temperature",
        valid.EMetricType.MAVM,
        selectors={valid.EValidResAxis.POINT: 0},
        output=valid.EMetricOutput.D_PLUS,
    )
    validplot.plot_valid_metric_time(
        result,
        "temperature",
        valid.EMetricType.MAVM,
        selectors={valid.EValidResAxis.POINT: 0},
        output=valid.EMetricOutput.D_MINUS,
        ax=ax,
    )
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "MAVM d+",
        "MAVM d−",
    ]
    field = result.metrics["temperature"][valid.EMetricType.MAVM]
    for line, output in zip(
        ax.lines, (valid.EMetricOutput.D_PLUS, valid.EMetricOutput.D_MINUS), strict=True
    ):
        np.testing.assert_allclose(line.get_ydata(), field.outputs[output][0, 0])
    plt.close(fig)


def create_plot_epistemic_case() -> valid.ValidCase:
    shifts = np.array([[1.0, 9.0, 8.0], [2.0, 3.0, 7.0], [4.0, 6.0, 5.0]])
    samples = shifts[..., None] + np.array([0.0, 2.0])
    quantity = valid.ValidArray(
        samples,
        (
            valid.EValidAxis.EPISTEMIC_TRANS,
            valid.EValidAxis.EPISTEMIC_NUIS,
            valid.EValidAxis.ALEATORY,
        ),
    )
    observable = valid.ValidObservable(
        components=("length",), geometry=valid.EValidGeometry.GLOBAL, phys_units="m"
    )
    exp = valid.ValidArray(np.array([2.0, 5.0, 8.0]), (valid.EValidAxis.ALEATORY,))
    return valid.ValidCase(
        {"length": observable}, {"length": quantity}, {"length": exp}
    )


def test_offsets_are_explicit_and_retained_as_coherent_scenarios() -> None:
    quantity = valid.ValidArray(
        np.array([1.0, 3.0]),
        (valid.EValidAxis.ALEATORY,),
        valid.EpistemicOffsets(0.5, 2.0),
    )
    nominal = valid.calc_ecdf_view(quantity)
    assert nominal.quantiles.shape == (1, 1, 1, 2)
    np.testing.assert_array_equal(nominal.quantiles[0, 0, 0], [1.0, 3.0])
    all_offsets = valid.calc_ecdf_view(
        quantity, offset=None, view=valid.ValidECDFView(scenario_idxs=None)
    )
    assert all_offsets.quantiles.shape == (1, 1, 3, 2)
    np.testing.assert_array_equal(
        all_offsets.quantiles[0, 0], [[1.0, 3.0], [0.5, 2.5], [3.0, 5.0]]
    )
    assert tuple(scenario[2] for scenario in all_offsets.scenarios) == tuple(
        valid.EEpisOffset
    )


def test_many_scenarios_default_to_two_bounds() -> None:
    case = create_plot_epistemic_case()
    case.sim["length"] = valid.ValidArray(
        np.arange(64.0)[:, None] + np.array([0.0, 1.0]),
        (valid.EValidAxis.EPISTEMIC_TRANS, valid.EValidAxis.ALEATORY),
    )
    plot = validplot.plot_sim_ecdfs(case, "length")
    try:
        axis = plot.axes[0, 0]
        assert len(axis.lines) == 2
        np.testing.assert_allclose(axis.lines[0].get_xdata()[1:], [0.0, 1.0])
        np.testing.assert_allclose(axis.lines[1].get_xdata()[1:], [63.0, 64.0])
    finally:
        plt.close(plot.figure)


def test_existing_axes_support_composition_but_reject_multiple_points() -> None:
    fig, ax = plt.subplots()
    ecdf_plot: validplot.ValidECDFPlot = validplot.plot_sim_ecdfs(
        create_plot_epistemic_case(), "length", ax=ax
    )
    returned, axes = (ecdf_plot.figure, ecdf_plot.axes)
    assert returned is fig and axes[0, 0] is ax
    with pytest.raises(ValueError, match="one point"):
        validplot.plot_sim_ecdfs(
            load_plate_case(), "temperature", point_idxs=(0, 1), ax=ax
        )
    plt.close(fig)


def create_ecdf_case(sim: np.ndarray, exp: np.ndarray) -> valid.ValidCase:
    sim_axes = (valid.EValidAxis.ALEATORY,)
    if sim.ndim == 2:
        sim_axes = (valid.EValidAxis.EPISTEMIC_TRANS, valid.EValidAxis.ALEATORY)
    exp_axes = (valid.EValidAxis.ALEATORY,)
    if exp.ndim == 2:
        exp_axes = (valid.EValidAxis.EPISTEMIC_TRANS, valid.EValidAxis.ALEATORY)
    observable = valid.ValidObservable(
        components=("length",), geometry=valid.EValidGeometry.GLOBAL, phys_units="m"
    )
    return valid.ValidCase(
        {"length": observable},
        {"length": valid.ValidArray(sim, sim_axes)},
        {"length": valid.ValidArray(exp, exp_axes)},
    )


@pytest.mark.parametrize("probability, expected", [(0.5, 5.0), (0.75, 7.0), (1.0, 7.0)])
def test_reference_quantile_normalisation_is_a_constant_secondary_axis(
    probability: float, expected: float
) -> None:
    case = create_ecdf_case(np.array([[0.0, 2.0], [10.0, 12.0]]), np.array([1.0, 3.0]))
    plot = validplot.plot_sim_exp_cdfs(case, "length")
    original_quantiles = plot.axes[0, 0].lines[0].get_xdata().copy()
    validplot.add_normalised_ecdf_axis(
        plot, validplot.ValidECDFReference(probability=probability)
    )
    plot.figure.canvas.draw()
    assert plot.normalisation == (expected,)
    np.testing.assert_array_equal(
        plot.axes[0, 0].lines[0].get_xdata(), original_quantiles
    )
    secondary = plot.axes[0, 0].child_axes[0]
    np.testing.assert_allclose(
        secondary.get_xlim(), np.asarray(plot.axes[0, 0].get_xlim()) / expected
    )
    with pytest.raises(ValueError, match="already"):
        validplot.add_normalised_ecdf_axis(plot, 2.0)
    plt.close(plot.figure)


def test_constant_and_exp_reference_scalings_are_explicit() -> None:
    case = create_ecdf_case(np.array([2.0, 4.0]), np.array([10.0, 20.0]))
    plot = validplot.plot_sim_exp_cdfs(case, "length")
    validplot.add_normalised_ecdf_axis(
        plot, validplot.ValidECDFReference(source=valid.EValidSource.EXP)
    )
    assert plot.normalisation == (10.0,)
    plt.close(plot.figure)
    plot = validplot.plot_sim_ecdfs(case, "length")
    validplot.add_normalised_ecdf_axis(plot, 4.0)
    assert plot.normalisation == (4.0,)
    plt.close(plot.figure)


@pytest.mark.parametrize(
    "output, exp, expected",
    [
        (valid.EMetricOutput.D_PLUS, np.array([11.0, 13.0]), np.array([11.0, 13.0])),
        (valid.EMetricOutput.D_MINUS, np.array([-1.0, 1.0]), np.array([-1.0, 1.0])),
    ],
)
def test_mavm_support_uses_outer_shifted_ecdfs_not_global_largest_distance(
    output: valid.EMetricOutput, exp: np.ndarray, expected: np.ndarray
) -> None:
    case = create_ecdf_case(np.array([[0.0, 2.0], [10.0, 12.0]]), exp)
    plot = validplot.plot_sim_exp_cdfs(case, "length")
    validplot.add_metric_overlay(plot, valid.MetricMAVM(alpha=0.5), output=output)
    ax = plot.axes[0, 0]
    shifted_line = (
        ax.lines[-1] if output is valid.EMetricOutput.D_PLUS else ax.lines[-2]
    )
    np.testing.assert_allclose(
        shifted_line.get_xdata()[1:], expected, rtol=0, atol=1e-12
    )
    assert "outer support shift" in ax.collections[-1].get_label()
    plt.close(plot.figure)


@pytest.mark.parametrize(
    "metric, output, expected_area",
    [
        (valid.MetricAVM(), valid.EMetricOutput.DISTANCE, 1.0),
        (valid.MetricMAVM(alpha=0.5), valid.EMetricOutput.D_PLUS, 0.5),
        (valid.MetricMAVM(alpha=0.5), valid.EMetricOutput.D_MINUS, 0.5),
    ],
)
def test_integration_polygon_area_equals_the_metric(
    metric: valid.IValidMetric, output: valid.EMetricOutput, expected_area: float
) -> None:
    sim = (
        np.array([0.0, 2.0])
        if isinstance(metric, valid.MetricAVM)
        else np.array([0.0, 4.0])
    )
    plot = validplot.plot_sim_exp_cdfs(
        create_ecdf_case(sim, np.array([1.0, 3.0])), "length"
    )
    validplot.add_metric_overlay(
        plot,
        metric,
        output=output,
        display=validplot.EValidMetricDisplay.INTEGRATION_AREA,
    )
    area = 0.0
    for path in plot.axes[0, 0].collections[-1].get_paths():
        coords = path.vertices
        area += (
            abs(
                np.dot(coords[:, 0], np.roll(coords[:, 1], -1))
                - np.dot(coords[:, 1], np.roll(coords[:, 0], -1))
            )
            / 2
        )
    assert area == pytest.approx(expected_area, rel=0, abs=1e-12)
    plt.close(plot.figure)


def test_ks_segments_show_exact_single_and_bound_cdf_differences() -> None:
    plot = validplot.plot_sim_exp_cdfs(
        create_ecdf_case(np.array([0.0, 2.0]), np.array([1.0, 3.0])), "length"
    )
    validplot.add_metric_overlay(plot, valid.MetricKS())
    line = plot.axes[0, 0].lines[-1]
    assert np.ptp(line.get_xdata()) == 0
    assert np.ptp(line.get_ydata()) == pytest.approx(0.5)
    plt.close(plot.figure)
    plot = validplot.plot_sim_exp_cdfs(
        create_ecdf_case(np.array([[0.0, 2.0], [10.0, 12.0]]), np.array([1.0, 3.0])),
        "length",
    )
    validplot.add_metric_overlay(plot, valid.MetricKS())
    labels = [line.get_label() for line in plot.axes[0, 0].lines[-2:]]
    assert "lower-support bounds" in labels[0]
    assert "upper-support bounds" in labels[1]
    plt.close(plot.figure)


def test_nongeometric_metrics_compose_as_annotations_without_reinterpreting_area() -> (
    None
):
    plot = validplot.plot_sim_exp_cdfs(
        create_ecdf_case(np.array([0.0, 2.0]), np.array([1.0, 3.0])), "length"
    )
    for metric in (valid.MetricCVM(), valid.MetricRMSE(), valid.MetricRelErr()):
        validplot.add_metric_overlay(plot, metric)
    assert len(plot.axes[0, 0].texts) == 3
    assert not plot.axes[0, 0].collections
    with pytest.raises(ValueError, match="AVM or MAVM"):
        validplot.add_metric_overlay(
            plot,
            valid.MetricRMSE(),
            display=validplot.EValidMetricDisplay.SUPPORT_SHIFT,
        )
    plt.close(plot.figure)


def test_u_pooling_has_its_own_probability_coordinate_plot() -> None:
    cdf = valid.ValidECDF(np.array([0.0, 2.0]), np.array([0.5, 1.0]))
    fig, ax = validplot.plot_u_pooling_cdf([cdf], np.array([1.0]))
    label = ax.lines[0].get_label()
    assert isinstance(label, str)
    assert "0.25" in label
    assert ax.get_xlim() == (0.0, 1.0)
    assert "[1]" in ax.get_xlabel()
    plt.close(fig)


@pytest.mark.parametrize(
    "output, expected",
    [
        (valid.EMetricOutput.D_PLUS, [55.0, 115.0]),
        (valid.EMetricOutput.D_MINUS, [-15.0, 45.0]),
    ],
)
def test_shift_bounds_can_switch_controlling_scenario_with_both_epistemic_axes(
    output: valid.EMetricOutput, expected: list[float]
) -> None:
    axes = (
        valid.EValidAxis.EPISTEMIC_TRANS,
        valid.EValidAxis.EPISTEMIC_NUIS,
        valid.EValidAxis.ALEATORY,
    )
    sim = valid.ValidArray(
        np.tile(np.array([[[0.0, 100.0], [40.0, 60.0]]]), (2, 1, 1)), axes
    )
    exp = valid.ValidArray(
        np.tile(np.array([[[10.0, 90.0], [30.0, 70.0]]]), (2, 1, 1)), axes
    )
    observable = valid.ValidObservable(
        components=("length",), geometry=valid.EValidGeometry.GLOBAL
    )
    case = valid.ValidCase({"length": observable}, {"length": sim}, {"length": exp})
    plot = validplot.plot_sim_exp_cdfs(case, "length")
    validplot.add_metric_overlay(plot, valid.MetricMAVM(alpha=0.5), output=output)
    line = (
        plot.axes[0, 0].lines[-1]
        if output is valid.EMetricOutput.D_PLUS
        else plot.axes[0, 0].lines[-2]
    )
    np.testing.assert_allclose(line.get_xdata()[1:], expected, rtol=0, atol=1e-12)
    plt.close(plot.figure)


def test_avm_shift_is_a_symmetric_magnitude_about_each_original_curve() -> None:
    plot = validplot.plot_sim_exp_cdfs(
        create_ecdf_case(np.array([[0.0, 2.0], [10.0, 12.0]]), np.array([11.0, 13.0])),
        "length",
    )
    validplot.add_metric_overlay(plot, valid.MetricAVM())
    np.testing.assert_allclose(plot.axes[0, 0].lines[-2].get_xdata()[1:], [-11.0, -9.0])
    np.testing.assert_allclose(plot.axes[0, 0].lines[-1].get_xdata()[1:], [11.0, 13.0])
    assert "symmetric" in plot.axes[0, 0].collections[-1].get_label()
    plt.close(plot.figure)


def test_precomputed_support_profiles_compose_without_metric_recalculation() -> None:
    case = create_ecdf_case(
        np.array([[0.0, 2.0], [10.0, 12.0]]), np.array([11.0, 13.0])
    )
    result = valid.ValidEngine([valid.MetricMAVM(alpha=0.5)]).run(case)
    supports = valid.find_metric_epis_supports(
        case, result, "length", valid.EMetricType.MAVM
    )
    plot = validplot.plot_sim_exp_cdfs(case, "length")
    validplot.add_support_overlay(plot, supports, output=valid.EMetricOutput.D_PLUS)
    np.testing.assert_allclose(plot.axes[0, 0].lines[-1].get_xdata()[1:], [11.0, 13.0])
    plt.close(plot.figure)


@pytest.mark.parametrize(
    "sim_field", ["temperature", np.array([100.0, 200.0, 300.0, 400.0])]
)
def test_overlay_preserves_xyz_ids_and_independent_scalar_bars(
    metric_scene: tuple[valid.ValidResult, io.SimData],
    sim_field: str | np.ndarray,
) -> None:
    result, sim_data = metric_scene
    original_coords = sim_data.coords.copy()
    scalars = np.array([100.0, 200.0, 300.0, 400.0])
    original_theme = pv.global_theme.font.color
    opts = sens.VisOptsSimSensors(
        interactive=False,
        window_size_px=(640, 480),
        camera_position="iso",
        font_colour="navy",
    )
    plotter = validvis.plot_valid_metric_on_sim(
        result,
        "temperature",
        valid.EMetricType.MAVM,
        selectors={valid.EValidResAxis.TIME: 1, valid.EValidResAxis.SIM_TRANS: 0},
        output=valid.EMetricOutput.D_MINUS,
        point_idxs=(2, 0, 3),
        sim_data=sim_data,
        sim_field=sim_field,
        sim_time_idx=1,
        descriptor=sens.SensorDescriptor(name="Temperature", units="degC", tag="P"),
        sim_coord_units="mm",
        sim_coord_frame="cube",
        vis_opts=opts,
        metric_opts=validvis.ValidMetricVisOpts(labels_always_visible=True),
    )
    try:
        assert plotter.off_screen
        metric_actor = plotter.actors["validation-points"]
        background_actor = plotter.actors["validation-background"]
        point_mesh = metric_actor.mapper.dataset
        np.testing.assert_array_equal(point_mesh.points, original_coords[[2, 0, 3]])
        np.testing.assert_array_equal(point_mesh["validation_point_id"], [2, 0, 3])
        np.testing.assert_allclose(point_mesh["validation_metric"], [1.0, 0.0, np.nan])
        assert metric_actor.mapper.scalar_range == (0.0, 1.0)
        assert background_actor.mapper.scalar_range == (100.0, 400.0)
        assert len(plotter.scalar_bars) == 2
        assert any("MAVM d−" in title for title in plotter.scalar_bars.keys())  # noqa: SIM118
        assert any(
            "Sim.:" in title and "degC" in title
            for title in plotter.scalar_bars.keys()  # noqa: SIM118
        )
        assert any(name.startswith("validation-labels") for name in plotter.actors)
        np.testing.assert_array_equal(sim_data.node_vars["temperature"][:, 1], scalars)
        if isinstance(sim_field, np.ndarray):
            np.testing.assert_array_equal(sim_field, scalars)
        np.testing.assert_array_equal(sim_data.coords, original_coords)
        np.testing.assert_array_equal(scalars, [100.0, 200.0, 300.0, 400.0])
        assert pv.global_theme.font.color == original_theme
        assert opts.colour_bar_lims is None
    finally:
        plotter.close()


def test_directional_and_zero_outputs_have_valid_colour_limits(
    metric_scene: tuple[valid.ValidResult, io.SimData],
) -> None:
    result, _ = metric_scene
    selectors = {valid.EValidResAxis.TIME: 0, valid.EValidResAxis.SIM_TRANS: 0}
    plotter = validvis.plot_valid_metric_on_sim(
        result,
        "temperature",
        valid.EMetricType.MAVM,
        selectors=selectors,
        output=valid.EMetricOutput.D_MINUS,
    )
    assert plotter.actors["validation-points"].mapper.scalar_range == (0.0, 3.0)
    plotter.close()
    plotter = validvis.plot_valid_metric_on_sim(
        result,
        "temperature",
        valid.EMetricType.MAVM,
        selectors=selectors,
        point_idxs=(0,),
        output=valid.EMetricOutput.D_MINUS,
    )
    assert plotter.actors["validation-points"].mapper.scalar_range == (0.0, 1.0)
    plotter.close()


def test_composition_preserves_existing_plotter_camera(
    metric_scene: tuple[valid.ValidResult, io.SimData],
) -> None:
    result, _ = metric_scene
    plotter = pv.Plotter(off_screen=True)
    plotter.camera_position = [(4.0, 2.0, 3.0), (0.5, 0.5, 0.5), (0.0, 0.0, 1.0)]
    camera = np.array(
        [plotter.camera.position, plotter.camera.focal_point, plotter.camera.up]
    )
    try:
        returned = validvis.plot_valid_metric_on_sim(
            result,
            "temperature",
            valid.EMetricType.MAVM,
            selectors={valid.EValidResAxis.TIME: 0, valid.EValidResAxis.SIM_TRANS: 0},
            plotter=plotter,
            output=valid.EMetricOutput.D_PLUS,
        )
        assert returned is plotter
        np.testing.assert_allclose(
            [plotter.camera.position, plotter.camera.focal_point, plotter.camera.up],
            camera,
        )
    finally:
        plotter.close()
