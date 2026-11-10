"""Analytic named-axis, conversion, and engine tests."""

from collections.abc import Callable

import numpy as np
import pytest

import pyvale.valid as val


def _domain(points: int = 2) -> val.ValDomain:
    coords = np.column_stack(
        (np.arange(points), np.zeros(points), np.zeros(points))
    ).astype(float)
    return val.ValDomain(
        {
            "temperature": val.ValObservable(
                val.ValGeometry(val.EValGeometry.POINT, coords),
                val.EValOperator.SAMPLE,
                ("temperature",),
                None,
                "degC",
            )
        }
    )


def test_arrays_are_zero_copy_and_axis_order_is_checked() -> None:
    samples = np.arange(12.0).reshape(2, 1, 2, 3)
    axes = (
        val.EValAxis.POINT,
        val.EValAxis.COMPONENT,
        val.EValAxis.EPISTEMIC_TRANS,
        val.EValAxis.ALEATORY,
    )
    wrapped = val.val_from_arrays({"temperature": val.ValArray(samples, axes)})
    assert wrapped.quantities["temperature"].data is samples
    with pytest.raises(ValueError, match="canonical"):
        val.ValArray(samples, axes[::-1])
    with pytest.raises(ValueError, match="unique"):
        val.ValArray(np.ones((2, 2)), (val.EValAxis.POINT, val.EValAxis.POINT))


def test_engine_retains_sampled_epistemic_scenarios_and_offsets() -> None:
    # Two points, two transferable samples, three aleatory samples each.
    simulation = np.array(
        [
            [[[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]],
            [[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]],
        ]
    )
    experiment = np.array([[[1.0, 2.0, 3.0]], [[2.0, 3.0, 4.0]]])
    sim_axes = (
        val.EValAxis.POINT,
        val.EValAxis.COMPONENT,
        val.EValAxis.EPISTEMIC_TRANS,
        val.EValAxis.ALEATORY,
    )
    exp_axes = (val.EValAxis.POINT, val.EValAxis.COMPONENT, val.EValAxis.ALEATORY)
    offsets = val.EpistemicOffsets(
        lower=np.array([0.5, 1.0]),
        upper=np.array([1.5, 2.0]),
        axes=(val.EValAxis.POINT,),
    )
    val_case = val.ValCase(
        _domain(),
        val.ValData({"temperature": val.ValArray(simulation, sim_axes)}),
        val.ValData({"temperature": val.ValArray(experiment, exp_axes, offsets)}),
    )
    result = val.ValidationEngine([val.MetricAVM(), val.MetricKS()]).run(val_case)
    field = result.get("temperature", val.EMetricType.AVM)
    assert field.axes == ("point", "component", "time", "sim_trans", "exp_offset")
    assert field.metric.shape == (2, 1, 1, 2, 3)
    maximum, where = result.find_extreme("temperature", val.EMetricType.AVM)
    assert maximum == pytest.approx(np.max(field.metric))
    assert field.metric[tuple(where[axis] for axis in field.axes)] == maximum
    for point in range(2):
        for trans in range(2):
            for offset in range(3):
                selected = val_case.exp.quantities["temperature"].select_aleatory(
                    point, 0, 0, offset=offset
                )
                expected = val.calc_avm_1d(simulation[point, 0, trans], selected)
                assert field.metric[point, 0, 0, trans, offset] == pytest.approx(
                    expected
                )


def test_pointwise_ecdf_envelope_handles_crossing_distributions() -> None:
    # At x=0, first CDF is larger; at x=2, second CDF is larger.
    support, lower, upper = val.calc_ecdf_envelope(
        [np.array([0.0, 3.0]), np.array([1.0, 2.0])]
    )
    np.testing.assert_array_equal(support, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(lower, [0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(upper, [0.5, 0.5, 1.0, 1.0])


def test_paired_rmse_rejects_unrelated_sample_lengths() -> None:
    metric = val.MetricRMSE()
    assert metric.calc(np.array([0.0, 2.0]), np.array([1.0, 0.0])).value == (
        pytest.approx(np.sqrt(2.5))
    )
    with pytest.raises(ValueError, match="paired"):
        metric.calc(np.array([0.0, 1.0]), np.array([1.0]))


@pytest.mark.parametrize(
    "metric", (val.calc_avm_1d, val.calc_ks_1d, val.calc_cvm_1d, val.calc_mavm_1d)
)
def test_distribution_kernels_reject_implicit_axis_pooling(
    metric: Callable[[np.ndarray, np.ndarray], object],
) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        metric(np.ones((2, 3)), np.ones(3))


def test_combine_cases_preserves_transient_run_pairing() -> None:
    coords = np.array([[0.0, 0.0, 0.0]])
    times = np.array([0.0, 1.0])
    domain = val.ValDomain(
        {
            "temperature": val.ValObservable(
                val.ValGeometry(val.EValGeometry.POINT, coords),
                val.EValOperator.SAMPLE,
                ("temperature",),
                times,
                "degC",
            )
        }
    )
    axes = (val.EValAxis.POINT, val.EValAxis.COMPONENT, val.EValAxis.TIME)
    sim = val.ValData({"temperature": val.ValArray(np.zeros((1, 1, 2)), axes)})
    cases = [
        val.ValCase(
            domain,
            sim,
            val.ValData(
                {"temperature": val.ValArray(np.array([[[i, i + 10.0]]]), axes)}
            ),
        )
        for i in (1.0, 2.0)
    ]
    joined = val.combine_val_cases(*cases)
    np.testing.assert_array_equal(
        joined.exp.quantities["temperature"].data, [[[[1.0, 2.0], [11.0, 12.0]]]]
    )
    np.testing.assert_array_equal(
        joined.exp.sample_inputs.run_ids.data, ["case_0", "case_1"]
    )


def test_combine_cases_preserves_parameter_provenance() -> None:
    domain = _domain(points=1)
    axes = (val.EValAxis.POINT, val.EValAxis.COMPONENT)
    sim = val.ValData({"temperature": val.ValArray(np.array([[0.0]]), axes)})
    cases = []
    for index, parameter in enumerate((10.0, 20.0)):
        inputs = val.ValSampleInputs(
            parameters={"loading": val.ValArray(np.array(parameter), ())}
        )
        exp = val.ValData(
            {"temperature": val.ValArray(np.array([[parameter]]), axes)}, inputs
        )
        cases.append(val.ValCase(domain, sim, exp))
    joined = val.combine_val_cases(*cases)
    np.testing.assert_array_equal(
        joined.exp.sample_inputs.parameters["loading"].data, [10.0, 20.0]
    )


def test_sensor_sim_bridge_retains_component_and_run_values() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    values = np.arange(16.0).reshape(2, 2, 2, 2)
    output = {
        ("plate", "strain", "meas"): values,
        ("plate", "strain", "sens_times"): np.array([0.0, 1.0]),
    }
    converted = val.conv_exp_sim_to_data(
        output, sim_key="plate", selections={"strain_xy": ("strain", 1, coords)}
    )
    assert len(converted) == 2
    np.testing.assert_array_equal(converted[1]["strain_xy"].fields, values[1, :, 1, :])


def test_invalid_points_remain_masked_without_affecting_other_points() -> None:
    axes = (val.EValAxis.POINT, val.EValAxis.COMPONENT, val.EValAxis.ALEATORY)
    simulation = val.ValData(
        {
            "temperature": val.ValArray(
                np.array([[[1.0, 2.0]], [[np.nan, np.nan]]]), axes
            )
        }
    )
    experiment = val.ValData(
        {"temperature": val.ValArray(np.array([[[2.0, 3.0]], [[4.0, 5.0]]]), axes)}
    )
    val_case = val.ValCase(_domain(), simulation, experiment)
    field = (
        val.ValidationEngine([val.MetricAVM()])
        .run(val_case)
        .get("temperature", val.EMetricType.AVM)
    )
    assert field.metric[0, 0, 0] == pytest.approx(1.0)
    assert np.isnan(field.metric[1, 0, 0])


def test_both_sided_epistemic_samples_and_offsets_are_not_collapsed() -> None:
    domain = _domain(points=1)
    sim = val.ValArray(
        np.array([[[[0.0, 2.0], [2.0, 4.0]]]]),
        (
            val.EValAxis.POINT,
            val.EValAxis.COMPONENT,
            val.EValAxis.EPISTEMIC_TRANS,
            val.EValAxis.ALEATORY,
        ),
        val.EpistemicOffsets(0.5, 1.0),
    )
    exp = val.ValArray(
        np.array([[[[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]]]]),
        (
            val.EValAxis.POINT,
            val.EValAxis.COMPONENT,
            val.EValAxis.EPISTEMIC_NUIS,
            val.EValAxis.ALEATORY,
        ),
        val.EpistemicOffsets(0.25, 0.75),
    )
    val_case = val.ValCase(
        domain,
        val.ValData({"temperature": sim}),
        val.ValData({"temperature": exp}),
    )
    field = (
        val.ValidationEngine([val.MetricAVM()])
        .run(val_case)
        .get("temperature", val.EMetricType.AVM)
    )
    assert field.axes == (
        "point",
        "component",
        "time",
        "sim_trans",
        "exp_nuis",
        "sim_offset",
        "exp_offset",
    )
    assert field.metric.shape == (1, 1, 1, 2, 3, 3, 3)
    for st in range(2):
        for en in range(3):
            for so in range(3):
                for eo in range(3):
                    left = sim.select_aleatory(0, 0, 0, epis_trans=st, offset=so)
                    right = exp.select_aleatory(0, 0, 0, epis_nuis=en, offset=eo)
                    assert field.metric[0, 0, 0, st, en, so, eo] == pytest.approx(
                        val.calc_avm_1d(left, right)
                    )
