"""Analytic named-axis, conversion, and engine tests."""

from collections.abc import Callable

import numpy as np
import pytest

import pyvale.sensorsim as sens
import pyvale.valid.validpreparation as validprep
from pyvale import valid
from pyvale.dataio.simdata import SimData


def _domain(points: int = 2) -> dict[str, valid.ValidObservable]:
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
    valid_case = valid.ValidCase(_domain(), wrapped, wrapped)
    assert valid_case.sim["temperature"].data is samples
    with pytest.raises(ValueError, match="canonical"):
        valid.ValidArray(samples, axes[::-1])
    with pytest.raises(ValueError, match="unique"):
        valid.ValidArray(
            np.ones((2, 2)), (valid.EValidAxis.POINT, valid.EValidAxis.POINT)
        )


def test_engine_retains_sampled_epistemic_scenarios_and_offsets() -> None:
    # Two points, two transferable samples, three aleatory samples each.
    sim_data = np.array(
        [
            [[[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]],
            [[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]],
        ]
    )
    exp = np.array([[[1.0, 2.0, 3.0]], [[2.0, 3.0, 4.0]]])
    sim_axes = (
        valid.EValidAxis.POINT,
        valid.EValidAxis.COMPONENT,
        valid.EValidAxis.EPISTEMIC_TRANS,
        valid.EValidAxis.ALEATORY,
    )
    exp_axes = (
        valid.EValidAxis.POINT,
        valid.EValidAxis.COMPONENT,
        valid.EValidAxis.ALEATORY,
    )
    offsets = valid.EpistemicOffsets(
        lower=np.array([0.5, 1.0]),
        upper=np.array([1.5, 2.0]),
        axes=(valid.EValidAxis.POINT,),
    )
    valid_case = valid.ValidCase(
        _domain(),
        {"temperature": valid.ValidArray(sim_data, sim_axes)},
        {"temperature": valid.ValidArray(exp, exp_axes, offsets)},
    )
    result = valid.ValidEngine([valid.MetricAVM(), valid.MetricKS()]).run(valid_case)
    field = result.metrics["temperature"][valid.EMetricType.AVM]
    assert field.axes == ("point", "component", "time", "sim_trans", "exp_offset")
    assert field.outputs[valid.EMetricOutput.DISTANCE].shape == (2, 1, 1, 2, 3)
    maximum, where = result.find_extreme(
        "temperature", valid.EMetricType.AVM, output=valid.EMetricOutput.DISTANCE
    )
    assert maximum == pytest.approx(np.max(field.outputs[valid.EMetricOutput.DISTANCE]))
    assert (
        field.outputs[valid.EMetricOutput.DISTANCE][
            tuple(where[axis] for axis in field.axes)
        ]
        == maximum
    )
    for point in range(2):
        for trans in range(2):
            for offset in range(3):
                selected = valid_case.exp["temperature"].select_aleatory(
                    point, 0, 0, offset=valid.EEpisOffset(offset)
                )
                expected = valid.calc_avm_1d(sim_data[point, 0, trans], selected)
                assert field.outputs[valid.EMetricOutput.DISTANCE][
                    point, 0, 0, trans, offset
                ] == pytest.approx(expected)


def test_pointwise_ecdf_envelope_handles_crossing_distributions() -> None:
    # At x=0, first CDF is larger; at x=2, second CDF is larger.
    support, lower, upper = valid.calc_ecdf_envelope(
        [np.array([0.0, 3.0]), np.array([1.0, 2.0])]
    )
    np.testing.assert_array_equal(support, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(lower, [0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(upper, [0.5, 0.5, 1.0, 1.0])


def test_paired_rmse_rejects_unrelated_sample_lengths() -> None:
    metric = valid.MetricRMSE()
    assert metric.calc(np.array([0.0, 2.0]), np.array([1.0, 0.0])).value == (
        pytest.approx(np.sqrt(2.5))
    )
    with pytest.raises(ValueError, match="paired"):
        metric.calc(np.array([0.0, 1.0]), np.array([1.0]))


@pytest.mark.parametrize(
    "metric",
    (valid.calc_avm_1d, valid.calc_ks_1d, valid.calc_cvm_1d, valid.calc_mavm_1d),
)
def test_distribution_kernels_reject_implicit_axis_pooling(
    metric: Callable[[np.ndarray, np.ndarray], object],
) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        metric(np.ones((2, 3)), np.ones(3))


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
    observables = _domain(points=1)
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


def test_invalid_points_remain_masked_without_affecting_other_points() -> None:
    axes = (
        valid.EValidAxis.POINT,
        valid.EValidAxis.COMPONENT,
        valid.EValidAxis.ALEATORY,
    )
    sim_data = {
        "temperature": valid.ValidArray(
            np.array([[[1.0, 2.0]], [[np.nan, np.nan]]]), axes
        )
    }
    exp = {
        "temperature": valid.ValidArray(np.array([[[2.0, 3.0]], [[4.0, 5.0]]]), axes)
    }
    valid_case = valid.ValidCase(_domain(), sim_data, exp)
    field = (
        valid.ValidEngine([valid.MetricAVM()])
        .run(valid_case)
        .metrics["temperature"][valid.EMetricType.AVM]
    )
    assert field.outputs[valid.EMetricOutput.DISTANCE][0, 0, 0] == pytest.approx(1.0)
    assert np.isnan(field.outputs[valid.EMetricOutput.DISTANCE][1, 0, 0])


def test_both_sided_epistemic_samples_and_offsets_are_not_collapsed() -> None:
    observables = _domain(points=1)
    sim = valid.ValidArray(
        np.array([[[[0.0, 2.0], [2.0, 4.0]]]]),
        (
            valid.EValidAxis.POINT,
            valid.EValidAxis.COMPONENT,
            valid.EValidAxis.EPISTEMIC_TRANS,
            valid.EValidAxis.ALEATORY,
        ),
        valid.EpistemicOffsets(0.5, 1.0),
    )
    exp = valid.ValidArray(
        np.array([[[[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]]]]),
        (
            valid.EValidAxis.POINT,
            valid.EValidAxis.COMPONENT,
            valid.EValidAxis.EPISTEMIC_NUIS,
            valid.EValidAxis.ALEATORY,
        ),
        valid.EpistemicOffsets(0.25, 0.75),
    )
    valid_case = valid.ValidCase(
        observables,
        {"temperature": sim},
        {"temperature": exp},
    )
    field = (
        valid.ValidEngine([valid.MetricAVM()])
        .run(valid_case)
        .metrics["temperature"][valid.EMetricType.AVM]
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
    assert field.outputs[valid.EMetricOutput.DISTANCE].shape == (1, 1, 1, 2, 3, 3, 3)
    for st in range(2):
        for en in range(3):
            for so in range(3):
                for eo in range(3):
                    left = sim.select_aleatory(
                        0, 0, 0, epis_trans_idx=st, offset=valid.EEpisOffset(so)
                    )
                    right = exp.select_aleatory(
                        0, 0, 0, epis_nuis_idx=en, offset=valid.EEpisOffset(eo)
                    )
                    assert field.outputs[valid.EMetricOutput.DISTANCE][
                        0, 0, 0, st, en, so, eo
                    ] == pytest.approx(valid.calc_avm_1d(left, right))


def test_case_validates_sample_input_dimensions_for_each_source() -> None:
    """Both provenance fields must describe the source's uncertainty axes."""
    observables = _domain(points=1)
    axes = (
        valid.EValidAxis.POINT,
        valid.EValidAxis.COMPONENT,
        valid.EValidAxis.ALEATORY,
    )
    quantities = {"temperature": valid.ValidArray(np.ones((1, 1, 3)), axes)}
    mismatched_inputs = valid.ValidSampleInputs(
        run_ids=valid.ValidArray(
            np.array(["run_0", "run_1"]), (valid.EValidAxis.ALEATORY,)
        )
    )
    with pytest.raises(ValueError, match="sample input ALEATORY mismatch"):
        valid.ValidCase(
            observables, quantities, quantities, sim_inputs=mismatched_inputs
        )
    with pytest.raises(ValueError, match="sample input ALEATORY mismatch"):
        valid.ValidCase(
            observables, quantities, quantities, exp_inputs=mismatched_inputs
        )


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
        _domain(points=1), quantities, quantities, sim_inputs=sim_inputs
    )
    combined = validprep.concatenate_valid_cases(valid_case, valid_case)
    assert combined.sim_inputs is sim_inputs
    np.testing.assert_array_equal(combined.sim_inputs.run_ids.data, ["sim_0", "sim_1"])
    assert combined.exp["temperature"].data.shape == (1, 1, 4)


def _create_linear_sim(shift: float = 0.0) -> SimData:
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


def test_sample_bound_field_matches_linear_space_and_time() -> None:
    sim_data = _create_linear_sim()
    field = sens.FieldScalar(sim_data, "temperature", sens.EDim.TWOD)
    observable = valid.ValidObservable(
        components=("temperature",),
        geometry=valid.EValidGeometry.POINT,
        coords=np.array([[0.25, 0.5, 0.0], [0.75, 0.25, 0.0]]),
        times=np.array([0.25, 0.75]),
    )
    sampled = validprep.sample_field_for_valid(field, observable)
    expected = np.array([[2.0, 3.5], [2.0, 3.5]])[:, None, :]
    assert sampled.axes == (
        valid.EValidAxis.POINT,
        valid.EValidAxis.COMPONENT,
        valid.EValidAxis.TIME,
    )
    np.testing.assert_allclose(sampled.data, expected, rtol=0, atol=1e-12)


def test_sample_sweep_reuses_field_and_preserves_named_axes_and_globals() -> None:
    field = sens.FieldScalar(_create_linear_sim(), "temperature", sens.EDim.TWOD)
    times = np.array([0.25, 0.75])
    observables = {
        "temperature": valid.ValidObservable(
            components=("temperature",),
            geometry=valid.EValidGeometry.POINT,
            coords=np.array([[0.25, 0.5, 0.0]]),
            times=times,
        ),
        "mean_temperature": valid.ValidObservable(
            components=("mean_temperature",),
            geometry=valid.EValidGeometry.GLOBAL,
            times=times,
        ),
    }
    # Run order is deliberately different from tensor index order.
    indices = ((1, 2), (0, 1), (1, 0), (0, 2), (1, 1), (0, 0))
    runs = ((index, _create_linear_sim(10 * index[0] + index[1])) for index in indices)
    sampled = validprep.sample_sim_sweep(
        {"temperature": field},
        observables,
        runs,
        sample_axes=(valid.EValidAxis.EPISTEMIC_TRANS, valid.EValidAxis.ALEATORY),
        sample_shape=(2, 3),
    )
    shifts = 10 * np.arange(2)[:, None] + np.arange(3)
    expected = np.array([2.0, 3.5])[:, None, None] + shifts
    np.testing.assert_allclose(sampled["temperature"].data[0, 0], expected, atol=1e-12)
    np.testing.assert_allclose(
        sampled["mean_temperature"].data[0, 0], expected + 0.25, atol=1e-12
    )
    assert sampled["temperature"].axes[-2:] == (
        valid.EValidAxis.EPISTEMIC_TRANS,
        valid.EValidAxis.ALEATORY,
    )
    np.testing.assert_array_equal(
        field.get_sim_data().node_vars["temperature"],
        _create_linear_sim().node_vars["temperature"],
    )


@pytest.mark.parametrize(
    "indices, message",
    [(((0,), (0,)), "Duplicate"), (((0,),), "incomplete"), (((2,),), "outside")],
)
def test_sample_sweep_rejects_invalid_run_tensor(
    indices: tuple[tuple[int, ...], ...], message: str
) -> None:
    field = sens.FieldScalar(_create_linear_sim(), "temperature", sens.EDim.TWOD)
    observable = valid.ValidObservable(
        components=("temperature",),
        geometry=valid.EValidGeometry.POINT,
        coords=np.array([[0.25, 0.5, 0.0]]),
        times=np.array([0.5]),
    )
    with pytest.raises(ValueError, match=message):
        validprep.sample_sim_sweep(
            {"temperature": field},
            {"temperature": observable},
            ((index, _create_linear_sim()) for index in indices),
            sample_axes=(valid.EValidAxis.ALEATORY,),
            sample_shape=(2,),
        )
