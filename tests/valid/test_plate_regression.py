"""Golden and conversion regressions for the packaged 2D plate."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from plate_fixture import load_plate_case

import pyvale.data as dataset
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.valid.validpreparation as validprep
from pyvale import valid
from pyvale.valid import validplot, validvis


def test_prepared_plate_gold_metrics() -> None:
    """Track the saved SensorSim fixture and selected metric diagnostics."""
    valid_case = load_plate_case()
    result = valid.ValidEngine([valid.MetricAVM(), valid.MetricMAVM()]).run(valid_case)
    assert result.metrics["temperature"][valid.EMetricType.AVM].outputs[
        valid.EMetricOutput.DISTANCE
    ][5, 0, 0] == (pytest.approx(0.26243433506330643, abs=1e-10))
    assert (
        result.metrics["temperature"][valid.EMetricType.MAVM].outputs[
            valid.EMetricOutput.D_PLUS
        ][5, 0, 0]
        + result.metrics["temperature"][valid.EMetricType.MAVM].outputs[
            valid.EMetricOutput.D_MINUS
        ][5, 0, 0]
    ) == (pytest.approx(0.4891546824953229, abs=1e-10))
    assert result.metrics["strain"][valid.EMetricType.AVM].outputs[
        valid.EMetricOutput.DISTANCE
    ][5, 0, 0] == (pytest.approx(1.2018068336589567e-6, abs=1e-12))
    assert (
        result.metrics["strain"][valid.EMetricType.MAVM].outputs[
            valid.EMetricOutput.D_PLUS
        ][5, 2, 0]
        + result.metrics["strain"][valid.EMetricType.MAVM].outputs[
            valid.EMetricOutput.D_MINUS
        ][5, 2, 0]
    ) == (pytest.approx(3.368761032966909e-6, abs=1e-12))


def test_real_simdata_expdata_and_direct_arrays_agree() -> None:
    """The field and acquisition converters match an explicit NumPy case."""
    with np.load(dataset.valid_data_dir() / "plate_sensorsim_2d.npz") as saved:
        coords = saved["coords"].copy()
        times = saved["times"].copy()
        measured = saved["exp_temp"].copy()

    observables = {
        "temperature": valid.ValidObservable(
            components=("temperature",),
            geometry=valid.EValidGeometry.POINT,
            coords=coords,
            times=times,
            phys_units="degC",
        )
    }
    nominal_path = dataset.thermomechanical_2d_experiment_paths()[0]
    sim_data = mh.ExodusLoader(nominal_path).load_all_sim_data()
    sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y"))
    sim = validprep.sample_sim_sweep(
        {"temperature": sens.FieldScalar(sim_data, "temperature", sens.EDim.TWOD)},
        observables,
        [((), sim_data)],
    )

    output = {
        ("perturbed", "temp", "meas"): measured,
        ("perturbed", "temp", "sens_times"): times,
    }
    acquisitions = validprep.extract_exp_sim_data(
        output,
        sim_key="perturbed",
        selections={"temperature": validprep.SensorOutputSelection("temp", 0, coords)},
    )
    cases = [
        valid.ValidCase(
            observables, sim, validprep.assemble_exp_arrays(run, observables)
        )
        for run in acquisitions
    ]
    converted = validprep.concatenate_valid_cases(*cases)

    direct_sim = {
        "temperature": valid.ValidArray(
            sim["temperature"].data,
            sim["temperature"].axes,
        )
    }
    direct_exp = {
        "temperature": valid.ValidArray(
            np.moveaxis(measured, 0, -1),
            (
                valid.EValidAxis.POINT,
                valid.EValidAxis.COMPONENT,
                valid.EValidAxis.TIME,
                valid.EValidAxis.ALEATORY,
            ),
        )
    }
    direct = valid.ValidCase(observables, direct_sim, direct_exp)
    np.testing.assert_array_equal(
        converted.exp["temperature"].data,
        direct.exp["temperature"].data,
    )
    engine = valid.ValidEngine([valid.MetricAVM(), valid.MetricMAVM()])
    converted_result = engine.run(converted)
    direct_result = engine.run(direct)
    for metric in (valid.EMetricType.AVM, valid.EMetricType.MAVM):
        expected = direct_result.metrics["temperature"][metric]
        actual = converted_result.metrics["temperature"][metric]
        for output in expected.outputs:
            np.testing.assert_allclose(
                actual.outputs[output], expected.outputs[output], rtol=0.0, atol=1e-12
            )


@pytest.mark.parametrize("spatial_averager", [None, sens.EIntSpatialType.RECT4PT])
def test_global_and_sensor_sim_extraction(
    spatial_averager: sens.EIntSpatialType | None,
) -> None:
    """Reuse SensorSim's averaging and existing MOOSE global output."""
    sim_path = dataset.thermomechanical_2d_experiment_paths()[0]
    sim_data = mh.ExodusLoader(sim_path).load_all_sim_data()
    sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y"))
    coords = np.array([[50.0, 25.0, 0.0]])
    times = np.array([0.0, 100.0, 200.0])
    average_sensor = sens.SensorData(
        positions=coords,
        sample_times=times,
        spatial_averager=spatial_averager,
        spatial_dims=np.array([5.0, 5.0, 0.0]),
    )
    observables = {
        "temperature_average": valid.ValidObservable(
            components=("temperature",),
            geometry=valid.EValidGeometry.POINT,
            coords=coords,
            times=times,
            phys_units="degC",
        ),
        "global_temperature": valid.ValidObservable(
            components=("temp_avg",),
            geometry=valid.EValidGeometry.GLOBAL,
            coords=None,
            times=times,
            phys_units="degC",
        ),
    }
    field = sens.FieldScalar(sim_data, "temperature", sens.EDim.TWOD)
    result = validprep.sample_sim_sweep(
        {
            "temperature_average": sens.FieldScalar(
                sim_data, "temperature", sens.EDim.TWOD
            ),
        },
        observables,
        [((), sim_data)],
        sensor_data={"temperature_average": average_sensor},
    )
    expected_average = sens.sample_field_with_sensor_data(field, average_sensor)
    np.testing.assert_allclose(
        result["temperature_average"].data,
        expected_average,
    )
    np.testing.assert_allclose(
        result["global_temperature"].data[0, 0],
        np.interp(times, sim_data.time, sim_data.glob_vars["temp_avg"]),
    )


def test_diagnostic_plots_show_units_and_registered_axes() -> None:
    valid_case = load_plate_case(transient=True)
    result = valid.ValidEngine([valid.MetricAVM(), valid.MetricMAVM()]).run(valid_case)
    ecdf_plot: validplot.ValidECDFPlot = validplot.plot_sim_exp_cdfs(
        valid_case,
        "temperature",
        point_idxs=(5,),
        time_idx=4,
    )
    (fig_cdf, axes_cdf) = ecdf_plot.figure, ecdf_plot.axes
    assert "degC" in axes_cdf[0, 0].get_xlabel()
    assert len(axes_cdf[0, 0].lines) == 2
    pv_plot = validvis.plot_valid_metric_on_sim(
        result,
        "temperature",
        valid.EMetricType.AVM,
        selectors={valid.EValidResAxis.TIME: 4},
    )
    assert pv_plot.actors["validation-points"].mapper.dataset.n_points == 12
    pv_plot.close()
    fig_time, ax_time = validplot.plot_valid_metric_time(
        result,
        "temperature",
        valid.EMetricType.AVM,
        selectors={valid.EValidResAxis.POINT: 5},
    )
    np.testing.assert_array_equal(
        ax_time.lines[0].get_xdata(), valid_case.observables["temperature"].times
    )
    assert ax_time.get_xlabel() == "Time [s]"
    plt.close(fig_cdf)
    plt.close(fig_time)


def test_spatial_temporal_offset_is_applied_per_conditional_cdf() -> None:
    valid_case = load_plate_case(transient=True, epistemic=True, offsets=True)
    quantity = valid_case.exp["temperature"]
    assert quantity.offsets.axes == (valid.EValidAxis.POINT, valid.EValidAxis.TIME)
    nominal = quantity.select_aleatory(5, 0, 0)
    lower_early = quantity.select_aleatory(5, 0, 0, offset=valid.EEpisOffset.LOWER)
    lower_late = quantity.select_aleatory(5, 0, 4, offset=valid.EEpisOffset.LOWER)
    late_nominal = quantity.select_aleatory(5, 0, 4)
    assert np.allclose(nominal - lower_early, quantity.offsets.lower[5, 0])
    assert np.allclose(late_nominal - lower_late, quantity.offsets.lower[5, 4])
    assert quantity.offsets.lower[5, 4] > quantity.offsets.lower[5, 0]


@pytest.mark.parametrize(
    ("filename", "transient", "epistemic"),
    (
        ("plate_validation_2d.npz", True, True),
        ("plate_validation_2d_nominal.npz", True, False),
        ("plate_validation_2d_steady.npz", False, True),
        ("plate_validation_2d_steady_nominal.npz", False, False),
    ),
)
def test_prepared_plate_arrays_preserve_sensor_observations(
    filename: str, transient: bool, epistemic: bool
) -> None:
    """Prepared axes and model scenarios preserve each original repeat exactly."""
    data_path = dataset.valid_data_dir()
    with (
        np.load(data_path / "plate_sensorsim_2d.npz") as raw,
        np.load(data_path / filename) as prepared,
    ):
        assert set(prepared.files) == {
            "coords",
            "sim_temp",
            "exp_temp",
            "sim_strain",
            "exp_strain",
        } | ({"times"} if transient else set())
        np.testing.assert_array_equal(prepared["coords"], raw["coords"])
        if transient:
            np.testing.assert_array_equal(prepared["times"], raw["times"])
        for quantity, components in (("temp", 1), ("strain", 3)):
            physical_shape = (12, components, 5) if transient else (12, components)
            expected_sim_shape = (
                (*physical_shape, 2, 6) if epistemic else (*physical_shape, 6)
            )
            assert prepared[f"sim_{quantity}"].shape == expected_sim_shape
            assert prepared[f"exp_{quantity}"].shape == (*physical_shape, 6)
            for repeat in range(6):
                nominal = raw[f"sim_{quantity}"][repeat]
                perturbed = raw[f"exp_{quantity}"][repeat]
                if not transient:
                    nominal = nominal[..., -1]
                    perturbed = perturbed[..., -1]
                np.testing.assert_array_equal(
                    prepared[f"exp_{quantity}"][..., repeat], perturbed
                )
                if epistemic:
                    np.testing.assert_array_equal(
                        prepared[f"sim_{quantity}"][..., 0, repeat], nominal
                    )
                    np.testing.assert_array_equal(
                        prepared[f"sim_{quantity}"][..., 1, repeat], perturbed
                    )
                else:
                    np.testing.assert_array_equal(
                        prepared[f"sim_{quantity}"][..., repeat], nominal
                    )
