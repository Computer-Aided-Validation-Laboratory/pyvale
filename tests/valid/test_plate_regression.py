"""Golden and conversion regressions for the packaged 2D plate."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pyvale.data as dataset
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.valid as val
from pyvale.examples.valid.tools import load_plate_case


def test_prepared_plate_gold_metrics() -> None:
    """Track the saved SensorSim fixture and selected metric diagnostics."""
    case = load_plate_case()
    result = val.ValidationEngine([val.MetricAVM(), val.MetricMAVM()]).run(case)
    assert result.get("temperature", val.EMetricType.AVM).value[5, 0, 0] == (
        pytest.approx(0.26243433506330643, abs=1e-10)
    )
    assert result.get("temperature", val.EMetricType.MAVM).value[5, 0, 0] == (
        pytest.approx(0.4891546824953229, abs=1e-10)
    )
    assert result.get("strain", val.EMetricType.AVM).value[5, 0, 0] == (
        pytest.approx(1.2018068336589567e-6, abs=1e-12)
    )
    assert result.get("strain", val.EMetricType.MAVM).value[5, 2, 0] == (
        pytest.approx(3.368761032966909e-6, abs=1e-12)
    )


def test_real_simdata_expdata_and_direct_arrays_agree() -> None:
    """The field and acquisition converters match an explicit NumPy case."""
    with np.load(dataset.valid_data_dir() / "plate_validation_2d.npz") as saved:
        coords = saved["coords"].copy()
        times = saved["times"].copy()
        measured = saved["perturbed_temperature"].copy()

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
    nominal_path = dataset.thermomechanical_2d_experiment_paths()[0]
    sim_data = mh.ExodusLoader(nominal_path).load_all_sim_data()
    sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y"))
    field = sens.FieldScalar(sim_data, "temperature", sens.EDim.TWOD)
    sim = val.val_from_sim({"temperature": field}, domain, [((), sim_data)])

    output = {
        ("perturbed", "temp", "meas"): measured,
        ("perturbed", "temp", "sens_times"): times,
    }
    acquisitions = val.conv_exp_sim_to_data(
        output,
        sim_key="perturbed",
        selections={"temperature": ("temp", 0, coords)},
    )
    cases = [
        val.ValCase(domain, sim, val.val_from_exp(run, domain)) for run in acquisitions
    ]
    converted = val.combine_val_cases(*cases)

    direct_sim = val.val_from_arrays(
        {
            "temperature": val.ValArray(
                sim.quantities["temperature"].data,
                sim.quantities["temperature"].axes,
            )
        }
    )
    direct_exp = val.val_from_arrays(
        {
            "temperature": val.ValArray(
                np.moveaxis(measured, 0, -1),
                (
                    val.EValAxis.POINT,
                    val.EValAxis.COMPONENT,
                    val.EValAxis.TIME,
                    val.EValAxis.ALEATORY,
                ),
            )
        }
    )
    direct = val.ValCase(domain, direct_sim, direct_exp)
    np.testing.assert_array_equal(
        converted.exp.quantities["temperature"].data,
        direct.exp.quantities["temperature"].data,
    )
    engine = val.ValidationEngine([val.MetricAVM(), val.MetricMAVM()])
    converted_result = engine.run(converted)
    direct_result = engine.run(direct)
    for metric in (val.EMetricType.AVM, val.EMetricType.MAVM):
        np.testing.assert_allclose(
            converted_result.get("temperature", metric).value,
            direct_result.get("temperature", metric).value,
            rtol=0.0,
            atol=1e-12,
        )


def test_global_and_spatial_average_simulation_extraction() -> None:
    """Reuse SensorSim's averaging and existing MOOSE global output."""
    sim_path = dataset.thermomechanical_2d_experiment_paths()[0]
    sim_data = mh.ExodusLoader(sim_path).load_all_sim_data()
    sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y"))
    coords = np.array([[50.0, 25.0, 0.0]])
    times = np.array([0.0, 100.0, 200.0])
    average_sensor = sens.SensorData(
        positions=coords,
        sample_times=times,
        spatial_averager=sens.EIntSpatialType.RECT4PT,
        spatial_dims=np.array([5.0, 5.0, 0.0]),
    )
    domain = val.ValDomain(
        {
            "temperature_average": val.ValObservable(
                val.ValGeometry(val.EValGeometry.POINT, coords),
                val.EValOperator.MEAN,
                ("temperature",),
                times,
                "degC",
            ),
            "global_temperature": val.ValObservable(
                val.ValGeometry(val.EValGeometry.GLOBAL, None),
                val.EValOperator.MEAN,
                ("temp_avg",),
                times,
                "degC",
            ),
        }
    )
    field = sens.FieldScalar(sim_data, "temperature", sens.EDim.TWOD)
    result = val.val_from_sim(
        {"temperature_average": field, "global_temperature": None},
        domain,
        [((), sim_data)],
        sensor_data={"temperature_average": average_sensor},
    )
    expected_average = sens.sample_field_with_sensor_data(field, average_sensor)
    np.testing.assert_allclose(
        result.quantities["temperature_average"].data,
        expected_average,
    )
    np.testing.assert_allclose(
        result.quantities["global_temperature"].data[0, 0],
        np.interp(times, sim_data.time, sim_data.glob_vars["temp_avg"]),
    )


def test_diagnostic_plots_show_units_and_registered_axes() -> None:
    case = load_plate_case(transient=True)
    result = val.ValidationEngine([val.MetricAVM(), val.MetricMAVM()]).run(case)
    fig_cdf, ax_cdf = val.plot_val_cdfs(
        case,
        result,
        "temperature",
        point=5,
        time=4,
        metric=val.MetricMAVM(),
    )
    assert "degC" in ax_cdf.get_xlabel()
    assert len(ax_cdf.lines) == 4
    fig_map, ax_map = val.plot_val_metric_map(
        result,
        "temperature",
        val.EMetricType.AVM,
        selectors={"time": 4},
    )
    assert "mm" in ax_map.get_xlabel()
    assert len(ax_map.collections[0].get_offsets()) == 12
    fig_time, ax_time = val.plot_val_metric_time(
        result,
        "temperature",
        val.EMetricType.AVM,
        selectors={"point": 5},
    )
    np.testing.assert_array_equal(
        ax_time.lines[0].get_xdata(), case.domain.observables["temperature"].times
    )
    assert ax_time.get_xlabel() == "Time [s]"
    plt.close(fig_cdf)
    plt.close(fig_map)
    plt.close(fig_time)


def test_spatial_temporal_offset_is_applied_per_conditional_cdf() -> None:
    case = load_plate_case(transient=True, epistemic=True, offsets=True)
    quantity = case.exp.quantities["temperature"]
    assert quantity.offsets.axes == (val.EValAxis.POINT, val.EValAxis.TIME)
    nominal = val.select_aleatory(quantity, 5, 0, 0)
    lower_early = val.select_aleatory(
        quantity, 5, 0, 0, offset=val.EEpisOffset.LOWER
    )
    lower_late = val.select_aleatory(
        quantity, 5, 0, 4, offset=val.EEpisOffset.LOWER
    )
    late_nominal = val.select_aleatory(quantity, 5, 0, 4)
    assert np.allclose(nominal - lower_early, quantity.offsets.lower[5, 0])
    assert np.allclose(late_nominal - lower_late, quantity.offsets.lower[5, 4])
    assert quantity.offsets.lower[5, 4] > quantity.offsets.lower[5, 0]
