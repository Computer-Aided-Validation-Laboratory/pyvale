"""Validation regression behaviour."""

import runpy
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from plate_fixture import load_plate_case
from scipy import stats

import pyvale.data as dataset
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.validation as vl
from pyvale.validation.metrics import EMAVMDuplicates, calc_mavm_1d


def create_fullfield_reference_cases() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Reproduce the eight inputs for the frozen fullfieldvalmetrics reference."""
    cases: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    num_exp = 4
    num_sim = 2
    exp_data = np.arange(num_exp, dtype=np.float64) - num_exp / 2.0
    sim_data = exp_data[num_exp - num_sim :] + exp_data[-1] + 1.0
    cases["test01"] = (sim_data, exp_data)
    num_exp = 2
    num_sim = 4
    sim_data = np.arange(num_sim, dtype=np.float64) - num_sim / 2.0
    exp_data = sim_data[num_sim - num_exp :] + sim_data[-1] + 2.0
    cases["test02"] = (sim_data, exp_data)
    num_exp = 4
    num_sim = 2
    exp_data = np.arange(num_exp, dtype=np.float64) - num_exp / 2.0
    sim_data = np.arange(num_sim, dtype=np.float64) - num_sim / 2.0
    cases["test03"] = (sim_data, exp_data)
    num_exp = 4
    num_sim = 3
    exp_data = np.arange(num_exp, dtype=np.float64) - num_exp / 2.0
    sim_data = exp_data[num_exp - num_sim :] + exp_data[-1] + 1.0
    cases["test04"] = (sim_data, exp_data)
    num_exp = 3
    num_sim = 4
    sim_data = np.arange(num_sim, dtype=np.float64) - num_sim / 2.0
    exp_data = sim_data[num_sim - num_exp :] + sim_data[-1] + 2.0
    cases["test05"] = (sim_data, exp_data)
    num_exp = 4
    num_sim = 3
    exp_data = np.arange(num_exp, dtype=np.float64) - num_exp / 2.0
    sim_data = np.arange(num_sim, dtype=np.float64) - num_sim / 2.0
    cases["test06"] = (sim_data, exp_data)
    num_exp = 100
    num_sim = 10000
    offset = num_sim
    exp_data = np.arange(num_exp, dtype=np.float64) - num_exp / 2.0
    exp_data += 1.25 * offset
    sim_data = np.arange(num_sim, dtype=np.float64) - num_sim / 2.0 + offset
    cases["test07"] = (sim_data, exp_data)
    generator = np.random.RandomState(2025)
    exp_data = generator.uniform(low=230.95, high=237.16, size=25)
    exp_data = np.repeat(exp_data, 4)
    sim_data = generator.normal(loc=238.0, scale=4.0, size=10000)
    cases["test08"] = (sim_data, exp_data)
    return cases


@pytest.mark.parametrize(
    ("case_name", "expected_d_plus", "expected_d_minus"),
    [
        ("test01", 0.0, 4.315570153994818),
        ("test02", 6.232248255376777, 0.0),
        ("test03", 1.315570153994818, 1.315570153994818),
        ("test04", 0.0, 3.815570153994818),
        ("test05", 4.87649440322337, 0.0),
        ("test06", 1.315570153994818, 1.315570153994818),
        ("test07", 2806.765551246437, 304.34465910224606),
        ("test08", 0.1125130844472775, 4.317398787045127),
    ],
)
def test_mavm_matches_fullfield_analytic_cases(
    case_name: str, expected_d_plus: float, expected_d_minus: float
) -> None:
    """Match fullfieldvalmetrics MAVM values to floating-point tolerance."""
    model_data, exp_data = create_fullfield_reference_cases()[case_name]
    result = calc_mavm_1d(model_data, exp_data, duplicates=EMAVMDuplicates.INCLUDE)
    assert result.d_plus == pytest.approx(expected_d_plus, abs=1e-10)
    assert result.d_minus == pytest.approx(expected_d_minus, abs=1e-10)


def test_mavm_accepts_strided_samples_and_filters_nan() -> None:
    model = np.array([0.0, 99.0, np.nan, 99.0, 2.0, 99.0])[::2]
    experimental = np.array([1.0, 99.0, 3.0, 99.0])[::2]
    calculated = calc_mavm_1d(model, experimental)
    expected = calc_mavm_1d(np.array([0.0, 2.0]), np.array([1.0, 3.0]))
    assert calculated.d_plus == pytest.approx(expected.d_plus)
    assert calculated.d_minus == pytest.approx(expected.d_minus)
    np.testing.assert_array_equal(calculated.model_quantiles, [0.0, 2.0])


@pytest.mark.parametrize("workers", [1])
def test_engine_matches_standalone_uncertainty_products(workers: int) -> None:
    generator = np.random.default_rng(1729)
    sim_data = generator.integers(-3, 4, (1, 1, 1, 2, 2, 8)).astype(float)
    exp = generator.integers(-2, 5, (1, 1, 1, 2, 2, 5)).astype(float)
    sim_data[..., 0, 0, 0] = np.nan
    exp[..., 1, 0, :] = np.nan
    exp[..., 0, 1, 0] = np.nan
    axes = tuple(vl.EValidAxis)
    simulated = vl.ValidArray(sim_data, axes, vl.EpistemicOffsets(0.25, 0.5))
    observed = vl.ValidArray(exp, axes, vl.EpistemicOffsets(0.4, 0.6))
    observables = {
        "response": vl.ValidObservable(
            components=("response",),
            geometry=vl.EValidGeometry.POINT,
            coords=np.zeros((1, 3)),
            times=np.array([0.0]),
        )
    }
    metrics = [
        vl.MetricMAVM(),
        vl.MetricAVM(),
        vl.MetricKS(),
        vl.MetricCVM(),
    ]
    result = vl.ValidEngine(
        metrics, opts=vl.ValidEngineOpts(workers=workers)
    ).run(vl.ValidCase(observables, {"response": simulated}, {"response": observed}))
    for sim_trans, sim_nuis, exp_trans, exp_nuis, sim_offset, exp_offset in product(
        range(2), range(2), range(2), range(2), vl.EEpisOffset, vl.EEpisOffset
    ):
        sim_samples = simulated.select_aleatory(
            0, 0, 0, epis_trans_idx=sim_trans, epis_nuis_idx=sim_nuis, offset=sim_offset
        )
        exp_samples = observed.select_aleatory(
            0, 0, 0, epis_trans_idx=exp_trans, epis_nuis_idx=exp_nuis, offset=exp_offset
        )
        selection = (
            0,
            0,
            0,
            sim_trans,
            sim_nuis,
            exp_trans,
            exp_nuis,
            sim_offset,
            exp_offset,
        )
        for metric in metrics:
            field = result.metrics["response"][metric.get_kind()]
            assert tuple(field.outputs) == metric.get_outputs()
            if not np.any(np.isfinite(exp_samples)):
                for array in field.outputs.values():
                    assert np.isnan(array[selection])
                continue
            expected = metric.calc(sim_samples, exp_samples)
            expected_outputs = metric.extract_outputs(expected)
            for output, expected_value in zip(
                metric.get_outputs(), expected_outputs, strict=True
            ):
                assert field.outputs[output][selection] == pytest.approx(
                    expected_value, rel=1e-13, abs=1e-12
                )


@pytest.mark.parametrize("sim_count,exp_count", [(11, 29)])
@pytest.mark.parametrize("repeated", [False, True])
def test_distribution_statistics_match_scipy(
    sim_count: int, exp_count: int, *, repeated: bool
) -> None:
    generator = np.random.default_rng(8921)
    if repeated:
        sim_data = generator.integers(-2, 4, sim_count).astype(float)
        exp = generator.integers(-1, 5, exp_count).astype(float)
    else:
        sim_data = generator.normal(0.0, 1.0, sim_count)
        exp = generator.normal(0.3, 1.1, exp_count)
    for sim_samples, exp_samples in (
        (sim_data, exp),
        (exp[::-1], sim_data[::-1]),
        (np.zeros(sim_count), np.zeros(exp_count)),
        (np.zeros(sim_count), np.ones(exp_count)),
    ):
        assert vl.calc_ks_1d(sim_samples, exp_samples) == pytest.approx(
            stats.ks_2samp(sim_samples, exp_samples).statistic, rel=1e-13, abs=1e-14
        )
        assert vl.calc_cvm_1d(sim_samples, exp_samples) == pytest.approx(
            stats.cramervonmises_2samp(sim_samples, exp_samples).statistic,
            rel=1e-13,
            abs=1e-14,
        )


@pytest.mark.parametrize("duplicates", list(vl.EMAVMDuplicates))
def test_native_and_interpreted_mavm_agree(duplicates: vl.EMAVMDuplicates) -> None:
    source = Path(__file__).resolve().parents[2] / "src/pyvale/validation/cython/mavm.py"
    interpreted = cast(
        Callable[..., vl.MAVMResult],
        runpy.run_path(str(source))["_cyth_calc_mavm_1d"],
    )
    generator = np.random.default_rng(2751)
    sim_data = generator.normal(size=64)
    exp = generator.normal(size=8)
    if duplicates is vl.EMAVMDuplicates.INCLUDE:
        sim_data = np.round(sim_data)
        exp = np.round(exp)
    sim_data[0] = np.nan
    exp[0] = np.nan
    native = vl.calc_mavm_1d(sim_data, exp, duplicates=duplicates)
    reference = interpreted(sim_data, exp, duplicates=duplicates)
    np.testing.assert_allclose(
        [native.d_plus, native.d_minus],
        [reference.d_plus, reference.d_minus],
        rtol=1e-13,
        atol=1e-12,
    )
    np.testing.assert_array_equal(native.model_quantiles, reference.model_quantiles)
    np.testing.assert_array_equal(native.model_probs, reference.model_probs)
    np.testing.assert_array_equal(native.exp_quantiles, reference.exp_quantiles)
    np.testing.assert_array_equal(native.exp_probs, reference.exp_probs)


def test_prepared_plate_gold_metrics() -> None:
    """Track the saved SensorSim fixture and selected metric diagnostics."""
    valid_case = load_plate_case()
    result = vl.ValidEngine([vl.MetricAVM(), vl.MetricMAVM()]).run(valid_case)
    assert result.metrics["temperature"][vl.EMetricType.AVM].outputs[
        vl.EMetricOut.DISTANCE
    ][5, 0, 0] == pytest.approx(0.26243433506330643, abs=1e-10)
    assert result.metrics["temperature"][vl.EMetricType.MAVM].outputs[
        vl.EMetricOut.D_PLUS
    ][5, 0, 0] + result.metrics["temperature"][vl.EMetricType.MAVM].outputs[
        vl.EMetricOut.D_MINUS
    ][5, 0, 0] == pytest.approx(0.4891546824953229, abs=1e-10)
    assert result.metrics["strain"][vl.EMetricType.AVM].outputs[
        vl.EMetricOut.DISTANCE
    ][5, 0, 0] == pytest.approx(1.2018068336589567e-06, abs=1e-12)
    assert result.metrics["strain"][vl.EMetricType.MAVM].outputs[
        vl.EMetricOut.D_PLUS
    ][5, 2, 0] + result.metrics["strain"][vl.EMetricType.MAVM].outputs[
        vl.EMetricOut.D_MINUS
    ][5, 2, 0] == pytest.approx(3.368761032966909e-06, abs=1e-12)


def test_real_simdata_expdata_and_direct_arrays_agree() -> None:
    """The field and acquisition converters match an explicit NumPy case."""
    with np.load(dataset.valid_data_dir() / "plate_sensorsim_2d.npz") as saved:
        coords = saved["coords"].copy()
        times = saved["times"].copy()
        measured = saved["exp_temp"].copy()
    observables = {
        "temperature": vl.ValidObservable(
            components=("temperature",),
            geometry=vl.EValidGeometry.POINT,
            coords=coords,
            times=times,
            phys_units="degC",
        )
    }
    nominal_path = dataset.thermomechanical_2d_experiment_paths()[0]
    sim_data = mh.ExodusLoader(nominal_path).load_all_sim_data()
    sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y"))
    sim = vl.sample_sim_sweep_for_valid(
        {"temperature": sens.FieldScalar(sim_data, "temperature", sens.EDim.TWOD)},
        observables,
        [((), sim_data)],
    )
    output = {
        ("perturbed", "temp", "meas"): measured,
        ("perturbed", "temp", "sens_times"): times,
    }
    acquisitions = vl.extract_exp_sim_data(
        output,
        sim_key="perturbed",
        selections={"temperature": vl.SensorOutputSelection("temp", 0, coords)},
    )
    cases = [
        vl.ValidCase(
            observables, sim, vl.assemble_exp_arrays(run, observables)
        )
        for run in acquisitions
    ]
    converted = vl.concatenate_valid_cases(*cases)
    direct_sim = {
        "temperature": vl.ValidArray(
            sim["temperature"].data, sim["temperature"].axes
        )
    }
    direct_exp = {
        "temperature": vl.ValidArray(
            np.moveaxis(measured, 0, -1),
            (
                vl.EValidAxis.POINT,
                vl.EValidAxis.COMPONENT,
                vl.EValidAxis.TIME,
                vl.EValidAxis.ALEATORY,
            ),
        )
    }
    direct = vl.ValidCase(observables, direct_sim, direct_exp)
    np.testing.assert_array_equal(
        converted.exp["temperature"].data, direct.exp["temperature"].data
    )
    engine = vl.ValidEngine([vl.MetricAVM(), vl.MetricMAVM()])
    converted_result = engine.run(converted)
    direct_result = engine.run(direct)
    for metric in (vl.EMetricType.AVM, vl.EMetricType.MAVM):
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
        "temperature_average": vl.ValidObservable(
            components=("temperature",),
            geometry=vl.EValidGeometry.POINT,
            coords=coords,
            times=times,
            phys_units="degC",
        ),
        "global_temperature": vl.ValidObservable(
            components=("temp_avg",),
            geometry=vl.EValidGeometry.GLOBAL,
            coords=None,
            times=times,
            phys_units="degC",
        ),
    }
    field = sens.FieldScalar(sim_data, "temperature", sens.EDim.TWOD)
    result = vl.sample_sim_sweep_for_valid(
        {
            "temperature_average": sens.FieldScalar(
                sim_data, "temperature", sens.EDim.TWOD
            )
        },
        observables,
        [((), sim_data)],
        sensor_data={"temperature_average": average_sensor},
    )
    expected_average = sens.sample_field_with_sensor_data(field, average_sensor)
    np.testing.assert_allclose(result["temperature_average"].data, expected_average)
    np.testing.assert_allclose(
        result["global_temperature"].data[0, 0],
        np.interp(times, sim_data.time, sim_data.glob_vars["temp_avg"]),
    )


def test_spatial_temporal_offset_is_applied_per_conditional_cdf() -> None:
    valid_case = load_plate_case(transient=True, epistemic=True, offsets=True)
    quantity = valid_case.exp["temperature"]
    assert quantity.offsets.axes == (vl.EValidAxis.POINT, vl.EValidAxis.TIME)
    nominal = quantity.select_aleatory(5, 0, 0)
    lower_early = quantity.select_aleatory(5, 0, 0, offset=vl.EEpisOffset.LOWER)
    lower_late = quantity.select_aleatory(5, 0, 4, offset=vl.EEpisOffset.LOWER)
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


@dataclass(slots=True)
class Pulse25XReference:
    sim_data: np.ndarray
    sensor_tags: tuple[str, ...]
    exp: dict[str, np.ndarray]
    offsets: dict[str, float]
    expected: pd.DataFrame


@pytest.fixture(scope="module")
def pulse_reference() -> Pulse25XReference:
    data_path = dataset.valid_data_dir()
    with np.load(data_path / "pulse25x_full_simulation.npz") as saved:
        sim_data = saved["simulation"]
        tags = tuple(saved["sensor_tags"].tolist())
    exp = {}
    for suffix, channels in (
        (
            "SteadyDICData",
            ("TC1", "TC3", "TC4", "TC5", "TC6", "TC7", "TC8", "TC9", "TC10"),
        ),
        ("SteadyHIVEData", ("TC2",)),
        ("SteadyPICOData", ("CV",)),
    ):
        frame = pd.read_csv(data_path / f"Pulse255_{suffix}.csv")
        for channel in channels:
            key = "Coil RMS Voltage" if channel == "CV" else channel
            exp[channel] = frame[key].to_numpy(dtype=float)
    frame = pd.read_csv(data_path / "pulse25x_uncertainty_summary.csv")
    uncertainty = frame.iloc[-1, -11:].to_numpy(dtype=float)
    uncertainty = np.r_[uncertainty[1:], uncertainty[0]]
    offsets = dict(zip(exp, uncertainty, strict=True))
    expected = pd.read_csv(data_path / "pulse25x_reference_mavm.csv", index_col=0)
    return Pulse25XReference(sim_data, tags, exp, offsets, expected)


@pytest.mark.parametrize(
    "channel", ("TC3", "TC5", "TC6", "TC8", "TC9", "TC10", "TC2", "CV")
)
def test_all_paper_endpoint_distances_match_independent_reference(
    pulse_reference: Pulse25XReference, channel: str
) -> None:
    scenarios = pulse_reference.sim_data[pulse_reference.sensor_tags.index(channel)]
    support_sums = np.sum(np.sort(scenarios, axis=1), axis=1)
    lower_idx = int(np.argmin(support_sums))
    upper_idx = int(np.argmax(support_sums))
    exp_support = np.unique(pulse_reference.exp[channel])
    for exp_name, sign in (("min", -1.0), ("max", 1.0)):
        exp = exp_support + sign * pulse_reference.offsets[channel]
        for sim_name, index in (("min", lower_idx), ("max", upper_idx)):
            calculated = vl.calc_mavm_1d(
                scenarios[index], exp, duplicates=vl.EMAVMDuplicates.REJECT
            )
            for direction, distance in (
                ("d+", calculated.d_plus),
                ("d-", calculated.d_minus),
            ):
                row = f"sim-{sim_name}_exp-{exp_name}-{direction}"
                assert distance == pytest.approx(
                    pulse_reference.expected.loc[row, channel], rel=1e-12, abs=1e-12
                )
    correction = vl.calc_mavm_bound_pairs(
        scenarios[lower_idx],
        scenarios[upper_idx],
        exp_support - pulse_reference.offsets[channel],
        exp_support + pulse_reference.offsets[channel],
        duplicates=vl.EMAVMDuplicates.REJECT,
    )
    expected_lower = max(
        pulse_reference.expected.loc[f"sim-min_exp-{exp_name}-d-", channel]
        for exp_name in ("min", "max")
    )
    expected_upper = max(
        pulse_reference.expected.loc[f"sim-max_exp-{exp_name}-d+", channel]
        for exp_name in ("min", "max")
    )
    assert correction.lower.d_minus == pytest.approx(
        expected_lower, rel=1e-12, abs=1e-12
    )
    assert correction.upper.d_plus == pytest.approx(
        expected_upper, rel=1e-12, abs=1e-12
    )
