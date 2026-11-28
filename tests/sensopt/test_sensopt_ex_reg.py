# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Regression tests for the sensor placement examples."""

import runpy
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from scipy.spatial.distance import pdist

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensopt as opt
import pyvale.sensorsim as sens
from pyvale.verif import pointsensscalar


@pytest.fixture(scope="module")
def plate_simulation() -> io.SimData:
    return mh.ExodusLoader(dataset.mechanical_2d_path()).load_all_sim_data()


@pytest.fixture
def prepare_example_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path / "pyvale-output"


# Reference layouts and scores come from direct FE sampling followed by a
# brute-force nodal distance matrix and independent candidate enumeration.
@pytest.mark.data
@pytest.mark.parametrize(
    ("example", "expected_xy", "expected_objective", "stop_condition", "iterations"),
    [
        (
            "ex01_greedy_placement",
            (
                (100 / 3, 1200 / 19),
                (100 / 3, 1350 / 19),
                (200 / 3, 1350 / 19),
                (100 / 3, 1500 / 19),
                (200 / 3, 1500 / 19),
            ),
            -0.005886421993069738,
            opt.EOptStopCond.SENSOR_COUNT_REACHED,
            5,
        ),
        (
            "ex02_optimal_sensor_count",
            (
                (1000 / 13, 150 / 17),
                (1100 / 13, 1650 / 17),
                (100 / 13, 1800 / 17),
                (200 / 13, 1800 / 17),
                (800 / 13, 1800 / 17),
                (900 / 13, 1800 / 17),
            ),
            0.0012266867750345644,
            opt.EOptStopCond.OBJECTIVE_TARGET_REACHED,
            6,
        ),
        (
            "ex03_sensor_removal",
            ((50, 30), (75, 30), (50, 60), (50, 90), (50, 120)),
            0.0007655738886931121,
            opt.EOptStopCond.SENSOR_COUNT_REACHED,
            7,
        ),
    ],
    ids=("greedy-signal", "target-rmse", "backward-removal"),
)
def test_placement_examples_match_reference(
    prepare_example_output: Path,
    example: str,
    expected_xy: tuple[tuple[float, float], ...],
    expected_objective: float,
    stop_condition: opt.EOptStopCond,
    iterations: int,
) -> None:
    namespace = runpy.run_module(f"pyvale.examples.sensopt.{example}")
    result = namespace["result"]
    assert isinstance(result, opt.OptResult)

    expected_xyz = np.column_stack(
        (np.asarray(expected_xy), np.zeros(len(expected_xy)))
    )
    positions = result.sensor_data["strain"].positions
    assert positions is not None
    np.testing.assert_allclose(positions, expected_xyz, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        result.objective_values, [expected_objective], rtol=1e-10, atol=1e-12
    )
    assert result.stop_condition is stop_condition
    assert result.num_iterations == iterations
    assert result.success
    image_path = prepare_example_output / f"sensopt_{example}" / "sensor_locations.png"
    assert image_path.stat().st_size > 0
    if example == "ex01_greedy_placement":
        peak_result = namespace["results"]["peak_abs"]
        peak_positions = peak_result.sensor_data["strain"].positions
        assert not np.array_equal(positions, peak_positions)
        assert (image_path.parent / "peak_abs_sensor_locations.png").is_file()
        assert (image_path.parent / "final_mean_sensor_locations.png").is_file()


@pytest.mark.data
def test_exchange_example_matches_enumerated_optimum(
    prepare_example_output: Path,
) -> None:
    namespace = runpy.run_module("pyvale.examples.sensopt.ex04_fixed_count_refinement")
    exchange = namespace["exchange"]
    exhaustive = namespace["exhaustive"]
    assert isinstance(exchange, opt.OptResult)
    assert isinstance(exhaustive, opt.OptResult)
    reference_layouts = (
        (
            exchange,
            [[25.0, 30.0, 0.0], [75.0, 30.0, 0.0], [50.0, 120.0, 0.0]],
            0.0007402701225393062,
        ),
        (
            exhaustive,
            [[50.0, 30.0, 0.0], [25.0, 120.0, 0.0], [75.0, 120.0, 0.0]],
            0.0007402701223788911,
        ),
    )
    for result, expected_positions, expected_rmse in reference_layouts:
        np.testing.assert_array_equal(
            result.sensor_data["strain"].positions, expected_positions
        )
        np.testing.assert_allclose(result.objective_values, [expected_rmse], rtol=1e-10)
        assert result.success

    assert exhaustive.objective_values[0] <= exchange.objective_values[0]
    assert exchange.stop_condition is opt.EOptStopCond.NO_IMPROVEMENT
    assert exchange.num_iterations == 2
    assert exhaustive.stop_condition is opt.EOptStopCond.SEARCH_EXHAUSTED
    assert exhaustive.num_iterations == 220
    exhaustive_results = namespace["exhaustive_results"]
    range_result = exhaustive_results["exhaustive_range"]
    target_result = exhaustive_results["exhaustive_target"]
    for result in (range_result, target_result):
        assert result.success
        assert result.stop_condition is opt.EOptStopCond.SEARCH_EXHAUSTED
        assert result.num_iterations == 298
    np.testing.assert_array_equal(
        range_result.sensor_data["strain"].positions,
        exhaustive.sensor_data["strain"].positions,
    )
    target_positions = target_result.sensor_data["strain"].positions
    assert target_positions is not None
    assert target_positions.shape == (1, 3)
    assert exhaustive.objective_values[0] < target_result.objective_values[0]
    assert target_result.objective_values[0] <= namespace["target_rmse"]
    output_path = prepare_example_output / "sensopt_ex04_fixed_count_refinement"
    for name in namespace["results"]:
        assert (output_path / f"{name}_sensor_locations.png").stat().st_size > 0
    max_error_result = namespace["exhaustive_max_error"]
    assert max_error_result.stop_condition is opt.EOptStopCond.SEARCH_EXHAUSTED
    assert not np.array_equal(
        exhaustive.sensor_data["strain"].positions,
        max_error_result.sensor_data["strain"].positions,
    )


@pytest.mark.data
def test_plate_exclusion_compares_signal_and_reconstruction(
    prepare_example_output: Path,
) -> None:
    example = "ex07_plate_site_constraints"
    namespace = runpy.run_module(f"pyvale.examples.sensopt.{example}")
    results = namespace["results"]
    output_path = prepare_example_output / f"sensopt_{example}"

    np.testing.assert_allclose(
        [
            -results["peak_signal", "unrestricted"].objective_values[0],
            -results["peak_signal", "excluded"].objective_values[0],
        ],
        [0.0181307, 0.0134850],
        rtol=5e-6,
    )
    for design in ("unrestricted", "excluded"):
        np.testing.assert_allclose(
            results["field_recon", design].objective_values, [0.0012925], rtol=5e-5
        )
    for (objective, design), result in results.items():
        assert result.success
        assert result.stop_condition is opt.EOptStopCond.SENSOR_COUNT_REACHED
        positions = result.sensor_data["strain"].positions
        assert positions is not None
        assert positions.shape == (1 if objective == "peak_signal" else 4, 3)
        residuals = namespace["site_constraint"].eval(result.sensor_data)
        if design == "excluded":
            assert np.all(residuals <= 0)
        elif objective == "peak_signal":
            assert np.any(residuals > 0)
        assert (output_path / f"{objective}_{design}.png").stat().st_size > 0


@pytest.mark.data
def test_candidate_filter_excludes_hole_and_exterior(
    plate_simulation: io.SimData,
) -> None:
    sites = np.array(
        [
            [75.0, 120.0, 0.0],
            [50.0, 75.0, 0.0],
            [-1.0, 30.0, 0.0],
            [25.0, 30.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(
        opt.filter_sensor_sites_in_mesh(sites, plate_simulation), sites[[0, 3]]
    )


@pytest.mark.data
@pytest.mark.parametrize(
    ("example", "expected_counts", "expected_score", "expected_stop"),
    [
        (
            "ex05_temperature_sensor_mix",
            {"precision": 1, "economy": 2},
            3.87348,
            opt.EOptStopCond.SENSOR_COUNT_REACHED,
        ),
        (
            "ex06_thermomechanical_placement",
            {"temperature": 1, "displacement": 2},
            9.49494,
            opt.EOptStopCond.SEARCH_EXHAUSTED,
        ),
    ],
)
def test_joint_placement_examples(
    prepare_example_output: Path,
    example: str,
    expected_counts: dict[str, int],
    expected_score: float,
    expected_stop: opt.EOptStopCond,
) -> None:
    namespace = runpy.run_module(f"pyvale.examples.sensopt.{example}")
    result = namespace["result"]
    assert isinstance(result, opt.OptResult)
    assert result.success
    assert result.stop_condition is expected_stop
    np.testing.assert_allclose(result.objective_values, [expected_score], rtol=2e-6)
    assert result.sensor_data.keys() == expected_counts.keys()

    positions: list[npt.NDArray[np.float64]] = []
    output_path = prepare_example_output / f"sensopt_{example}"
    for key, count in expected_counts.items():
        selected = result.sensor_data[key].positions
        assert selected is not None
        assert selected.shape == (count, 3)
        positions.append(selected)
        assert (output_path / f"{key}_sensor_locations.png").stat().st_size > 0
    distances = pdist(np.concatenate(positions))
    assert np.all(distances > 0)
    assert np.all(distances >= namespace.get("minimum_spacing", 0.0))
    if "budget" in namespace:
        assert namespace["cost"] <= namespace["budget"]

    for prefix, layout in namespace["layouts"].items():
        if not prefix:
            continue
        assert layout.success
        assert layout.stop_condition is opt.EOptStopCond.SENSOR_COUNT_REACHED
        positions = [data.positions for data in layout.sensor_data.values()]
        assert sum(len(selected) for selected in positions) == 3
        distances = pdist(np.concatenate(positions))
        assert np.all(distances > 0)
        assert np.all(distances >= namespace.get("minimum_spacing", 0.0))
        for key in layout.sensor_data:
            assert (output_path / f"{prefix}{key}_sensor_locations.png").stat().st_size > 0

    if "fixed_start_result" in namespace:
        fixed_start = namespace["fixed_start_result"]
        assert fixed_start.num_start_layouts == 1
        assert fixed_start.num_iterations == 2
        np.testing.assert_array_equal(
            fixed_start.sensor_data["precision"].positions, namespace["sites"][:1]
        )
    else:
        greedy_results = namespace["greedy_results"]
        assert greedy_results["exhaustive_start"].num_start_layouts == (
            len(namespace["sites"]) ** 2
        )
        assert greedy_results["random_start"].num_start_layouts == 12
        assert greedy_results["exhaustive_start"].num_iterations == 1
        assert greedy_results["random_start"].num_iterations == 1
        assert greedy_results["empty_score"].num_start_layouts == 0
        assert greedy_results["empty_score"].num_iterations == 3


@pytest.fixture
def analytic_array() -> sens.SensorsPoint:
    simulation = pointsensscalar.simdata_2d_analytic()
    nominal_positions = np.array([[0.25, 0.25, 0.0], [0.5, 0.5, 0.0]])
    return pointsensscalar.sens_array_noerrs(
        simulation, sens.SensorData(positions=nominal_positions), sens.EDim.TWOD
    )


class CountStartLayouts(opt.IConstrainInp):
    def __init__(self) -> None:
        self.layouts: list[tuple[float, ...]] = []

    def eval(
        self, sensor_data: Mapping[str, sens.SensorData]
    ) -> npt.NDArray[np.float64]:
        layout: list[float] = []
        for key in ("first", "second"):
            positions = sensor_data[key].positions
            assert positions is not None
            layout.extend(positions[:, 0].tolist())
        self.layouts.append(tuple(layout))
        return np.zeros(1, dtype=np.float64)


def test_random_start_tests_distinct_layouts_and_repeats_with_seed(
    analytic_array: sens.SensorsPoint,
) -> None:
    sites = np.array([[0.25, 0.25, 0.0], [0.5, 0.5, 0.0], [0.75, 0.75, 0.0]])
    counter = CountStartLayouts()
    engine = opt.SensorOptEngine(
        sensor_arrays={"first": analytic_array, "second": analytic_array},
        design_space=opt.DesignSpace(
            positions={"first": sites, "second": sites},
            position_axes=dict.fromkeys(("first", "second"), opt.EDesignAxis.DISC),
            input_constraints=(counter, opt.ConstrainSensSep()),
        ),
        objectives=(opt.ObjSensorCost(unit_costs={"first": 1.0, "second": 1.0}),),
        optimiser=opt.OptGreedyForward(
            num_sensors=3,
            start=opt.OptStartRandom(
                counts={"first": 2, "second": 1}, num_layouts=20
            ),
        ),
        seed=123,
    )
    result = engine.optimise()
    draws = counter.layouts.copy()
    assert result.success
    assert result.num_iterations == 0
    assert result.num_start_layouts == len(draws) == len(set(draws)) == 9
    counter.layouts.clear()
    engine.optimise()
    assert counter.layouts == draws


def test_greedy_start_reports_infeasibility(
    analytic_array: sens.SensorsPoint,
) -> None:
    sites = np.array([[0.25, 0.25, 0.0]])
    engine = opt.SensorOptEngine(
        sensor_arrays={"first": analytic_array, "second": analytic_array},
        design_space=opt.DesignSpace(
            positions={"first": sites, "second": sites},
            position_axes=dict.fromkeys(("first", "second"), opt.EDesignAxis.DISC),
            input_constraints=(opt.ConstrainSensSep(),),
        ),
        objectives=(opt.ObjSensorCost(unit_costs={"first": 1.0, "second": 1.0}),),
        optimiser=opt.OptGreedyForward(
            num_sensors=2,
            start=opt.OptStartExhaustive(counts={"first": 1, "second": 1}),
        ),
    )
    result = engine.optimise()
    assert result.stop_condition is opt.EOptStopCond.NO_FEASIBLE_START
    assert result.num_start_layouts == 1
    assert result.num_iterations == 0
    assert not result.success

    engine.optimiser = opt.OptGreedyForward(
        num_sensors=2, start=opt.OptStartLayout(sites={"first": (0,), "second": (0,)})
    )
    with pytest.raises(ValueError, match="starting layout violates constraints"):
        engine.optimise()


@pytest.mark.parametrize(
    ("sensor_keys", "expected"), [(("second",), 4.0), (("first", "second"), 0.0)]
)
def test_empty_score_is_used_only_when_all_term_arrays_are_empty(
    analytic_array: sens.SensorsPoint,
    sensor_keys: tuple[str, ...],
    expected: float,
) -> None:
    sites = np.array([[0.25, 0.25, 0.0]])
    result = opt.SensorOptEngine(
        sensor_arrays={"first": analytic_array, "second": analytic_array},
        design_space=opt.DesignSpace(
            positions={"first": sites, "second": sites},
            position_axes=dict.fromkeys(("first", "second"), opt.EDesignAxis.DISC),
        ),
        objectives=opt.ObjWeighted(
            terms=(
                opt.ObjTerm(
                    objective=opt.ObjSensorCost(unit_costs={"second": 1.0}),
                    sensor_keys=sensor_keys,
                    empty_score=8.0,
                    normalisation=opt.ObjNormAffine(gain=0.25),
                    weight=2.0,
                ),
            )
        ),
        optimiser=opt.OptGreedyForward(
            target_objective=8.0, start=opt.OptStartLayout(sites={"first": (0,)})
        ),
    ).optimise()
    assert result.stop_condition is opt.EOptStopCond.OBJECTIVE_TARGET_REACHED
    assert result.num_iterations == 0
    np.testing.assert_array_equal(result.objective_values, [expected])


def test_sensor_limit_returns_partial_layout(
    analytic_array: sens.SensorsPoint,
) -> None:
    sites = np.array([[0.25, 0.25, 0.0], [0.5, 0.5, 0.0], [0.75, 0.75, 0.0]])
    result = opt.SensorOptEngine(
        sensor_arrays={"temperature": analytic_array},
        design_space=opt.DesignSpace(
            positions={"temperature": sites},
            position_axes={"temperature": opt.EDesignAxis.DISC},
        ),
        objectives=(opt.ObjSignalRMS(sensor_keys=("temperature",)),),
        optimiser=opt.OptGreedyForward(target_objective=-1e100, max_sensors=1),
    ).optimise()

    positions = result.sensor_data["temperature"].positions
    assert positions is not None
    assert positions.shape == (1, 3)
    assert result.stop_condition is opt.EOptStopCond.SENSOR_LIMIT_REACHED
    assert result.num_iterations == 1
    assert not result.success


def test_weighted_objective_normalises_each_term_and_skips_zero_weights(
    analytic_array: sens.SensorsPoint,
) -> None:
    sites = np.array([[0.25, 0.25, 0.0]])
    first_cost = opt.ObjSensorCost(unit_costs={"first": 2.0})
    combined_cost = opt.ObjSensorCost(unit_costs={"first": 2.0, "second": 4.0})
    objective = opt.ObjWeighted(
        terms=(
            opt.ObjTerm(
                objective=first_cost,
                weight=2.0,
                normalisation=opt.ObjNormAffine(gain=0.5, offset=1.0),
            ),
            opt.ObjTerm(
                objective=combined_cost,
                normalisation=opt.ObjNormLog(reference=3.0),
            ),
            opt.ObjTerm(
                objective=combined_cost,
                normalisation=opt.ObjNormLog1p(reference=2.0),
            ),
            opt.ObjTerm(
                objective=opt.ObjSignalRMS(sensor_keys=("missing",)), weight=0.0
            ),
        )
    )
    result = opt.SensorOptEngine(
        sensor_arrays={"first": analytic_array, "second": analytic_array},
        design_space=opt.DesignSpace(
            positions={"first": sites, "second": sites},
            position_axes={"first": opt.EDesignAxis.DISC, "second": opt.EDesignAxis.DISC},
        ),
        objectives=objective,
        optimiser=opt.OptExhaustive(num_sensors=2),
    ).optimise()

    assert result.success
    np.testing.assert_allclose(result.objective_values, [4.0 + np.log(8.0)])


@pytest.mark.parametrize(
    ("distance", "minimum_distance", "feasible"),
    [(0.0, 0.0, False), (1.0, 0.0, True), (1.0, 2.0, False), (2.0, 2.0, True)],
)
def test_separation_applies_across_arrays(
    distance: float, minimum_distance: float, feasible: bool
) -> None:
    constraint = opt.ConstrainSensSep(min_distance=minimum_distance)
    residuals = constraint.eval(
        {
            "first": sens.SensorData(positions=np.array([[0.0, 0.0, 0.0]])),
            "second": sens.SensorData(positions=np.array([[distance, 0.0, 0.0]])),
        }
    )
    assert bool(np.all(residuals <= 0)) is feasible


@pytest.mark.parametrize(
    ("max_channels", "expected_residual"), [(4, 1.0), (5, 0.0), (6, -1.0)]
)
def test_channel_limit_counts_hardware_channels(
    max_channels: int, expected_residual: float
) -> None:
    constraint = opt.ConstrainChannCount(
        channels_per_sensor={"temperature": 1, "rosette": 3, "unused": 8},
        max_channels=max_channels,
    )
    residuals = constraint.eval(
        {
            "temperature": sens.SensorData(positions=np.zeros((2, 3))),
            "rosette": sens.SensorData(positions=np.zeros((1, 3))),
            "empty": sens.SensorData(positions=np.empty((0, 3))),
        }
    )
    np.testing.assert_array_equal(residuals, [expected_residual])


def test_channel_limit_rejects_unconfigured_selected_array() -> None:
    constraint = opt.ConstrainChannCount(channels_per_sensor={}, max_channels=5)
    with pytest.raises(ValueError, match="channels_per_sensor"):
        constraint.eval({"rosette": sens.SensorData(positions=np.zeros((1, 3)))})


class MeanMeasurement(opt.IObjFunc):
    def get_required_outputs(self) -> frozenset[sens.EExpSimKeys]:
        return frozenset({sens.EExpSimKeys.MEASUREMENT})

    def eval(
        self,
        *,
        sensor_data: Mapping[str, sens.SensorData],
        experiment_data: opt.ExperimentData,
        scenario_keys: tuple[str, ...],
    ) -> npt.NDArray[np.float64]:
        measurements = experiment_data[
            scenario_keys[0], "temperature", sens.EExpSimKeys.MEASUREMENT
        ]
        return np.array(measurements.mean(), dtype=np.float64)


@pytest.mark.parametrize(
    (
        "num_sensors",
        "max_iterations",
        "expected_count",
        "expected_iterations",
        "expected_stop",
    ),
    [
        (None, None, 1, 6, opt.EOptStopCond.SEARCH_EXHAUSTED),
        (range(1, 3), 4, 1, 4, opt.EOptStopCond.ITERATION_LIMIT),
        (range(2, 3), None, 2, 3, opt.EOptStopCond.SEARCH_EXHAUSTED),
        (2, None, 2, 3, opt.EOptStopCond.SEARCH_EXHAUSTED),
    ],
)
def test_exhaustive_target_prefers_smallest_qualifying_layout(
    analytic_array: sens.SensorsPoint,
    num_sensors: int | range | None,
    max_iterations: int | None,
    expected_count: int,
    expected_iterations: int,
    expected_stop: opt.EOptStopCond,
) -> None:
    sites = np.array([[0.25, 0.25, 0.0], [0.5, 0.5, 0.0], [0.75, 0.75, 0.0]])
    result = opt.SensorOptEngine(
        sensor_arrays={"temperature": analytic_array},
        design_space=opt.DesignSpace(
            positions={"temperature": sites},
            position_axes={"temperature": opt.EDesignAxis.DISC},
        ),
        objectives=(opt.ObjSignalRMS(sensor_keys=("temperature",)),),
        optimiser=opt.OptExhaustive(
            num_sensors=num_sensors, target_objective=0.0, max_iterations=max_iterations
        ),
    ).optimise()

    positions = result.sensor_data["temperature"].positions
    assert positions is not None
    assert positions.shape == (expected_count, 3)
    assert result.num_iterations == expected_iterations
    assert result.stop_condition is expected_stop
    assert result.success


def test_output_constraints_request_histories_and_reject_cheaper_layout(
    analytic_array: sens.SensorsPoint,
) -> None:
    biased = analytic_array.copy_for_simulation()
    biased.set_error_chain([sens.ErrSysOffset(offset=2.0)])
    sites = np.array([[0.25, 0.25, 0.0]])
    arrays = {"economy": biased, "precision": analytic_array}
    costs = opt.ObjSensorCost(unit_costs={"economy": 1.0, "precision": 2.0})
    design = opt.DesignSpace(
        positions=dict.fromkeys(arrays, sites),
        position_axes=dict.fromkeys(arrays, opt.EDesignAxis.DISC),
        output_constraints=(
            opt.ConstrainMeasErr(sensor_keys=tuple(arrays), max_error=1.0),
            opt.ConstrainObjVal(objective=costs, max_value=2.0),
        ),
    )
    engine = opt.SensorOptEngine(
        sensor_arrays=arrays,
        design_space=design,
        objectives=(costs,),
        optimiser=opt.OptExhaustive(num_sensors=1),
    )
    result = engine.optimise()
    assert result.success
    assert result.sensor_data.keys() == {"precision"}
    np.testing.assert_array_equal(result.objective_values, [2.0])

    design.output_constraints = (*design.output_constraints, opt.ConstrainObjVal(
        objective=costs, max_value=1.5
    ))
    assert not engine.optimise().success


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (opt.EErrMetric.RMSE, np.sqrt(5.0)),
        (opt.EErrMetric.MAE, 2.0),
        (opt.EErrMetric.MAX_ERR, 3.0),
    ],
)
def test_measurement_constraint_reduces_each_realisation(
    metric: opt.EErrMetric, expected: float
) -> None:
    constraint = opt.ConstrainMeasErr(
        sensor_keys=("strain",), max_error=2.0, metric=metric
    )
    measured = np.array([[[[1.0, -3.0]]], [[[2.0, 2.0]]]])
    residuals = constraint.eval(
        sensor_data={"strain": sens.SensorData(positions=np.zeros((1, 3)))},
        experiment_data={
            ("nominal", "strain", sens.EExpSimKeys.MEASUREMENT): measured,
            ("nominal", "strain", sens.EExpSimKeys.TRUTH): np.zeros_like(measured),
        },
        scenario_keys=("nominal",),
    )
    np.testing.assert_allclose(residuals, [expected - 2.0, 0.0])


def test_seeded_search_leaves_caller_array_and_rng_intact(
    analytic_array: sens.SensorsPoint,
) -> None:
    analytic_array.set_error_chain([sens.ErrRandGen(sens.GenNormal(std=1.0, seed=456))])
    previous_measurements = analytic_array.sim_measurements().copy()
    untouched_copy = analytic_array.copy_for_simulation()
    nominal_positions = analytic_array.get_sensor_positions()
    assert nominal_positions is not None
    engine = opt.SensorOptEngine(
        sensor_arrays={"temperature": analytic_array},
        design_space=opt.DesignSpace(
            positions={"temperature": nominal_positions},
            position_axes={"temperature": opt.EDesignAxis.DISC},
        ),
        objectives=(MeanMeasurement(),),
        optimiser=opt.OptExhaustive(num_sensors=1),
        seed=123,
    )

    first = engine.optimise()
    second = engine.optimise()
    np.testing.assert_array_equal(first.objective_values, second.objective_values)
    np.testing.assert_array_equal(
        first.sensor_data["temperature"].positions,
        second.sensor_data["temperature"].positions,
    )
    engine.seed = 456
    assert not np.array_equal(
        first.objective_values, engine.optimise().objective_values
    )

    np.testing.assert_array_equal(
        analytic_array.get_sensor_positions(), nominal_positions
    )
    np.testing.assert_array_equal(
        analytic_array.get_measurements(), previous_measurements
    )
    np.testing.assert_array_equal(
        analytic_array.sim_measurements(), untouched_copy.sim_measurements()
    )


def test_updating_sensor_data_invalidates_cached_outputs(
    analytic_array: sens.SensorsPoint,
) -> None:
    analytic_array.set_error_chain([sens.ErrSysOffset(offset=2.0)])
    original_measurements = analytic_array.sim_measurements()
    field = analytic_array.get_field()
    positions = np.array([[0.75, 0.75, 0.0]])
    times = field.get_time_steps()[::2]
    analytic_array.set_sensor_data(
        sens.SensorData(positions=positions, sample_times=times)
    )

    expected_truth = field.sample_field(positions, times)
    assert original_measurements.shape != expected_truth.shape
    np.testing.assert_array_equal(analytic_array.get_truth(), expected_truth)
    np.testing.assert_array_equal(
        analytic_array.get_errors_total(), np.zeros_like(expected_truth)
    )
    np.testing.assert_array_equal(
        analytic_array.get_measurements(), expected_truth + 2.0
    )
