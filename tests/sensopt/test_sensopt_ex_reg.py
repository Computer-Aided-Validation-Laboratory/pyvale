"""Regression tests for the first four sensor placement examples."""

from collections.abc import Callable, Mapping

import numpy as np
import numpy.typing as npt
import pytest

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensopt as opt
import pyvale.sensorsim as sens
from pyvale.examples.sensopt.ex01_greedy_placement import optimise_placement
from pyvale.examples.sensopt.ex02_optimal_sensor_count import optimise_sensor_count
from pyvale.examples.sensopt.ex03_sensor_removal import optimise_removal
from pyvale.examples.sensopt.ex04_fixed_count_refinement import compare_optimisers
from pyvale.verif import pointsensscalar


@pytest.fixture(scope="module")
def plate_simulation() -> io.SimData:
    return mh.ExodusLoader(dataset.mechanical_2d_path()).load_all_sim_data()


# Reference layouts and scores come from direct FE sampling followed by a
# brute-force nodal distance matrix and independent candidate enumeration.
@pytest.mark.data
@pytest.mark.parametrize(
    ("example", "expected_xy", "expected_objective", "stop_condition", "iterations"),
    [
        (
            optimise_placement,
            (
                (100 / 3, 2700 / 19),
                (40, 2700 / 19),
                (140 / 3, 2700 / 19),
                (160 / 3, 2700 / 19),
                (60, 2700 / 19),
            ),
            -0.3957312797885622,
            opt.EOptStopCond.SENSOR_COUNT_REACHED,
            5,
        ),
        (
            optimise_sensor_count,
            (
                (1100 / 13, 150 / 17),
                (1100 / 13, 600 / 17),
                (1200 / 13, 1200 / 17),
                (1100 / 13, 1800 / 17),
                (1100 / 13, 1950 / 17),
                (1100 / 13, 2400 / 17),
            ),
            0.046706516484125146,
            opt.EOptStopCond.OBJECTIVE_TARGET_REACHED,
            6,
        ),
        (
            optimise_removal,
            ((25, 30), (75, 30), (75, 60), (25, 90), (75, 120)),
            0.0782752608338788,
            opt.EOptStopCond.SENSOR_COUNT_REACHED,
            7,
        ),
    ],
    ids=("greedy-signal", "target-rmse", "backward-removal"),
)
def test_placement_examples_match_reference(
    plate_simulation: io.SimData,
    example: Callable[[io.SimData], opt.OptimisationResult],
    expected_xy: tuple[tuple[float, float], ...],
    expected_objective: float,
    stop_condition: opt.EOptStopCond,
    iterations: int,
) -> None:
    result = example(plate_simulation)

    expected_xyz = np.column_stack(
        (np.asarray(expected_xy), np.zeros(len(expected_xy)))
    )
    positions = result.sensor_data["displacement"].positions
    assert positions is not None
    np.testing.assert_allclose(positions, expected_xyz, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        result.objective_values, [expected_objective], rtol=1e-10, atol=1e-12
    )
    assert result.stop_condition is stop_condition
    assert result.num_iterations == iterations
    assert result.success


@pytest.mark.data
def test_exchange_example_matches_enumerated_optimum(
    plate_simulation: io.SimData,
) -> None:
    exchange, exhaustive = compare_optimisers(plate_simulation, seed=123)
    expected_positions = np.array(
        [[75.0, 30.0, 0.0], [75.0, 60.0, 0.0], [75.0, 120.0, 0.0]]
    )
    expected_rmse = 0.0736011970306538
    for result in (exchange, exhaustive):
        np.testing.assert_array_equal(
            result.sensor_data["displacement"].positions, expected_positions
        )
        np.testing.assert_allclose(result.objective_values, [expected_rmse], rtol=1e-10)
        assert result.success

    assert exchange.stop_condition is opt.EOptStopCond.NO_IMPROVEMENT
    assert exchange.num_iterations == 4
    assert exhaustive.stop_condition is opt.EOptStopCond.SEARCH_EXHAUSTED
    assert exhaustive.num_iterations == 220


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


@pytest.fixture
def analytic_array() -> sens.SensorsPoint:
    simulation = pointsensscalar.simdata_2d_analytic()
    nominal_positions = np.array([[0.25, 0.25, 0.0], [0.5, 0.5, 0.0]])
    return pointsensscalar.sens_array_noerrs(
        simulation, sens.SensorData(positions=nominal_positions), sens.EDim.TWOD
    )


def test_iteration_limit_returns_partial_layout(
    analytic_array: sens.SensorsPoint,
) -> None:
    sites = np.array([[0.25, 0.25, 0.0], [0.5, 0.5, 0.0], [0.75, 0.75, 0.0]])
    result = opt.SensorOptEngine(
        sensor_arrays={"temperature": analytic_array},
        design_space=opt.DesignSpace(
            positions={"temperature": sites},
            position_axes={"temperature": opt.EDesignAxis.DISC},
        ),
        objectives=(opt.ObjSignal(sensor_keys=("temperature",)),),
        optimiser=opt.OptimiserGreedyForward(num_sensors=2, max_iterations=1),
    ).optimise()

    positions = result.sensor_data["temperature"].positions
    assert positions is not None
    assert positions.shape == (1, 3)
    assert result.stop_condition is opt.EOptStopCond.ITERATION_LIMIT
    assert result.num_iterations == 1
    assert not result.success


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
        optimiser=opt.OptimiserExhaustive(num_sensors=1, max_iterations=2),
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
