# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Discrete acquisition time and joint placement checks."""

import runpy
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

import pyvale.sensoropt as opt
import pyvale.sensorsim as sens
from pyvale.verif.analyticsimdatafactory import scalar_linear_2d


def test_time_discretisation_preserves_count_interval_and_window() -> None:
    schedules = opt.gen_sample_time_combinations(
        candidate_times=np.array([0.0, 0.4, 0.9]), num_samples=2
    )
    np.testing.assert_array_equal(schedules, [[0.0, 0.4], [0.0, 0.9], [0.4, 0.9]])

    periods = opt.gen_sample_times_from_periods(
        periods=np.array([0.1, 0.2]), time_window=(0.0, 0.3)
    )
    freqs = opt.gen_sample_times_from_freqs(
        frequencies=np.array([10.0, 5.0]), time_window=(0.0, 0.3)
    )
    np.testing.assert_allclose(periods[0], [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(periods[1], [0.0, 0.2])
    assert periods[0][-1] == 0.3
    for period_times, freq_times in zip(periods, freqs, strict=True):
        np.testing.assert_array_equal(period_times, freq_times)
    with pytest.raises(ValueError, match="strictly increasing"):
        opt.gen_sample_time_combinations(
            candidate_times=np.array([0.0, 0.0]), num_samples=1
        )


def test_joint_search_against_independent_position_times_product() -> None:
    mesh, _ = scalar_linear_2d()
    assert mesh.coords is not None
    native_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    profile = np.array([0.0, 0.0, 3.0, 1.0, 0.0])
    sim_data = replace(
        mesh,
        time=native_times,
        node_vars={"temperature": 20.0 + mesh.coords[:, :1] * profile},
    )
    sites = np.array([[2.0, 2.5, 0.0], [5.0, 2.5, 0.0], [8.0, 2.5, 0.0]])
    targets = np.array([[1.0, 2.5, 0.0], [2.0, 2.5, 0.0], [8.0, 2.5, 0.0]])
    target_times = np.array([0.0, 0.375, 0.5, 0.625, 1.0])
    schedules = (
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 0.75, 1.0]),
        np.array([0.25, 0.5, 0.75]),  # Does not cover the score window.
    )
    array_field = sens.FieldScalar(
        sim_data,
        comp_key="temperature",
        spatial_dims=sens.EDim.TWOD,
    )
    array = sens.SensorsPoint(
        sensor_data=sens.SensorData(positions=sites[:2]),
        field=array_field,
    )
    engine = opt.SensorOptEngine(
        sensor_arrays={"temperature": array},
        design_space=opt.DesignSpace(
            positions={"temperature": sites},
            position_axes={"temperature": opt.EDesignAxis.DISC},
            sample_times={"temperature": schedules},
        ),
        objectives=(
            opt.ObjFieldRecon(
                target_positions=targets,
                target_times=target_times,
                reference_field="temperature",
                sensor_keys=("temperature",),
                component_index=0,
            ),
        ),
        optimiser=opt.OptExhaustive(num_sensors=2),
    )

    # Independently compute nearest-site values and interpolate over the same
    # fixed scoring grid for every position and feasible schedule.
    reference = 20.0 + targets[:, :1] * np.interp(target_times, native_times, profile)
    expected = float("inf")
    for schedule in schedules[:2]:
        sampled_profile = np.interp(schedule, native_times, profile)
        reconstructed_profile = np.interp(target_times, schedule, sampled_profile)
        for indices in combinations(range(len(sites)), 2):
            candidate_x = sites[list(indices), 0]
            closest = np.argmin(np.abs(targets[:, :1] - candidate_x), axis=1)
            predicted = 20.0 + candidate_x[closest, None] * reconstructed_profile
            expected = min(
                expected, float(np.sqrt(np.mean((predicted - reference) ** 2)))
            )

    exhaustive = engine.optimise()
    assert exhaustive.success
    assert exhaustive.num_iterations == 6
    np.testing.assert_allclose(exhaustive.objective_values, [expected])
    engine.optimiser = opt.OptGreedyForward(num_sensors=2)
    greedy = engine.optimise()
    assert greedy.success
    np.testing.assert_allclose(greedy.objective_values, [expected])
    np.testing.assert_array_equal(
        greedy.sensor_data["temperature"].sample_times, schedules[0]
    )

    # With no target grid, score only the times actually sampled. This schedule
    # misses the pulse entirely, and so has zero reconstruction error there.
    baseline_times = np.array([0.0, 0.25, 1.0])
    engine.design_space = opt.DesignSpace(
        sample_times={"temperature": (baseline_times,)}
    )
    engine.objectives = (
        opt.ObjFieldRecon(
            target_positions=targets,
            reference_field="temperature",
            sensor_keys=("temperature",),
            component_index=0,
        ),
    )
    engine.optimiser = opt.OptFixedLayout()
    acquired_only = engine.optimise()
    assert acquired_only.success
    np.testing.assert_allclose(acquired_only.objective_values, [0.0], atol=1e-12)


def test_transient_examples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    peak = runpy.run_module("pyvale.examples.sensopt.ex10_peak_sample_times")
    for result in peak["results"].values():
        assert result.success
        np.testing.assert_allclose(result.objective_values, [-38.0])
        selected = result.sensor_data["temperature"]
        np.testing.assert_array_equal(selected.positions, [[3.75, 4.5, 0.0]])
        assert len(selected.sample_times) == 3
        assert np.any(np.isclose(selected.sample_times, 0.45))
    assert (peak["output_path"] / "peak_samples.png").stat().st_size > 0
    joint = runpy.run_module("pyvale.examples.sensopt.ex11_joint_pulse_recon")
    assert joint["result"].success
    assert len(joint["selected"].sample_times) == 5
    assert joint["selected"].positions.shape == (3, 3)
    assert (joint["output_path"] / "joint_sensor_locations.png").stat().st_size > 0
