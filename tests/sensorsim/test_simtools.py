"""Tests for simulation data transformations."""

import numpy as np
import pytest

import pyvale.sensorsim as sens
from pyvale.dataio import SimData


def test_scale_length_units_accepts_simulation_list() -> None:
    simulations = [
        SimData(
            coords=np.array([[length, 0.0, 0.0]]),
            node_vars={
                "disp_x": np.array([[length]]),
                "strain_xx": np.array([[0.01]]),
            },
        )
        for length in (1.0, 2.0)
    ]

    assert (
        sens.scale_length_units(1000.0, simulations, disp_keys=("disp_x",))
        is simulations
    )
    for simulation, expected in zip(simulations, (1000.0, 2000.0), strict=True):
        assert simulation.node_vars is not None
        np.testing.assert_array_equal(simulation.coords, [[expected, 0.0, 0.0]])
        np.testing.assert_array_equal(simulation.node_vars["disp_x"], [[expected]])
        np.testing.assert_array_equal(simulation.node_vars["strain_xx"], [[0.01]])


@pytest.mark.parametrize("use_list", [False, True])
def test_scale_nodal_fields_preserves_coordinates_and_other_fields(
    use_list: bool,
) -> None:
    simulations = [
        SimData(
            coords=np.array([[1.0, 2.0, 3.0]]),
            node_vars={
                "strain_xx": np.array([[0.01]]),
                "strain_xy": np.array([[-0.02]]),
                "temperature": np.array([[25.0]]),
            },
        )
        for _ in range(2 if use_list else 1)
    ]
    sim_data = simulations if use_list else simulations[0]

    assert sens.scale_nodal_fields(
        1000.0, sim_data, field_keys=("strain_xx", "strain_xy")
    ) is sim_data
    for simulation in simulations:
        assert simulation.node_vars is not None
        np.testing.assert_array_equal(simulation.coords, [[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(simulation.node_vars["strain_xx"], [[10.0]])
        np.testing.assert_array_equal(simulation.node_vars["strain_xy"], [[-20.0]])
        np.testing.assert_array_equal(simulation.node_vars["temperature"], [[25.0]])
