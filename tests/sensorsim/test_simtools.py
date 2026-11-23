"""Tests for simulation data transformations."""

import numpy as np

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
