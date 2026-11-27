"""
Fixed-count layout refinement
=============================

Refine a three-sensor layout by exchange and benchmark it by enumeration.
"""

from pathlib import Path

import numpy as np

# pyvale imports
import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensopt as opt
import pyvale.sensorsim as sens
from pyvale.examples.sensopt.example_objectives import ObjNodalReconRMSE


def compare_optimisers(
    sim_data: io.SimData, *, seed: int | None = None
) -> tuple[opt.OptimisationResult, opt.OptimisationResult]:
    """Refine an installed three-sensor layout and compute the exhaustive optimum."""
    disp_keys: tuple[str, ...] = ("disp_x", "disp_y")

    # Build virtual sensor arrays.

    sim_dims: dict[str, tuple[float, float]] = sens.get_sim_dims(sim_data)
    sites: np.ndarray = sens.gen_pos_grid_inside(
        num_sensors=(3, 4, 1),
        x_lims=sim_dims["x"],
        y_lims=sim_dims["y"],
        z_lims=sim_dims["z"],
    )
    sites = opt.filter_sensor_sites_in_mesh(sites, sim_data)
    if sim_data.coords is None or sim_data.node_vars is None:
        raise ValueError("The dataset must include mesh coordinates and nodal fields.")
    target_positions: np.ndarray = sim_data.coords
    target_displacement: np.ndarray = sim_data.node_vars["disp_y"]
    displacement_data: sens.SensorData = sens.SensorData(positions=sites[:3])
    displacement: sens.SensorsPoint = sens.SensorFactory.vector_point(
        sim_data,
        displacement_data,
        comp_keys=disp_keys,
        spatial_dims=sens.EDim.TWOD,
        descriptor=sens.DescriptorFactory.displacement(),
    )
    sensor_arrays: dict[str, sens.ISensorArray] = {"displacement": displacement}

    # Define the placement problem.

    design_space: opt.DesignSpace = opt.DesignSpace(
        positions={"displacement": sites},
        position_axes={"displacement": opt.EDesignAxis.DISC},
    )
    objective: ObjNodalReconRMSE = ObjNodalReconRMSE(
        target_positions=target_positions,
        target_values={"nominal": target_displacement},
        sensor_keys=("displacement",),
        component_index=displacement.get_field().get_component_index("disp_y"),
    )
    # Run the placement search.

    exchange: opt.OptimisationResult = opt.SensorOptEngine(
        sensor_arrays=sensor_arrays,
        seed=seed,
        design_space=design_space,
        objectives=(objective,),
        optimiser=opt.OptimiserExchange(max_iterations=20),
    ).optimise()
    exhaustive: opt.OptimisationResult = opt.SensorOptEngine(
        sensor_arrays=sensor_arrays,
        seed=seed,
        design_space=design_space,
        objectives=(objective,),
        optimiser=opt.OptimiserExhaustive(num_sensors=3, max_iterations=220),
    ).optimise()
    return exchange, exhaustive


if __name__ == "__main__":
    # The plate-with-hole dataset uses millimetres and newtons.
    data_path: Path = dataset.mechanical_2d_path()
    simulation: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
    exchange_result, exhaustive_result = compare_optimisers(simulation)
    for search_result in (exchange_result, exhaustive_result):
        print(search_result.objective_values, search_result.stop_condition.name)
