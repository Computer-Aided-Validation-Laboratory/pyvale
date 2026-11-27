"""
Greedy sensor placement
=======================

Place five displacement sensors on a plate with a hole using the proposed API.
"""

from pathlib import Path

import numpy as np

# pyvale imports
import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensopt as opt
import pyvale.sensorsim as sens


def optimise_placement(
    sim_data: io.SimData, *, seed: int | None = None
) -> opt.OptimisationResult:
    """Select five sites with the strongest nominal displacement signal."""
    disp_keys: tuple[str, ...] = ("disp_x", "disp_y")

    # Build virtual sensor arrays.

    sim_dims: dict[str, tuple[float, float]] = sens.get_sim_dims(sim_data)
    sites: np.ndarray = sens.gen_pos_grid_inside(
        num_sensors=(14, 18, 1),
        x_lims=sim_dims["x"],
        y_lims=sim_dims["y"],
        z_lims=sim_dims["z"],
    )
    sites = opt.filter_sensor_sites_in_mesh(sites, sim_data)
    displacement_data: sens.SensorData = sens.SensorData(positions=sites[:5])
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
    objective: opt.IObjFunc = opt.ObjSignal(sensor_keys=("displacement",))
    optimiser: opt.OptimiserGreedyForward = opt.OptimiserGreedyForward(
        num_sensors=5, max_iterations=5
    )
    engine: opt.SensorOptEngine = opt.SensorOptEngine(
        sensor_arrays=sensor_arrays,
        seed=seed,
        design_space=design_space,
        objectives=(objective,),
        optimiser=optimiser,
    )

    # Run the placement search.

    result: opt.OptimisationResult = engine.optimise()
    return result


if __name__ == "__main__":
    # The plate-with-hole dataset uses millimetres and newtons.
    data_path: Path = dataset.mechanical_2d_path()
    simulation: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
    placement: opt.OptimisationResult = optimise_placement(simulation)
    print(placement.sensor_data)
    print(placement.objective_values, placement.stop_condition.name)
