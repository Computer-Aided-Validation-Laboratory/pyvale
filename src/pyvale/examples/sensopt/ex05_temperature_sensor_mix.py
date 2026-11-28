# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Select a mix of temperature sensors
===================================

Pool measurements from two temperature arrays, balancing reconstruction
accuracy against the cost of the selected sensors.
"""

from pathlib import Path
from typing import cast

import numpy as np
import pyvista as pv

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensoropt as opt
import pyvale.sensorsim as sens

# %%
# 1. Load the thermomechanical plate and choose candidate sites
# ------------------------------------------------------------

SHOW_PLOTS = False
output_path: Path = Path.cwd() / "pyvale-output" / "sensopt_ex05_temperature_sensor_mix"
output_path.mkdir(parents=True, exist_ok=True)

data_path: Path = dataset.thermomechanical_2d_experiment_paths()[0]
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
sim_data = sens.scale_length_units(1000.0, sim_data, ("disp_x", "disp_y"))
dimensions: dict[str, tuple[float, float]] = sens.get_sim_dims(sim_data)
sites: np.ndarray = sens.gen_pos_grid_inside(
    num_sensors=(4, 3, 1),
    x_lims=dimensions["x"],
    y_lims=dimensions["y"],
    z_lims=dimensions["z"],
)
sites = opt.filter_sensor_sites_in_mesh(sites, sim_data)
temperature_key: str = "temperature"
nodal_fields: dict[str, np.ndarray] = cast(dict[str, np.ndarray], sim_data.node_vars)

# %%
# 2. Configure two sensor types and a shared reconstruction objective
# ------------------------------------------------------------------

sensor_arrays: dict[str, sens.SensorsPoint] = {}
unit_costs: dict[str, float] = {"precision": 3.0, "economy": 1.0}
biases: dict[str, float] = {"precision": 0.0, "economy": 8.0}
for sensor_key, bias in biases.items():
    array_field = sens.FieldScalar(
        sim_data,
        comp_key=temperature_key,
        spatial_dims=sens.EDim.TWOD,
    )
    array: sens.SensorsPoint = sens.SensorsPoint(
        sensor_data=sens.SensorData(positions=sites[:2]),
        field=array_field,
        descriptor=sens.DescriptorFactory.temperature(),
    )
    array.set_error_chain([sens.ErrSysOffset(offset=bias)])
    sensor_arrays[sensor_key] = array

temperature_recon: opt.ObjFieldRecon = opt.ObjFieldRecon(
    target_positions=cast(np.ndarray, sim_data.coords),
    target_values={"nominal": nodal_fields[temperature_key]},
    sensor_keys=tuple(sensor_arrays),
    component_index=0,
)
temperature_tolerance: float = 10.0
budget: float = 5.0
objective: opt.ObjWeighted = opt.ObjWeighted(
    terms=(
        opt.ObjTerm(
            objective=temperature_recon,
            normalisation=opt.ObjNormAffine(gain=1.0 / temperature_tolerance),
        ),
        opt.ObjTerm(
            objective=opt.ObjSensorCost(unit_costs=unit_costs),
            normalisation=opt.ObjNormAffine(gain=1.0 / budget),
            weight=0.5,
        ),
    ),
)
design_space: opt.DesignSpace = opt.DesignSpace(
    positions=dict.fromkeys(sensor_arrays, sites),
    position_axes=dict.fromkeys(sensor_arrays, opt.EDesignAxis.DISC),
    input_constraints=(
        opt.ConstrainSensSep(),
        opt.ConstrainSensCost(unit_costs=unit_costs, budget=budget),
    ),
)

# %%
# 3. Select three sensors across the two arrays
# ---------------------------------------------

result: opt.OptResult = opt.SensorOptEngine(
    sensor_arrays=sensor_arrays,
    design_space=design_space,
    objectives=objective,
    optimiser=opt.OptGreedyForward(num_sensors=3),
    seed=123,
).optimise()
print(f"Weighted temperature and cost score: {result.objective_values[0]:.6g}")
cost: float = 0.0
for sensor_key, selected_data in result.sensor_data.items():
    selected_positions = cast(np.ndarray, selected_data.positions)
    cost += len(selected_positions) * unit_costs[sensor_key]
    print(f"{sensor_key}: {len(selected_positions)} selected")
print(f"Total sensor cost: {cost:g} of {budget:g}")

# Keep a precision sensor already installed at the first candidate site.
fixed_start_result: opt.OptResult = opt.SensorOptEngine(
    sensor_arrays=sensor_arrays,
    design_space=design_space,
    objectives=objective,
    optimiser=opt.OptGreedyForward(
        num_sensors=3,
        start=opt.OptStartLayout(sites={"precision": (0,)}),
    ),
    seed=123,
).optimise()
print(
    f"Installed precision sensor: score {fixed_start_result.objective_values[0]:.6g}"
    f" after {fixed_start_result.num_iterations} additions"
)

# %%
# 4. Save the selected positions by sensor type
# ---------------------------------------------

layouts: dict[str, opt.OptResult] = {"": result, "fixed_start_": fixed_start_result}
for prefix, layout in layouts.items():
    for sensor_key, selected_data in layout.sensor_data.items():
        source: sens.SensorsPoint = sensor_arrays[sensor_key]
        selected_array: sens.SensorsPoint = sens.SensorsPoint(
            selected_data, source.get_field(), source.get_descriptor()
        )
        plotter: pv.Plotter = sens.plot_point_sensors_on_sim(
            selected_array,
            temperature_key,
            vis_opts=sens.VisOptsSimSensors(interactive=SHOW_PLOTS),
        )
        plotter.add_text(sensor_key.capitalize(), position="upper_right")
        image_path: Path = output_path / f"{prefix}{sensor_key}_sensor_locations.png"
        if SHOW_PLOTS:
            plotter.show(screenshot=image_path)
        else:
            plotter.screenshot(image_path)
        plotter.close()
