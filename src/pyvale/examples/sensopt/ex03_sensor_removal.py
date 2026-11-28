# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Greedy sensor removal
=====================

Reduce an existing sensor layout while minimising shear strain reconstruction error.
"""

from pathlib import Path
from typing import cast

import numpy as np
import pyvista as pv

# pyvale imports
import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensoropt as opt
import pyvale.sensorsim as sens

# %%

SHOW_PLOTS = False

output_path: Path = Path.cwd() / "pyvale-output" / "sensopt_ex03_sensor_removal"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load physics simulation data
# -------------------------------

# The plate-with-hole dataset uses millimetres and newtons.
data_path: Path = dataset.mechanical_2d_path()
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
strain_key: str = "strain_xy"
norm_comp_keys: tuple[str, ...] = ("strain_xx", "strain_yy")
dev_comp_keys: tuple[str, ...] = ("strain_xy",)

# %%
# 2. Build virtual sensor arrays
# ------------------------------

sim_dims: dict[str, tuple[float, float]] = sens.get_sim_dims(sim_data)
installed_positions: np.ndarray = sens.gen_pos_grid_inside(
    num_sensors=(3, 4, 1),
    x_lims=sim_dims["x"],
    y_lims=sim_dims["y"],
    z_lims=sim_dims["z"],
)
installed_positions = opt.filter_sensor_sites_in_mesh(installed_positions, sim_data)
target_positions: np.ndarray = cast(np.ndarray, sim_data.coords)
nodal_fields: dict[str, np.ndarray] = cast(dict[str, np.ndarray], sim_data.node_vars)
target_strain: np.ndarray = nodal_fields[strain_key]
strain_data: sens.SensorData = sens.SensorData(positions=installed_positions)
strain: sens.SensorsPoint = sens.SensorFactory.tensor_point(
    sim_data,
    strain_data,
    norm_comp_keys=norm_comp_keys,
    dev_comp_keys=dev_comp_keys,
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.strain(sens.EDim.TWOD),
)
sensor_arrays: dict[str, sens.ISensorArray] = {"strain": strain}

# %%
# 3. Define the placement problem
# -------------------------------

seed: int = 123
objective: opt.ObjFieldRecon = opt.ObjFieldRecon(
    target_positions=target_positions,
    target_values={"nominal": target_strain},
    sensor_keys=("strain",),
    component_index=strain.get_field().get_component_index(strain_key),
)
design_space: opt.DesignSpace = opt.DesignSpace(
    positions={"strain": installed_positions},
    position_axes={"strain": opt.EDesignAxis.DISC},
)
optimiser: opt.OptGreedyBackward = opt.OptGreedyBackward(num_sensors=5)
engine: opt.SensorOptEngine = opt.SensorOptEngine(
    sensor_arrays=sensor_arrays,
    seed=seed,
    design_space=design_space,
    objectives=(objective,),
    optimiser=optimiser,
)

# %%
# 4. Run the placement search
# ---------------------------

result: opt.OptResult = engine.optimise()

# %%
# 5. Inspect the result
# ---------------------

print(result.sensor_data)
print(result.objective_values, result.stop_condition.name)


# %%
# 6. Visualise the selected sensors
# --------------------------------

selected_sensors: sens.SensorsPoint = sens.SensorsPoint(
    result.sensor_data["strain"],
    strain.get_field(),
    strain.get_descriptor(),
)
pv_plot: pv.Plotter = sens.plot_point_sensors_on_sim(
    selected_sensors,
    strain_key,
    vis_opts=sens.VisOptsSimSensors(
        interactive=SHOW_PLOTS,
        show_edges=False,
        sens_label_font_size=14,
        sens_point_size=12.0,
    ),
)
image_path: Path = output_path / "sensor_locations.png"
if SHOW_PLOTS:
    pv_plot.show(screenshot=image_path)
else:
    pv_plot.screenshot(image_path)
pv_plot.close()
