# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Fixed-count layout refinement
=============================

Refine a three-sensor layout for shear strain and benchmark it by enumeration.
"""

from dataclasses import replace
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

output_path: Path = Path.cwd() / "pyvale-output" / "sensopt_ex04_fixed_count_refinement"
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
sites: np.ndarray = sens.gen_pos_grid_inside(
    num_sensors=(3, 4, 1),
    x_lims=sim_dims["x"],
    y_lims=sim_dims["y"],
    z_lims=sim_dims["z"],
)
sites = opt.filter_sensor_sites_in_mesh(sites, sim_data)
target_positions: np.ndarray = cast(np.ndarray, sim_data.coords)
nodal_fields: dict[str, np.ndarray] = cast(dict[str, np.ndarray], sim_data.node_vars)
target_strain: np.ndarray = nodal_fields[strain_key]
strain_data: sens.SensorData = sens.SensorData(positions=sites[:3])
strain_field = sens.FieldTensor(
    sim_data,
    norm_comp_keys=norm_comp_keys,
    dev_comp_keys=dev_comp_keys,
    spatial_dims=sens.EDim.TWOD,
)
strain: sens.SensorsPoint = sens.SensorsPoint(
    sensor_data=strain_data,
    field=strain_field,
    descriptor=sens.DescriptorFactory.strain(sens.EDim.TWOD),
)
sensor_arrays: dict[str, sens.ISensorArray] = {"strain": strain}

# %%
# 3. Define the placement problem
# -------------------------------

seed: int = 123
design_space: opt.DesignSpace = opt.DesignSpace(
    positions={"strain": sites},
    position_axes={"strain": opt.EDesignAxis.DISC},
)
objective: opt.ObjFieldRecon = opt.ObjFieldRecon(
    target_positions=target_positions,
    target_values={"nominal": target_strain},
    sensor_keys=("strain",),
    component_index=strain.get_field().get_component_index(strain_key),
)

# %%
# 4. Run the placement search
# ---------------------------

exchange: opt.OptResult = opt.SensorOptEngine(
    sensor_arrays=sensor_arrays,
    seed=seed,
    design_space=design_space,
    objectives=(objective,),
    optimiser=opt.OptExchange(max_iterations=20),
).optimise()

exhaustive: opt.OptResult = opt.SensorOptEngine(
    sensor_arrays=sensor_arrays,
    seed=seed,
    design_space=design_space,
    objectives=(objective,),
    optimiser=opt.OptExhaustive(num_sensors=3),
).optimise()

# Compare the best RMSE across counts with the fewest sensors meeting a target.
sensor_counts: range = range(1, 4)
target_rmse: float = 0.0008
exhaustive_modes: dict[str, opt.OptExhaustive] = {
    "exhaustive_range": opt.OptExhaustive(num_sensors=sensor_counts),
    "exhaustive_target": opt.OptExhaustive(
        num_sensors=sensor_counts, target_objective=target_rmse
    ),
}
exhaustive_results: dict[str, opt.OptResult] = {}
for name, optimiser in exhaustive_modes.items():
    exhaustive_results[name] = opt.SensorOptEngine(
        sensor_arrays=sensor_arrays,
        seed=seed,
        design_space=design_space,
        objectives=(objective,),
        optimiser=optimiser,
    ).optimise()

# Use the same exhaustive search to isolate the effect of the error metric.
max_error_objective = replace(objective, metric=opt.EErrMetric.MAX_ERR)
exhaustive_max_error: opt.OptResult = opt.SensorOptEngine(
    sensor_arrays=sensor_arrays,
    seed=seed,
    design_space=design_space,
    objectives=(max_error_objective,),
    optimiser=opt.OptExhaustive(num_sensors=3),
).optimise()

# %%
# 5. Inspect the result
# ---------------------

results: dict[str, opt.OptResult] = {
    "exchange": exchange,
    "exhaustive": exhaustive,
    **exhaustive_results,
    "exhaustive_max_error": exhaustive_max_error,
}
for name, result in results.items():
    selected_positions = result.sensor_data["strain"].positions
    count: int = 0 if selected_positions is None else len(selected_positions)
    print(
        f"{name}: sensor count {count}, score {result.objective_values[0]:.6g}, "
        f"{result.num_iterations} iterations ({result.stop_condition.name})"
    )


# %%
# 6. Visualise the selected sensors
# --------------------------------

for name, result in results.items():
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
    pv_plot.add_text(
        name.replace("_", " ").capitalize(), position="upper_right", font_size=14
    )
    image_path: Path = output_path / f"{name}_sensor_locations.png"
    if SHOW_PLOTS:
        pv_plot.show(screenshot=image_path)
    else:
        pv_plot.screenshot(image_path)
    pv_plot.close()
