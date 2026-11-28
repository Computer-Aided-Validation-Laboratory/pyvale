# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Optimise positions and times for whole-pulse reconstruction
===========================================================
"""

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pyvista as pv

import pyvale.dataio as io
import pyvale.sensoropt as opt
import pyvale.sensorsim as sens
from pyvale.examples.sensopt.thermal_pulse import build_thermal_pulse

# %%
# 1. Define candidate sites and five-sample schedules that cover the pulse
# -----------------------------------------------------------------------

plate: io.SimData = build_thermal_pulse()
sites: npt.NDArray[np.float64] = sens.gen_pos_grid_inside(
    num_sensors=(5, 4, 1), x_lims=(0.0, 10.0), y_lims=(0.0, 7.5), z_lims=(0.0, 0.0)
)
target_times = cast(npt.NDArray[np.float64], plate.time)
schedules = tuple(
    np.array(
        [target_times[0], target_times[14], time, target_times[27], target_times[-1]]
    )
    for time in target_times[16:23]
)
array_field = sens.FieldScalar(
    plate,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
)
array: sens.SensorsPoint = sens.SensorsPoint(
    sensor_data=sens.SensorData(positions=sites[:3]),
    field=array_field,
    descriptor=sens.DescriptorFactory.temperature(),
)

# %%
# 2. For each schedule, greedily place three sensors and score the entire pulse
# ----------------------------------------------------------------------------

recon: opt.ObjFieldRecon = opt.ObjFieldRecon(
    target_positions=cast(npt.NDArray[np.float64], plate.coords),
    target_times=target_times,
    reference_field="temperature",
    sensor_keys=("temperature",),
    component_index=0,
)
result: opt.OptResult = opt.SensorOptEngine(
    sensor_arrays={"temperature": array},
    design_space=opt.DesignSpace(
        positions={"temperature": sites},
        position_axes={"temperature": opt.EDesignAxis.DISC},
        sample_times={"temperature": schedules},
    ),
    objectives=(recon,),
    optimiser=opt.OptGreedyForward(num_sensors=3),
    seed=123,
).optimise()
selected: sens.SensorData = result.sensor_data["temperature"]
print(
    f"Selected times {selected.sample_times}, "
    f"whole-pulse RMSE {result.objective_values[0]:.4f}"
)
print(selected.positions)

# %%
# 3. Plot selected sites over the temperature field at the pulse peak
# ------------------------------------------------------------------

output_path: Path = Path.cwd() / "pyvale-output" / "sensopt_ex11_joint_pulse_recon"
output_path.mkdir(parents=True, exist_ok=True)
selected_array: sens.SensorsPoint = sens.SensorsPoint(
    selected, array.get_field(), array.get_descriptor()
)
plotter: pv.Plotter = sens.plot_point_sensors_on_sim(
    selected_array,
    "temperature",
    time_step=int(
        np.argmax(
            cast(dict[str, np.ndarray], plate.node_vars)["temperature"].max(axis=0)
        )
    ),
    vis_opts=sens.VisOptsSimSensors(interactive=False, show_edges=False),
)
plotter.screenshot(output_path / "joint_sensor_locations.png")
plotter.close()
