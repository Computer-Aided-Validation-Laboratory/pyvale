# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Greedy sensor placement
=======================

Compare three signal criteria for placing five normal-strain sensors.
"""

from pathlib import Path

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

output_path: Path = Path.cwd() / "pyvale-output" / "sensopt_ex01_greedy_placement"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load physics simulation data
# -------------------------------

# The plate-with-hole dataset uses millimetres and newtons.
data_path: Path = dataset.mechanical_2d_path()
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
strain_key: str = "strain_yy"

# %%
# 2. Build virtual sensor arrays
# ------------------------------

sim_dims: dict[str, tuple[float, float]] = sens.get_sim_dims(sim_data)
sites: np.ndarray = sens.gen_pos_grid_inside(
    num_sensors=(14, 18, 1),
    x_lims=sim_dims["x"],
    y_lims=sim_dims["y"],
    z_lims=sim_dims["z"],
)
sites = opt.filter_sensor_sites_in_mesh(sites, sim_data)
strain_data: sens.SensorData = sens.SensorData(positions=sites[:5])
strain: sens.SensorsPoint = sens.SensorFactory.scalar_point(
    sim_data,
    strain_data,
    comp_key=strain_key,
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.SensorDescriptor(
        name="Strain", symbol=r"\varepsilon", tag="SG", components=("yy",)
    ),
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
# Once the peak site is selected, extra sites can tie under the peak criterion.
objectives: dict[str, opt.IObjFunc] = {
    "rms": opt.ObjSignalRMS(sensor_keys=("strain",)),
    "peak_abs": opt.ObjSignalPeakAbs(sensor_keys=("strain",)),
    "final_mean": opt.ObjSignal(
        sensor_keys=("strain",),
        time_index=-1,
        reductions=(
            opt.SignalReduction(
                axis=opt.ESignalAxis.COMPONENT, method=opt.ESignalReduce.L2
            ),
            opt.SignalReduction(
                axis=opt.ESignalAxis.POSITION, method=opt.ESignalReduce.MEAN
            ),
        ),
    ),
}
optimiser: opt.OptGreedyForward = opt.OptGreedyForward(num_sensors=5)

# %%
# 4. Run the placement searches
# -----------------------------

results: dict[str, opt.OptResult] = {}
for name, objective in objectives.items():
    engine: opt.SensorOptEngine = opt.SensorOptEngine(
        sensor_arrays=sensor_arrays,
        seed=seed,
        design_space=design_space,
        objectives=(objective,),
        optimiser=optimiser,
    )
    results[name] = engine.optimise()

result: opt.OptResult = results["rms"]

# %%
# 5. Inspect the result
# ---------------------

for name, placement in results.items():
    print(name, placement.sensor_data)
    print(placement.objective_values, placement.stop_condition.name)


# %%
# 6. Visualise the selected sensors
# --------------------------------

for name, placement in results.items():
    selected_sensors: sens.SensorsPoint = sens.SensorsPoint(
        placement.sensor_data["strain"],
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
    file_name = (
        "sensor_locations.png" if name == "rms" else f"{name}_sensor_locations.png"
    )
    image_path: Path = output_path / file_name
    if SHOW_PLOTS:
        pv_plot.show(screenshot=image_path)
    else:
        pv_plot.screenshot(image_path)
    pv_plot.close()
