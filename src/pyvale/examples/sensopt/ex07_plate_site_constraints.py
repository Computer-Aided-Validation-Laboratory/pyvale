# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Sensor placement with a plate exclusion zone
===========================================

Compare peak axial strain and whole-field reconstruction when sensors cannot
be mounted near the high-strain region on the right side of the hole.
"""

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pyvista as pv

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensoropt as opt
import pyvale.sensorsim as sens

# %%
# 1. Load the plate and select candidate sites in the mesh
# -------------------------------------------------------

SHOW_PLOTS = False
output_path: Path = Path.cwd() / "pyvale-output" / "sensopt_ex07_plate_site_constraints"
output_path.mkdir(parents=True, exist_ok=True)

data_path: Path = dataset.mechanical_2d_path()
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
strain_key: str = "strain_yy"
x_sites: npt.NDArray[np.float64] = np.array(
    [10.0, 20.0, 30.0, 35.0, 40.0, 50.0, 60.0, 63.0, 70.0, 80.0, 90.0]
)
y_sites: npt.NDArray[np.float64] = np.arange(15.0, 140.0, 5.0)
grid_xyz: npt.NDArray[np.float64] = np.stack(
    np.meshgrid(x_sites, y_sites, np.array([0.0]), indexing="ij"), axis=-1
)
sites: npt.NDArray[np.float64] = opt.filter_sensor_sites_in_mesh(
    grid_xyz.reshape(-1, 3), sim_data
)
target_positions: np.ndarray = cast(np.ndarray, sim_data.coords)
nodal_fields: dict[str, np.ndarray] = cast(dict[str, np.ndarray], sim_data.node_vars)

# %%
# 2. Define sensor inventory and exclude the right-hand strain hotspot
# -------------------------------------------------------------------

strain: sens.SensorsPoint = sens.SensorFactory.scalar_point(
    sim_data,
    sens.SensorData(positions=sites[:4]),
    comp_key=strain_key,
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.SensorDescriptor(
        name="Strain", symbol=r"\varepsilon", tag="SG", components=("yy",)
    ),
)

# The exclusion box is in the same millimetre coordinates as the plate.
exclusion_x: tuple[float, float] = (60.0, 70.0)
exclusion_y: tuple[float, float] = (65.0, 85.0)
exclusion: opt.SensSiteBounds = opt.SensSiteBounds(x=exclusion_x, y=exclusion_y)
site_constraint: opt.ConstrainSensSites = opt.ConstrainSensSites(
    default=opt.SensSiteNot(selector=exclusion),
)

design_space: opt.DesignSpace = opt.DesignSpace(
    positions={"strain": sites},
    position_axes={"strain": opt.EDesignAxis.DISC},
)
design_names: tuple[str, ...] = ("unrestricted", "excluded")
design_spaces: dict[str, opt.DesignSpace] = dict(
    zip(
        design_names,
        (design_space, replace(design_space, input_constraints=(site_constraint,))),
        strict=True,
    )
)

# %%
# 3. Run peak-signal and reconstruction searches with and without the exclusion
# ----------------------------------------------------------------------------

# Select the peak site for axial strain only, then compare four-sensor layouts
# that reconstruct the complete axial-strain field over all simulation times.
peak_signal: opt.ObjSignalPeakAbs = opt.ObjSignalPeakAbs(sensor_keys=("strain",))
field_recon: opt.ObjFieldRecon = opt.ObjFieldRecon(
    target_positions=target_positions,
    target_values={"nominal": nodal_fields[strain_key]},
    sensor_keys=("strain",),
    component_index=strain.get_field().get_component_index(strain_key),
)
objectives: dict[str, tuple[opt.IObjFunc, int]] = {
    "peak_signal": (peak_signal, 1),
    "field_recon": (field_recon, 4),
}
results: dict[tuple[str, str], opt.OptResult] = {}
for objective_name, (objective, num_sensors) in objectives.items():
    for design_name, space in design_spaces.items():
        result: opt.OptResult = opt.SensorOptEngine(
            sensor_arrays={"strain": strain},
            design_space=space,
            objectives=(objective,),
            optimiser=opt.OptGreedyForward(num_sensors=num_sensors),
            seed=123,
        ).optimise()
        results[objective_name, design_name] = result
        score: float = float(result.objective_values[0])
        if objective_name == "peak_signal":
            print(f"{design_name}: peak axial strain = {-score * 1e6:.1f} microstrain")
        else:
            print(f"{design_name}: reconstruction RMSE = {score * 1e6:.1f} microstrain")

# %%
# 4. Plot the selected sensors and mark the excluded region
# ---------------------------------------------------------

for (objective_name, design_name), result in results.items():
    selected: sens.SensorsPoint = sens.SensorsPoint(
        result.sensor_data["strain"], strain.get_field(), strain.get_descriptor()
    )
    plotter: pv.Plotter = sens.plot_point_sensors_on_sim(
        selected,
        strain_key,
        vis_opts=sens.VisOptsSimSensors(interactive=SHOW_PLOTS, show_edges=False),
    )
    plotter.add_text(
        f"{objective_name.replace('_', ' ').capitalize()} | {design_name}",
        position="upper_right",
        font_size=12,
    )
    if design_name == "excluded":
        region: pv.PolyData = pv.Box(bounds=(*exclusion_x, *exclusion_y, 0.1, 0.2))
        plotter.add_mesh(region, style="wireframe", color="orange", line_width=3)
    image_path: Path = output_path / f"{objective_name}_{design_name}.png"
    if SHOW_PLOTS:
        plotter.show(screenshot=image_path)
    else:
        plotter.screenshot(image_path)
    plotter.close()
