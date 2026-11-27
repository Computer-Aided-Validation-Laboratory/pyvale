# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Joint temperature and displacement reconstruction
=================================================

Place both sensor types with a shared spacing constraint. Score each field
against its own engineering tolerance before combining the reconstruction errors.
"""

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pyvista as pv

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensopt as opt
import pyvale.sensorsim as sens

# %%
# 1. Load the plate and choose candidate sites
# -------------------------------------------

SHOW_PLOTS = False
output_path: Path = (
    Path.cwd() / "pyvale-output" / "sensopt_ex06_thermomechanical_placement"
)
output_path.mkdir(parents=True, exist_ok=True)

data_path: Path = dataset.thermomechanical_2d_experiment_paths()[0]
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
displacement_components: tuple[str, ...] = ("disp_x", "disp_y")
sim_data = sens.scale_length_units(1000.0, sim_data, displacement_components)
dimensions: dict[str, tuple[float, float]] = sens.get_sim_dims(sim_data)
sites: np.ndarray = sens.gen_pos_grid_inside(
    num_sensors=(3, 3, 1),
    x_lims=dimensions["x"],
    y_lims=dimensions["y"],
    z_lims=dimensions["z"],
)
sites = opt.filter_sensor_sites_in_mesh(sites, sim_data)
nodal_fields: dict[str, np.ndarray] = cast(dict[str, np.ndarray], sim_data.node_vars)
target_positions: np.ndarray = cast(np.ndarray, sim_data.coords)

# %%
# 2. Define the two sensor arrays and the weighted reconstruction score
# --------------------------------------------------------------------

temperature: sens.SensorsPoint = sens.SensorFactory.scalar_point(
    sim_data,
    sens.SensorData(positions=sites[:2]),
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.temperature(),
)
displacement: sens.SensorsPoint = sens.SensorFactory.vector_point(
    sim_data,
    sens.SensorData(positions=sites[:2]),
    comp_keys=displacement_components,
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.displacement(),
)
sensor_arrays: dict[str, sens.SensorsPoint] = {
    "temperature": temperature,
    "displacement": displacement,
}
field_components: dict[str, str] = {
    "temperature": "temperature",
    "displacement": "disp_y",
}
rmse_tolerances: dict[str, float] = {"temperature": 10.0, "displacement": 0.01}
field_objectives: dict[str, opt.ObjFieldRecon] = {
    key: opt.ObjFieldRecon(
        target_positions=target_positions,
        target_values={"nominal": nodal_fields[component]},
        sensor_keys=(key,),
        component_index=sensor_arrays[key].get_field().get_component_index(component),
    )
    for key, component in field_components.items()
}
objective: opt.ObjWeighted = opt.ObjWeighted(
    terms=tuple(
        opt.ObjTerm(
            objective=field_objective,
            normalisation=opt.ObjNormAffine(gain=1.0 / rmse_tolerances[key]),
        )
        for key, field_objective in field_objectives.items()
    ),
)

# %%
# 3. Enumerate feasible layouts with both fields represented
# ----------------------------------------------------------

# Each type has two available sensors. A three-sensor layout uses both types.
minimum_spacing: float = 0.3 * min(
    dimensions["x"][1] - dimensions["x"][0],
    dimensions["y"][1] - dimensions["y"][0],
)
design_space: opt.DesignSpace = opt.DesignSpace(
    positions={key: sites for key in sensor_arrays},
    position_axes={key: opt.EDesignAxis.DISC for key in sensor_arrays},
    input_constraints=(opt.ConstraintSensorSeparation(min_distance=minimum_spacing),),
)
result: opt.OptResult = opt.SensorOptEngine(
    sensor_arrays=sensor_arrays,
    design_space=design_space,
    objectives=objective,
    optimiser=opt.OptExhaustive(num_sensors=3),
    seed=123,
).optimise()
print(f"Weighted temperature and displacement score: {result.objective_values[0]:.6g}")
print(f"Minimum allowed spacing: {minimum_spacing:.6g} mm")
for sensor_key, selected_data in result.sensor_data.items():
    print(f"{sensor_key}: {len(cast(np.ndarray, selected_data.positions))} selected")

# %%
# 4. Compare starting strategies for greedy placement
# ---------------------------------------------------

start_counts: dict[str, int] = {"temperature": 1, "displacement": 1}
greedy_optimisers: dict[str, opt.OptGreedyForward] = {
    "exhaustive_start": opt.OptGreedyForward(
        num_sensors=3, start=opt.OptStartExhaustive(counts=start_counts)
    ),
    "random_start": opt.OptGreedyForward(
        num_sensors=3,
        start=opt.OptStartRandom(counts=start_counts, num_layouts=12),
    ),
    "empty_score": opt.OptGreedyForward(num_sensors=3),
}

# With no starting pair, score an unmeasured field at twenty tolerance units.
# The fallback is specified in the field's original units.
objective_with_empty_scores: opt.ObjWeighted = opt.ObjWeighted(
    terms=tuple(
        replace(term, empty_score=20.0 * rmse_tolerances[key])
        for key, term in zip(field_objectives, objective.terms, strict=True)
    )
)
greedy_results: dict[str, opt.OptResult] = {}
for name, optimiser in greedy_optimisers.items():
    score_objective: opt.ObjWeighted = (
        objective_with_empty_scores if optimiser.start is None else objective
    )
    greedy_result: opt.OptResult = opt.SensorOptEngine(
        sensor_arrays=sensor_arrays,
        design_space=design_space,
        objectives=score_objective,
        optimiser=optimiser,
        seed=123,
    ).optimise()
    greedy_results[name] = greedy_result
    print(
        f"{name}: score {greedy_result.objective_values[0]:.6g}, "
        f"{greedy_result.num_start_layouts} starts tested, "
        f"addition rounds: {greedy_result.num_iterations}"
    )

# %%
# 5. Save the locations over the reconstructed field components
# ------------------------------------------------------------

layouts: dict[str, opt.OptResult] = {
    "": result,
    **{f"{name}_": layout for name, layout in greedy_results.items()},
}
for prefix, layout in layouts.items():
    for sensor_key, selected_data in layout.sensor_data.items():
        source: sens.SensorsPoint = sensor_arrays[sensor_key]
        selected_array: sens.SensorsPoint = sens.SensorsPoint(
            selected_data, source.get_field(), source.get_descriptor()
        )
        plotter: pv.Plotter = sens.plot_point_sensors_on_sim(
            selected_array,
            field_components[sensor_key],
            vis_opts=sens.VisOptsSimSensors(interactive=SHOW_PLOTS),
        )
        plotter.add_text(sensor_key.capitalize(), position="upper_right")
        image_path: Path = output_path / f"{prefix}{sensor_key}_sensor_locations.png"
        if SHOW_PLOTS:
            plotter.show(screenshot=image_path)
        else:
            plotter.screenshot(image_path)
        plotter.close()
