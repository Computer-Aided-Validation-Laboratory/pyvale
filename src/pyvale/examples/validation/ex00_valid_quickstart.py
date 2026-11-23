# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Validation quickstart
================================================================================
"""

from itertools import count
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.validation as vl

matplotlib.use("Agg")

# %%
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex00_quickstart"
output_path.mkdir(parents=True, exist_ok=True)
figure_numbers = count(1)

# %%
# 1. Load & transform data
# ------------------------------------------------------------
with np.load(dataset.valid_data_dir() / "plate_validation_2d.npz") as saved:
    coords: np.ndarray = saved["coords"].copy()
    times: np.ndarray = saved["times"].copy()
    sim_temp: np.ndarray = saved["sim_temp"].copy()
    exp_temp: np.ndarray = saved["exp_temp"].copy()
    sim_strain: np.ndarray = saved["sim_strain"] * 1.0e6
    exp_strain: np.ndarray = saved["exp_strain"] * 1.0e6

exp_temp_sort_order = np.argsort(exp_temp, axis=-1, stable=True)
exp_temp_ranks = np.argsort(exp_temp_sort_order, axis=-1)
nominal_temp_by_exp_rank = np.take_along_axis(
    np.sort(sim_temp[..., 0, :], axis=-1), exp_temp_ranks, axis=-1
)
exp_temp += 0.25 * (nominal_temp_by_exp_rank - exp_temp)

# %%
# 2. Setup a validation case
# ------------------------------------------------------------
temperature_tag = "plate_temperature"
strain_tag = "plate_strain"

observables = {
    temperature_tag: vl.ValidObservable(
        components=("temperature",),
        geometry=vl.EValidGeometry.POINT,
        coords=coords,
        times=times,
        phys_units="degC",
        coord_units="mm",
        time_units="s",
    ),
    strain_tag: vl.ValidObservable(
        components=("strain_xx", "strain_yy", "strain_xy"),
        geometry=vl.EValidGeometry.POINT,
        coords=coords,
        times=times,
        phys_units="µε",
        coord_units="mm",
        time_units="s",
    ),
}

sim_axes = (
    vl.EValidAxis.POINT,
    vl.EValidAxis.COMPONENT,
    vl.EValidAxis.TIME,
    vl.EValidAxis.EPISTEMIC_TRANS,
    vl.EValidAxis.ALEATORY,
)
sim_arrays = {
    temperature_tag: vl.ValidArray(sim_temp, sim_axes),
    strain_tag: vl.ValidArray(sim_strain, sim_axes),
}

exp_axes = (
    vl.EValidAxis.POINT,
    vl.EValidAxis.COMPONENT,
    vl.EValidAxis.TIME,
    vl.EValidAxis.ALEATORY,
)
exp_arrays = {
    temperature_tag: vl.ValidArray(
        exp_temp,
        exp_axes,
        offsets=vl.EpistemicOffsets(lower=0.5, upper=0.75),
    ),
    strain_tag: vl.ValidArray(
        exp_strain,
        exp_axes,
        offsets=vl.EpistemicOffsets(lower=4.0e-6, upper=6.0e-6),
    ),
}

valid_case = vl.ValidCase(observables, sim_arrays, exp_arrays)

# %%
# 3. Run validation analysis & inspect output
# ------------------------------------------------------------
valid_result: vl.ValidResult = vl.ValidEngine(
    [vl.MetricAVM(), vl.MetricMAVM()]
).run(valid_case)

metric_field: vl.MetricField = valid_result.metrics[temperature_tag][
    vl.EMetricType.MAVM
]
metric_shape = metric_field.outputs[vl.EMetricOut.D_PLUS].shape
metric_axes = metric_field.axes

print("Simulation axes:", sim_arrays[temperature_tag].axes)
print("Simulation shape:", sim_arrays[temperature_tag].data.shape)
print("Result axes:", metric_axes, "shape:", metric_shape)

# %%
# 4. Visualisation & analysis
# ------------------------------------------------------------

point_idxs = tuple(range(len(coords)))
ecdf_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
    valid_case,
    temperature_tag,
    point_idxs=point_idxs,
    time_idx=4,
    valid_plot_opts=vl.ValidPlotOpts(x_label="Temperature [degC]"),
)
ecdf_plot.figure.savefig(
    output_path / f"val_ex00_fig{next(figure_numbers):02d}_temperature_cdfs.png",
    bbox_inches="tight",
)

# %%
ecdf_plot: vl.ECDFPlot = vl.plot_exp_ecdfs(
    valid_case,
    temperature_tag,
    time_idx=4,
    valid_plot_opts=vl.ValidPlotOpts(x_label="Temperature [degC]"),
    point_idxs=None,
    offset=None,
)
ecdf_plot.figure.savefig(
    output_path / f"val_ex00_fig{next(figure_numbers):02d}_temperature_envelopes.png",
    bbox_inches="tight",
)

# %%
fig, _ = vl.plot_metric_time_points(
    valid_result,
    temperature_tag,
    vl.EMetricType.MAVM,
    point_idxs=point_idxs,
    selectors={
        vl.EValidResAxis.SIM_TRANS: 0,
        vl.EValidResAxis.EXP_OFFSET: vl.EEpisOffset.UPPER,
    },
)
fig.savefig(
    output_path / f"val_ex00_fig{next(figure_numbers):02d}_temperature_mavm_time.png",
    bbox_inches="tight",
)

# %%
sim_path = dataset.thermomechanical_2d_experiment_paths()[0]
sim_data: io.SimData = mh.ExodusLoader(sim_path).load_all_sim_data()
sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y"))

background_time_idx = sim_data.time.size - 1

for oo, dd in (
    (vl.EMetricOut.D_PLUS, "d_plus"),
    (vl.EMetricOut.D_MINUS, "d_minus"),
):
    pv_plot: pv.Plotter = vl.plot_metric_on_sim(
        valid_result,
        temperature_tag,
        vl.EMetricType.MAVM,
        selectors={
            vl.EValidResAxis.SIM_TRANS: 0,
            vl.EValidResAxis.TIME: 4,
            vl.EValidResAxis.EXP_OFFSET: vl.EEpisOffset.UPPER,
        },
        output=oo,
        sim_data=sim_data,
        sim_field="temperature",
        sim_time_idx=background_time_idx,
        descriptor=sens.SensorDescriptor(
            name="Temperature", symbol=r"T", units="degC", tag="TC"
        ),
        sim_coord_units="mm",
        vis_opts=vl.create_valid_scene_opts(interactive=SHOW_PLOTS),
        metric_opts=vl.MetricVisOpts(labels_always_visible=True),
    )
    pv_plot.screenshot(
        output_path
        / (
            f"val_ex00_fig{next(figure_numbers):02d}_"
            f"temperature_mavm_{dd}_on_sim.png"
        ),
    )

# %%
if SHOW_PLOTS:
    pv_plot.show()
    plt.show()
else:
    plt.close("all")

pv_plot.close()
