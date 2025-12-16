# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Experiment simulation in 2D
================================================================================

In previous examples we have built our virtual sensor array and used this to
run a single simulated experiment. However, we will generally want to run many
simulated experiments and perform statistical analysis on the results. In this
example we demonstrate how `pyvale` can be used to run a set of simulated
experiments with a series of sensor arrays, one measuring temperature and the
other measuring displacement. We also show how this analysis can be performed
over a set of input physics simulations with different parameters.

Note that this example has minimal explanation and assumes you have reviewed the
basic sensor simulation examples to understand how the underlying engine works
as well as the sensor simulation workflow.
"""

import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# pyvale imports
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.dataset as dataset

#%%
# 1. Load physics simulation data
# -------------------------------

sim_paths: list[Path] = dataset.thermomechanical_2d_experiment_paths()
sim_keys: set[str] = {"sim_nominal","sim_perturbed"}

disp_keys = ("disp_x","disp_y")

sim_data_dict: dict[str,mh.SimData] = {}
for ss,kk in zip(sim_paths,sim_keys):
    sim_data = mh.ExodusLoader(ss).load_all_sim_data()
    sim_data = sens.scale_length_units(scale=1000.0,
                                       sim_data=sim_data,
                                       disp_keys=disp_keys)
    sim_data_dict[kk] = sim_data

#%%
# 2. Build virtual sensor arrays
# ------------------------------

sim_dims: dict[str,tuple[float,float]] = sens.simtools.get_sim_dims(sim_data)

sample_times = np.linspace(0.0,np.max(sim_data.time),50)

#%%
# 2.1 Build scalar field sensor array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

temp_sens_pos: np.ndarray = sens.gen_pos_grid_inside(num_sensors=(4,1,1),
                                                     x_lims=sim_dims["x"],
                                                     y_lims=sim_dims["y"],
                                                     z_lims=(0.0,0.0))

temp_sens_data = sens.SensorData(positions=temp_sens_pos,
                                 sample_times=sample_times)

temp_sens: sens.SensorsPoint = sens.SensorFactory.scalar_point(
    sim_data,
    temp_sens_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.temperature(),
)


#%%
# 2.2 Add errors to the scalar field sensors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

temp_pos_uncert = 1.0 # units = mm
temp_pos_rand = (sens.GenUniform(low=-temp_pos_uncert,high=temp_pos_uncert),
                 sens.GenUniform(low=-temp_pos_uncert,high=temp_pos_uncert),
                 None)

temp_field_err_data = sens.ErrFieldData(pos_rand_xyz=temp_pos_rand)

temp_err_chain: list[sens.IErrSimulator] = [
    sens.ErrSysField(temp_sens.get_field(),temp_field_err_data),
    sens.ErrRandGenPercent(sens.GenNormal(std=1.0),
                           err_dep=sens.EErrDep.DEPENDENT),
    sens.ErrSysOffsetPercent(offset_percent=-1.0),
    sens.ErrSysDigitisation(bits_per_unit=2**24/100),
    sens.ErrSysSaturation(meas_min=0.0,meas_max=700.0),
]

temp_sens.set_error_chain(temp_err_chain)

#%%
# 2.3 Build vector field sensor array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
disp_sens_pos: np.ndarray = sens.gen_pos_grid_inside(num_sensors=(2,2,1),
                                                     x_lims=sim_dims["x"],
                                                     y_lims=sim_dims["y"],
                                                     z_lims=(0.0,0.0))

sens_angles: tuple[Rotation] = (
    Rotation.from_euler("zyx",[0,0,0], degrees=True),
)

disp_sens_data = sens.SensorData(positions=disp_sens_pos,
                                 sample_times=sample_times)

disp_sens: sens.SensorsPoint = sens.SensorFactory.vector_point(
    sim_data,
    disp_sens_data,
    comp_keys=disp_keys,
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.displacement(),
)

#%%
# 2.4 Add errors to the vector field sensors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pos_rand = sens.GenUniform(low=-1.0,high=1.0)   # units = mm
angle_rand = sens.GenUniform(low=-2.0,high=2.0) # units = degrees

field_err_data = sens.ErrFieldData(pos_rand_xyz=(pos_rand,pos_rand,None),
                                   ang_rand_zyx=(angle_rand,None,None))

disp_err_chain: list[sens.IErrSimulator] = [
    sens.ErrRandGenPercent(sens.GenNormal(std=1.0)),
    sens.ErrSysField(disp_sens.get_field(),field_err_data),
    sens.ErrSysOffsetPercent(offset_percent=1.0),
    sens.ErrSysDigitisation(bits_per_unit=2**24/1.0),
    sens.ErrSysSaturation(meas_min=-5.0,meas_max=5.0),
]

disp_sens.set_error_chain(disp_err_chain)

#%%
# 3. Create & run simulated experiments
# -------------------------------------

sensor_arrays: dict[str,sens.ISensorArray] = {
    "temp": temp_sens,
    "disp": disp_sens,
}

exp_sim_opts = sens.ExpSimOpts(workers=4,para=sens.EExpSimPara.ALL)
exp_save_keys = sens.ExpSimSaveKeys(pert_sens_times=None)

exp_sim = sens.ExperimentSimulator(sim_data_dict,
                                   sensor_arrays,
                                   exp_sim_opts,
                                   exp_save_keys)

start_exp: float = time.perf_counter()
exp_data: dict[tuple[str,...],np.ndarray] = (
    exp_sim.run_experiments(num_exp_per_sim=1000)
)
exp_time: float = time.perf_counter() - start_exp

start_stats: float = time.perf_counter()
exp_stats: dict[tuple[str,...],sens.ExpSimStats] = (
    sens.calc_exp_sim_stats(exp_data)
)
stats_time: float = time.perf_counter() - start_stats

print(80*"-")
print(f"Exp. sim. time    = {exp_time:.3f} seconds")
print(f"Stats. calc. time = {stats_time:.3f} seconds")
print(80*"-")
print()

# %%
# .. image:: ../../../../_static/ext_ex5a_term_out_0.png
#    :alt: Terminal output showing the simulated experiment times
#    :width: 700px
#    :align: center


#%%
# 4. Analyse & visualise the results
# ----------------------------------

print(80*"-")
print("Keys in the simulated experimental data dictionary:")
for kk in exp_data:
    print(kk)
print(80*"-")
print()

# %%
# .. image:: ../../../../_static/ext_ex5a_term_out_1.png
#    :alt: Terminal output showing the keys of the experiment data dictionary
#    :width: 700px
#    :align: center

print(80*"-")
print("Thermal sensor array:")
print()
print(f"    {exp_data[('sim_nominal','temp','meas')].shape=}")
print("    shape=(n_exps,n_sensors,n_field_comps,n_time_steps)")
print()
print(f"    {exp_stats[('sim_nominal','temp','meas')].max.shape=}")
print("    shape=(n_sensors,n_field_comps,n_time_steps)")
print()
print(f"    {exp_data[('sim_nominal','temp','pert_sens_pos')].shape=}")
print("    shape=(n_exps,n_sensors,coord[X,Y,Z])")
print()
print(80*"-")
print("Mechanical sensor array:")
print()
print(f"    {exp_data[('sim_nominal','disp','meas')].shape=}")
print("    shape=(n_exps,n_sensors,n_field_comps,n_time_steps)")
print()
print(f"    {exp_stats[('sim_nominal','disp','meas')].max.shape=}")
print("    shape=(n_sensors,n_field_comps,n_time_steps)")
print()
print(f"    {exp_data[('sim_nominal','disp','pert_sens_pos')].shape=}")
print("    shape=(n_exps,n_sensors,coord[X,Y,Z])")
print()
print(80*"-")

# %%
# .. image:: ../../../../_static/ext_ex5a_term_out_2.png
#    :alt: Terminal output showing the simulated measurement arrays
#    :width: 700px
#    :align: center

output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)


sens.save_exp_sim_data(output_path/"ex5a_exp_sim_data.npz",exp_data)
exp_data = sens.load_exp_sim_data(output_path/"ex5a_exp_sim_data.npz")


pert_sens_pos = exp_data[("sim_nominal","temp","pert_sens_pos")][-1,:,:]
pv_plot = sens.plot_point_sensors_on_sim(sensor_array=temp_sens,
                                         comp_key="temperature",
                                         time_step=-1,
                                         perturbed_sens_pos=pert_sens_pos)
pv_plot.camera_position = "xy"

# Set to False to show an interactive plot instead of saving the figure
pv_plot.off_screen = True
if pv_plot.off_screen:
    pv_plot.screenshot(output_path/"ext_ex5a_temp_locs.png")
else:
    pv_plot.show()

# %%
# Visualisation of the virtual temperature sensor locations:
#
# .. image:: ../../../../_static/ext_ex5a_temp_locs.png
#    :alt: Visualisation of the virtual temperature sensor locations
#    :width: 800px
#    :align: center

pert_sens_pos = exp_data[("sim_nominal","disp","pert_sens_pos")][-1,:,:]
pv_plot = sens.plot_point_sensors_on_sim(sensor_array=disp_sens,
                                         comp_key="disp_x",
                                         time_step=-1,
                                         perturbed_sens_pos=pert_sens_pos)
pv_plot.camera_position = "xy"

# Set to False to show an interactive plot instead of saving the figure
pv_plot.off_screen = True
if pv_plot.off_screen:
    pv_plot.screenshot(output_path/"ext_ex5a_disp_locs.png")
else:
    pv_plot.show()

# Uncomment to show interactive figure and set off_screen = False above
# pv_plot.show()

# %%
# Visualisation of the virtual displacement sensor locations:
#
# .. image:: ../../../../_static/basics_ex5a_disp_locs.png
#    :alt: Visualisation of the virtual displacement sensor locations
#    :width: 800px
#    :align: center

for kk in sim_data_dict:
    (fig,ax) = sens.plot_exp_traces(
        exp_data,
        comp_ind=0,
        sens_key="temp",
        sim_key=kk,
        descriptor=sens.DescriptorFactory.temperature(),
    )

    save_fig: Path = output_path/f"ext_ex5a_traces_{kk}_temp.png"
    fig.savefig(save_fig,dpi=300,bbox_inches="tight")

# %%
# Simulated temperatures traces for input physics simulation 0:
#
# .. image:: ../../../../_static/ext_ex5a_traces_sim0_temp.png
#    :alt: Simulated temperature sensor traces for input simulation 0.
#    :width: 600px
#    :align: center
#
# Simulated temperature traces for input physics simulation 1:
#
# .. image:: ../../../../_static/ext_ex5a_traces_sim1_temp.png
#    :alt: Simulated temperature sensor traces for input simulation 1.
#    :width: 600px
#    :align: center


for key_sim in sim_data_dict:
    for ii,key_disp in enumerate(disp_keys):
        (fig,ax) = sens.plot_exp_traces(
            exp_data,
            comp_ind=ii,
            sens_key="disp",
            sim_key=key_sim,
            descriptor=sens.DescriptorFactory.displacement(),
        )

        save_fig: Path = (output_path
            /f"ext_ex5a_traces_{key_sim}_{key_disp}.png")
        fig.savefig(save_fig,dpi=300,bbox_inches="tight")

# %%
# Simulated displacement traces for input physics simulation 0:
#
# .. image:: ../../../../_static/ext_ex5a_traces_sim0_disp_y.png
#    :alt: Simulated displacement sensor traces form input simulation 0.
#    :width: 600px
#    :align: center
#
# Simulated displacement traces for input physics simulation 1:
#
# .. image:: ../../../../_static/ext_ex5a_traces_sim1_disp_y.png
#    :alt: Simulated displacement sensor traces form input simulation 1.
#    :width: 600px
#    :align: center

# Uncomment this to display the sensor trace plot
# plt.show()
