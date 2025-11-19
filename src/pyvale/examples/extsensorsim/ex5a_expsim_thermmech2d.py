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

disp_keys = ("disp_x","disp_y")

sim_data_list: list[mh.SimData] = []
for ss in sim_paths:
    sim_data = mh.ExodusLoader(ss).load_all_sim_data()
    sim_data = sens.scale_length_units(scale=1000.0,
                                       sim_data=sim_data,
                                       disp_keys=disp_keys)
    sim_data_list.append(sim_data)

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

temp_sens: sens.SensorArrayPoint = sens.SensorFactory.scalar_point(
    sim_data,
    temp_sens_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.temperature(),
)


#%%
# 2.2 Add errors to the scalar field sensors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
temp_err_chain: list[sens.IErrSimulator] = []


temp_pos_uncert = 1.0 # units = mm
temp_pos_rand = (sens.GenNormal(std=temp_pos_uncert),
                 sens.GenNormal(std=temp_pos_uncert),
                 None)

temp_field_err_data = sens.ErrFieldData(pos_rand_xyz=temp_pos_rand)
temp_err_chain.append(sens.ErrSysField(temp_sens.get_field(),
                                       temp_field_err_data))

temp_err_chain.append(
    sens.ErrRandNormPercent(std_percent=2.0, 
                            err_dep=sens.EErrDep.DEPENDENT)
)

temp_err_chain.append(sens.ErrSysOffsetPercent(offset_percent=-1.0))
temp_err_chain.append(sens.ErrSysDigitisation(bits_per_unit=2**24/100))
temp_err_chain.append(sens.ErrSysSaturation(meas_min=0.0,meas_max=700.0))

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

disp_sens: sens.SensorArrayPoint = sens.SensorFactory.vector_point(
    sim_data,
    disp_sens_data,
    comp_keys=disp_keys,
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.displacement(),
)

#%%
# 2.4 Add errors to the vector field sensors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

disp_err_chain: list[sens.IErrSimulator] = []

rand_gen = sens.GenNormal(std=1.0) # units = % truth
disp_err_chain.append(sens.ErrRandGenPercent(rand_gen))

pos_rand = sens.GenUniform(low=-1.0,high=1.0)   # units = mm
angle_rand = sens.GenUniform(low=-2.0,high=2.0) # units = degrees

field_err_data = sens.ErrFieldData(pos_rand_xyz=(pos_rand,pos_rand,None),
                                   ang_rand_zyx=(angle_rand,None,None))

disp_err_chain.append(sens.ErrSysField(disp_sens.get_field(),
                                       field_err_data))

disp_err_chain.append(sens.ErrSysOffsetPercent(offset_percent=1.0))
disp_err_chain.append(sens.ErrSysDigitisation(bits_per_unit=2**24/1.0))
disp_err_chain.append(sens.ErrSysSaturation(meas_min=-1.0,meas_max=1.0))

disp_sens.set_error_chain(disp_err_chain)

#%%
# 3. Create & run simulated experiments
# -------------------------------------

sensor_arrays: list[sens.ISensorArray] = [temp_sens,disp_sens]

exp_sim = sens.ExperimentSimulator(sim_data_list,sensor_arrays)

exp_data: list[np.ndarray] = exp_sim.run_experiments(num_exp_per_sim=1000)
exp_stats: list[sens.ExperimentStats] = sens.calc_experiment_stats(exp_data)


#%%
# 4. Analyse & visualise the results 
# ----------------------------------

print(80*"=")
print("exp_data and exp_stats are lists where the index is the sensor array")
print("position in the list as field components are not consistent dims:")
print(f"{len(exp_data)=}")
print(f"{len(exp_stats)=}")
print()
print(80*"-")
print("Thermal sensor array @ exp_data[0]")
print(80*"-")
print("shape=(n_sims,n_exps,n_sensors,n_field_comps,n_time_steps)")
print(f"{exp_data[0].shape=}")
print()
print("Stats are calculated over all experiments (axis=1)")
print("shape=(n_sims,n_sensors,n_field_comps,n_time_steps)")
print(f"{exp_stats[0].max.shape=}")
print()
print(80*"-")
print("Mechanical sensor array @ exp_data[1]")
print(80*"-")
print("shape=(n_sims,n_exps,n_sensors,n_field_comps,n_time_steps)")
print(f"{exp_data[1].shape=}")
print()
print("shape=(n_sims,n_sensors,n_field_comps,n_time_steps)")
print(f"{exp_stats[1].max.shape=}")
print(80*"=")

# %%
# .. image:: ../../../../_static/ext_ex5a_term_out.png
#    :alt: Terminal output showing the simulated measurements
#    :width: 700px
#    :align: center


output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True) 

pv_plot = sens.plot_point_sensors_on_sim(temp_sens,"temperature")
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

pv_plot = sens.plot_point_sensors_on_sim(disp_sens,"disp_x")
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

for ii,_ in enumerate(sim_data_list):
    (fig,ax) = sens.plot_exp_traces(exp_sim,
                                    component="temperature",
                                    sens_array_num=0,
                                    sim_num=ii)

    save_fig: Path = output_path/f"ext_ex5a_traces_sim{ii}_temp.png" 
    fig.savefig(save_fig,dpi=300,bbox_inches="tight")

# Uncomment this to display the sensor trace plot 
# plt.show()

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


for ii,_ in enumerate(sim_data_list):
    for kk in disp_keys:
        (fig,ax) = sens.plot_exp_traces(exp_sim,
                                        component=kk,
                                        sens_array_num=1,
                                        sim_num=ii)
                                        
        save_fig: Path = output_path/f"ext_ex5a_traces_sim{ii}_{kk}.png"
        fig.savefig(save_fig,dpi=300,bbox_inches="tight")

# %%
# Simulated displacement traces for input physics simulation 0:
# 
# .. image:: ../../../../_static/ext_ex5a_traces_disp_y.png
#    :alt: Simulated displacement sensor traces form input simulation 0.
#    :width: 600px
#    :align: center
#
# Simulated displacement traces for input physics simulation 1:
# 
# .. image:: ../../../../_static/ext_ex5a_traces_disp_y.png
#    :alt: Simulated displacement sensor traces form input simulation 1.
#    :width: 600px
#    :align: center
