# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Multi-physics experiment simulation
================================================================================

In this example we apply multiple sensor arrays across a number of different 
physics simulations with different inputs allowing us to run a series of virtual
experiments and analyse the results.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# pyvale imports
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.dataset as dataset

#%%
# 1. Load physics simulation data
# -------------------------------
# First we load a set of simulations all of the same thermo-mechanical test case
# where one simulation uses the reference thermal conductivity and expansion 
# coefficient and the remaining simulation represents a -10% perturbation to
# these simulation inputs.

sim_paths: list[Path] = dataset.thermomechanical_3d_experiment_paths()
sim_keys: set[str] = {"sim_nominal","sim_perturbed"}

disp_keys = ("disp_x","disp_y","disp_z")

sim_data_dict: dict[str,mh.SimData] = {}
for ss,kk in zip(sim_paths,sim_keys):
    sim_data: mh.SimData = mh.ExodusLoader(ss).load_all_sim_data()
    
    sim_data: mh.SimData = sens.scale_length_units(scale=1000.0,
                                                   sim_data=sim_data,
                                                   disp_keys=disp_keys)
    sim_data_dict[kk] = sim_data


#%%
# 2. Build virtual sensor arrays
# ------------------------------

#%%
# 2.1 Build scalar field sensor array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we build a scalar field sensor array to simulate thermocouples applied to
# our simulation in a similar way to what we have seen in previous examples.

sample_times = np.linspace(0.0,np.max(sim_data.time),50)

temp_sens_pos: np.ndarray = sens.gen_pos_grid_inside(num_sensors=(1,4,1),
                                                     x_lims=(12.5,12.5),
                                                     y_lims=(0.0,33.0),
                                                     z_lims=(0.0,12.0))

temp_sens_data = sens.SensorData(positions=temp_sens_pos,
                                 sample_times=sample_times)

temp_sens: sens.SensorArrayPoint = sens.SensorFactory.scalar_point(
    sim_data,
    temp_sens_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.THREED,
    descriptor=sens.DescriptorFactory.temperature(),
)

#%%
# 2.2 Add errors to the scalar field sensors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now we add some errors to our thermocouple sensor array starting with a random
# noise with a standard deviation of 2 degrees. We then build a field error 
# that includes sensor position uncertainty. Here we generate position 
# uncertainty in all dimensions but then we use the `pos_lock_xyz` numpy array
# in our `ErrFieldData` object to constrain all sensors to not move off the 
# surface they are on. This feature is particularly useful when we have sensors 
# on different faces of a 3D simulation and we want to constrain the sensors in 
# particular axes for particular sensors. However, here we could have just 
# omitted the position random  generator in the X direction and replace it with 
# `None`. 

temp_err_chain: list[sens.IErrSimulator] = []
temp_err_chain.append(sens.ErrRandNorm(std=2.0)) # units = degrees

temp_pos_uncert = 0.5 # units = mm
temp_pos_rand = (sens.GenNormal(std=temp_pos_uncert),
                 sens.GenNormal(std=temp_pos_uncert),
                 sens.GenNormal(std=temp_pos_uncert))

temp_pos_lock = np.full(temp_sens_pos.shape,False,dtype=bool)
temp_pos_lock[:,0] = True

temp_field_err_data = sens.ErrFieldData(pos_rand_xyz=temp_pos_rand,
                                        pos_lock_xyz=temp_pos_lock)
temp_err_chain.append(sens.ErrSysField(temp_sens.get_field(),
                                       temp_field_err_data))

temp_sens.set_error_chain(temp_err_chain)

#%%
# 2.3 Build tensor field sensor array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we build a tensor field sensor array to simulate strain gauges applied to
# our simulation in a similar way to what we have seen in previous examples.

strain_sens_pos: np.ndarray = sens.gen_pos_grid_inside(num_sensors=(1,4,1),
                                                       x_lims=(9.4,9.4),
                                                       y_lims=(0.0,33.0),
                                                       z_lims=(12.0,12.0))

strain_sens_data = sens.SensorData(positions=strain_sens_pos,
                                   sample_times=sample_times)

strain_sens: sens.SensorArrayPoint = sens.SensorFactory.tensor_point(
    sim_data,
    strain_sens_data,
    norm_comp_keys=("strain_xx","strain_yy","strain_zz"),
    dev_comp_keys=("strain_xy","strain_yz","strain_xz"),
    spatial_dims=sens.EDim.THREED,
    descriptor=sens.DescriptorFactory.strain(sens.EDim.THREED),
)

#%%
# 2.4 Add errors to the tensor field sensors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now we add some errors to our strain gauge array starting with random noise 
# with a standard deviation of 2% of the ground truth. We then build a field 
# error to simulate orientation uncertainty and demonstrate the same 'lock'
# functionality that allows us to constrain the sensors to only rotate on the 
# surface they are on. 

strain_err_chain: list[sens.IErrSimulator] = []
strain_err_chain.append(sens.ErrRandNormPercent(std_percent=2.0))

angle_uncert: float = 2.0 # units = degrees
angle_rand_zyx = (sens.GenUniform(low=-angle_uncert,high=angle_uncert),
                  sens.GenUniform(low=-angle_uncert,high=angle_uncert),
                  sens.GenUniform(low=-angle_uncert,high=angle_uncert))

angle_lock = np.full(strain_sens_pos.shape,True,dtype=bool)
angle_lock[:,0] = False   # Allow rotation about z

strain_field_err_data = sens.ErrFieldData(ang_rand_zyx=angle_rand_zyx,
                                          ang_lock_zyx=angle_lock)
strain_err_chain.append(sens.ErrSysField(strain_sens.get_field(),
                                         strain_field_err_data))

strain_sens.set_error_chain(strain_err_chain)

#%%
# 3. Create & run simulated experiments
# -------------------------------------
# We can now run our experiments over all simulations for all our virtual  
# sensor arrays. We are returned a dicitionary of numpy arrays. The key in the 
# dicitionary corresponds to the key we used in our input sensor arrays 
# dictionary in the list . So if we want our thermocouple results we want 
# exp_data["temp"] and for our strain gauges exp_data["strain"]. The numpy array 
# with our simulated experimental data has the following shape:
# (n_sims,n_exps,n_sensors,n_field_comps,n_time_steps)
#
# We can also calculate summary statistics for each sensor array which is
# returned as a dictionary where the key again corresponds to the key we used
# in the input sensor arrays dictionary. The experiment stats object contains 
# numpy arrays for each statistic that is collapsed over the number of
# experiments. The statistics we can access include: mean, standard deviation
# minimum, maximum, median, median absolute deviation and the 25% and 75%
# quartiles. See the `ExperimentStats` data class for details.
#
# There is also a reserved key in the... 
# TODO!

sensor_array_dict: dict[str,sens.ISensorArray] = {
    "temp": temp_sens,
    "strain": strain_sens,
}

exp_sim = sens.ExperimentSimulator(sim_data_dict,
                                   sensor_array_dict)

exp_data: dict[str,np.ndarray] = exp_sim.run_experiments(num_exp_per_sim=100)
exp_stats: dict[str,sens.ExperimentStats] = sens.calc_experiment_stats(exp_data)

#%%
# 4. Analyse & visualise the results 
# ----------------------------------
# Here we print the lengths of our exp_data and exp_stats lists along with the
# shape of the numpy arrays they contain to illustrate the format of the data
# structures output from our simulated experiment. This should give you all the
# information you need to slice out the data you want for additional analysis.

print(80*"=")
print("Simulation keys corresponding to the first axis:")
print(f"{exp_data['sim_keys']=}")
print()
print(80*"-")
print("Thermal sensor array @ exp_data['temp']")
print(80*"-")
print("shape=(n_sims,n_exps,n_sensors,n_field_comps,n_time_steps)")
print(f"{exp_data['temp'].shape=}")
print()
print("Stats are calculated over all experiments (axis=1)")
print("shape=(n_sims,n_sensors,n_field_comps,n_time_steps)")
print(f"{exp_stats['temp'].max.shape=}")
print()
print(80*"-")
print("Mechanical sensor array @ exp_data['strain']")
print(80*"-")
print("shape=(n_sims,n_exps,n_sensors,n_field_comps,n_time_steps)")
print(f"{exp_data['strain'].shape=}")
print()
print("shape=(n_sims,n_sensors,n_field_comps,n_time_steps)")
print(f"{exp_stats['strain'].max.shape=}")
print(80*"=")

# %%
# Example terminal output:
#
# .. image:: ../../../../_static/basics_ex3_term_out.png
#    :alt: Terminal output showing the simulated experiment data structures
#    :width: 700px
#    :align: center

#%%
# Now we are going to save visualisations of our temperature and strain sensor
# locations on the simulation mesh so we can verify we have setup their 
# locations as we expected.

output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True) 

cam_pos = np.array([(59.354, 43.428, 69.946),
                    (-2.858, 13.189, 4.523),
                    (-0.215, 0.948, -0.233)])

pv_plot = sens.plot_point_sensors_on_sim(temp_sens,"temperature")
pv_plot.camera_position = cam_pos

# Set to False to show an interactive plot instead of saving the figure
pv_plot.off_screen = True
if pv_plot.off_screen: 
    pv_plot.screenshot(output_path/"basics_ex3_temp_locs.png")
else:
    pv_plot.show()

# %%
# Visualisation of the virtual temperature sensor locations:
#
# .. image:: ../../../../_static/basics_ex3_temp_locs.png
#    :alt: Visualisation of the virtual temperature sensor locations
#    :width: 800px
#    :align: center

pv_plot = sens.plot_point_sensors_on_sim(strain_sens,"strain_yy")
pv_plot.camera_position = cam_pos

# Set to False to show an interactive plot instead of saving the figure
pv_plot.off_screen = True
if pv_plot.off_screen: 
    pv_plot.screenshot(output_path/"basics_ex3_strain_locs.png")
else:
    pv_plot.show()

# Uncomment to show interactive figure and set off_screen = False above
# pv_plot.show()

# %%
# Visualisation of the virtual strain sensor locations:
#
# .. image:: ../../../../_static/basics_ex3_strain_locs.png
#    :alt: Visualisation of the virtual strain sensor locations
#    :width: 800px
#    :align: center

#%%
# Finally, we plot the traces for all simulated experiments for our virtual 
# temperature and strain sensors.

trace_opts = sens.TraceOptsExperiment(plot_all_exp_points=True)

(fig,ax) = sens.plot_exp_traces(exp_sim,
                                comp_key="temperature",
                                sens_array_key="temp",
                                sim_key="sim_nominal",
                                trace_opts=trace_opts)

save_fig: Path = output_path/"basics_ex3_traces_temp.png" 
fig.savefig(save_fig,dpi=300,bbox_inches="tight")

# %%
# Virtual temperature sensor traces over all simulated experiments for input
# simulation 0:
#
# .. image:: ../../../../_static/basics_ex3_traces_temp.png
#    :alt: Simulated temperature sensor traces.
#    :width: 500px
#    :align: center

strain_plot_keys = ("strain_xx","strain_yy","strain_xy")
for kk in strain_plot_keys:
    (fig,ax) = sens.plot_exp_traces(exp_sim,
                                    comp_key=kk,
                                    sens_array_key="strain",
                                    sim_key="sim_nominal",
                                    trace_opts=trace_opts)
                                    
    save_fig: Path = output_path/f"basics_ex3_traces_{kk}.png"
    fig.savefig(save_fig,dpi=300,bbox_inches="tight")

# %%
# Virtual strain sensor traces over all simulated experiments for input
# simulation 0:
#
# .. image:: ../../../../_static/basics_ex3_traces_strain_yy.png
#    :alt: Simulated strain sensor traces.
#    :width: 500px
#    :align: center
    
# Uncomment this to display the sensor trace plots 
# plt.show()
