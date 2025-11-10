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
# coefficient and the remaining simulations represent a +/-10% perturbation to
# these simulation inputs.

sim_paths: list[Path] = dataset.thermomechanical_3d_experiment_paths()

sim_data_list: list[mh.SimData] = []
for ss in sim_paths:
    sim_data: mh.SimData = mh.ExodusLoader(ss).load_all_sim_data()

    disp_keys = ("disp_x","disp_y","disp_z")
    sim_data: mh.SimData = sens.scale_length_units(scale=1000.0,
                                                   sim_data=sim_data,
                                                   disp_keys=disp_keys)
    sim_data_list.append(sim_data)


#%%
# 2. Build virtual sensor arrays
# ------------------------------

#%%
# 2.1 Build scalar field sensor array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we build a scalar field sensor array to simulate thermocouples applied to
# our simulation.

sample_times = np.linspace(0.0,np.max(sim_data.time),50)

temp_sens_pos: np.ndarray = sens.gen_pos_grid_inside(num_sensors=(1,4,1),
                                                     x_lims=(12.5,12.5),
                                                     y_lims=(0.0,33.0),
                                                     z_lims=(0.0,12.0))

temp_sens_data = sens.SensorData(positions=temp_sens_pos,
                                 sample_times=sample_times)

temp_sens: sens.SensorArrayPoint = sens.SensorFactory.scalar_no_errs(
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
# particular axes. However, here we could have just omitted the position random 
# generator in the X direction and replace it with `None` 

temp_err_chain: list[sens.IErrSimulator] = []
temp_err_chain.append(sens.ErrRandNorm(std=2.0)) #

temp_pos_uncert = 0.1 # units = mm
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
# our simulation. 

strain_sens_pos: np.ndarray = sens.gen_pos_grid_inside(num_sensors=(1,4,1),
                                                       x_lims=(9.4,9.4),
                                                       y_lims=(0.0,33.0),
                                                       z_lims=(12.0,12.0))

strain_sens_data = sens.SensorData(positions=strain_sens_pos,
                                   sample_times=sample_times)

strain_sens: sens.SensorArrayPoint = sens.SensorFactory.tensor_no_errs(
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
# ------------------------------------
# TODO


sensor_arrays: list[sens.ISensorArray] = [temp_sens,strain_sens]

exp_sim = sens.ExperimentSimulator(sim_data_list,
                                   sensor_arrays)

exp_data = exp_sim.run_experiments(num_exp_per_sim=100)
exp_stats = exp_sim.calc_stats()

#%%
# 4. Analyse & visualise the results 
# ----------------------------------
# TODO

#%%
# We print the lengths of our exp_data and exp_stats lists along with the
# shape of the numpy arrays they contain so we can index into them easily.
print(80*"=")
print("exp_data and exp_stats are lists where the index is the sensor array")
print("position in the list as field components are not consistent dims:\n")
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

#%%
# TODO

#%%
# We visualise our thermcouple locations on our mesh to make sure they are
# in the correct positions.
cam_pos = np.array([(59.354, 43.428, 69.946),
                    (-2.858, 13.189, 4.523),
                    (-0.215, 0.948, -0.233)])


pv_plot = sens.plot_point_sensors_on_sim(temp_sens,"temperature")
pv_plot.camera_position = cam_pos
pv_plot.show()


#%%
# Now we visualise the strain gauge locations to make sure they are where
# we expect them to be.
pv_plot = sens.plot_point_sensors_on_sim(strain_sens,"strain_yy")
pv_plot.camera_position = cam_pos
pv_plot.show()
