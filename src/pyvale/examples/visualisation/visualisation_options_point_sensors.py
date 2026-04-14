# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Visualisation:
================================================================================
TODO

"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

# Pyvale imports
import pyvale.sensorsim as sens
import pyvale.mooseherder as mh
import pyvale.dataset as dataset



#%%
# This is a basic set up of thermal data to be plot
# See examples/basics/ex1a_basicscalars_therm2d.py for detail regarding this

data_path = dataset.thermal_2d_path()
sim_data = mh.ExodusLoader(data_path).load_all_sim_data()

sim_data = sens.scale_length_units(scale=1000.0,
                                    sim_data=sim_data,
                                    disp_keys=None)

n_sens = (3,2,1)
x_lims = (0.0,100.0)
y_lims = (0.0,50.0)
z_lims = (0.0,0.0)
sens_pos = sens.gen_pos_grid_inside(n_sens,x_lims,y_lims,z_lims)


sens_data = sens.SensorData(positions=sens_pos)
field_key: str = "temperature"

sens_array = sens.SensorFactory.scalar_point(
    sim_data,
    sens_data,
    comp_key=field_key,
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.temperature(),
)

err_chain= [
    sens.ErrSysGen(sens.GenUniform(low=-5.0,high=5.0)),
    sens.ErrRandGen(sens.GenNormal(std=2.0)),
]

sens_array.set_error_chain(err_chain)


measurements = sens_array.sim_measurements()
print(f"\nMeasurements for last sensor:\n{measurements[-1,0,:]}\n")

#%%
# This is a basic set up of displacement data to be plot
# See examples/basics/ex2a_basicvectors_disp2d.py for detail regarding this

data_path2: Path = dataset.mechanical_2d_path()
sim_data2: mh.SimData = mh.ExodusLoader(data_path2).load_all_sim_data()

disp_keys = ("disp_x","disp_y")

sim_data2: mh.SimData  = sens.scale_length_units(scale=1000.0,
                                                sim_data=sim_data2,
                                                disp_keys=disp_keys)

#%%
# 2. Build virtual sensor arrays
# ------------------------------
# Creating a vector or tensor field sensor array is similar to what we
# have already done for scalar fields we just need to specify the string
# keys for the field components we want to use in the sim data object we have
# loaded. For vector and tensor field sensors we can also specify a sensor
# orientation which we demonstrate here.
#
# The information we provide in the `SensorData` object is treated as the ground
# truth so any 'field errors' we simulate later are calculated with respect to
# this.

sens_pos2: np.ndarray = sens.gen_pos_grid_inside(num_sensors=(2,2,1),
                                                x_lims=(0.0,100.0),
                                                y_lims=(0.0,150.0),
                                                z_lims=(0.0,0.0))

sample_times2: np.ndarray = np.linspace(0.0,np.max(sim_data2.time),50)

sens_angles2: tuple[Rotation] = sens_pos.shape[0] * \
    (Rotation.from_euler("zyx",[90,0,0], degrees=True),)

disp_sens_data = sens.SensorData(positions=sens_pos2,
                                 sample_times=sample_times2,
                                 angles=sens_angles2)

disp_sens: sens.SensorsPoint = sens.SensorFactory.vector_point(
    sim_data2,
    disp_sens_data,
    comp_keys=disp_keys,
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.displacement(),
)


pos_offset_xyz = np.array((2.0,2.0,0.0),dtype=np.float64)
pos_offset_xyz = np.tile(pos_offset_xyz,(sens_pos.shape[0],1))

pos_rand = sens.GenUniform(low=-2.0,high=2.0)  # units = mm

angle_offset2 = np.zeros_like(sens_pos2)
angle_offset2[:,0] = 1.0 # only rotate about z in 2D, units = degrees

angle_rand2 = sens.GenUniform(low=-5.0,high=5.0)

field_err_data2 = sens.ErrFieldData(pos_offset_xyz=pos_offset_xyz,
                                   pos_rand_xyz=(pos_rand,pos_rand,None),
                                   ang_offset_zyx=angle_offset2,
                                   ang_rand_zyx=(angle_rand2,None,None))


disp_err_chain: list[sens.IErrSimulator] = [
    sens.ErrRandGen(sens.GenNormal(std=2.0)),
    sens.ErrSysField(disp_sens.get_field(),field_err_data2),
]

disp_sens.set_error_chain(disp_err_chain)


#%%
# We can now visualise the sensor locations on the simulation mesh and the
# simulated sensor traces using pyvale's visualisation tools which use
# pyvista for meshes and matplotlib for sensor traces. pyvale will return
# plot and axes objects to the user allowing additional customisation using
# pyvista and matplotlib. This also means that we need to call `.show()`
# ourselves to display the figure as pyvale does not do this for us.
#
# If we are going to save figures we need to make sure the path exists. Here
# we create a default output path based on your current working directory.
output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)


save_render = output_path / "basics_ex1_1_sensorlocs.svg"

#%%
# This creates a pyvista visualisation of the sensor locations on the
# simulation mesh. The plot will can be shown in interactive mode by calling
# `pv_plot.show()`.

# plot a single sensor array

pv_plot = sens.plot_point_sensors_on_sim(sens_array, field_key)
pv_plot.show()

pv_plot = sens.plot_point_sensors_on_sim(disp_sens, "disp_x")
pv_plot.show()


# using two sensor arrays, each with a different component key
pv_plot = sens.plot_point_sensors_on_sim([sens_array, disp_sens], [field_key, "disp_x"])
pv_plot.show()


pv_plot = sens.plot_point_sensors_on_sim(disp_sens, "disp_x")
pv_plot.show()

#%%
# We determined manually by moving camera in interative mode and then
# printing camera position to console after window close, as below.
# pyvista applies changes to the last accessed subplot - (0,1) here
# to apply to all subplots, we must manually change subplot and apply
# changes individually
pv_plot.subplot(0,0)
pv_plot.camera_position = [(-7.547, 59.753, 134.52),
                            (41.916, 25.303, 9.297),
                            (0.0810, 0.969, -0.234)]


pv_plot.screenshot(save_render.with_suffix(".png"))

#%%
# We can also show the simulation and sensor locations in interative mode
# by calling `.show()`

print(80*"-")
print("Camera position after interactive view:")
print(pv_plot.camera_position)
print(80*"-"+"\n")


