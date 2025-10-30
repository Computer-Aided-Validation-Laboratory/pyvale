# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Scalar field sensor simulation
================================================================================

In this example we introduce the basic features of `pyvale` for scalar field 
sensor simulation. We demonstrate quick sensor array construction with defaults 
using the `pyvale` sensor factory.

We also introduce some key concepts for `pyvale` sensor simulation 

Finally we run a sensor simulation and display the output.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# pyvale imports
import pyvale.sensorsim as sens
import pyvale.mooseherder as mh
import pyvale.dataset as dataset


#%%
# Here we load a pre-generated MOOSE finite element simulation dataset that
# comes packaged with `pyvale` in exodus (*.e format). We use the `mooseherder` 
# module to load the exodus  file into a `SimData` object which contains the 
# simulation nodal coordinates, the simulation time stesp, the element 
# connectivity tables and nodal output field variables.
#
# We also convert the length units of our simulation from meters to milli-meters
# as our visualisation tools are based on unit scaling by default.

data_path: Path = dataset.thermal_3d_path()
sim_data: mh.SimData = mh.ExodusLoader(data_path).load_all_sim_data()

sim_data: mh.SimData = sens.scale_length_units(scale=1000.0,
                                               sim_data=sim_data,
                                               disp_comps=None)
#%%
# .. note::
#   You can load your own exodus (*.e) file here by changing the path or you can
#   load your own simulation data from delimited text files or numpy npy files. 
#   See the advanced example 'Bring your own simulation data'.      

#%%
# To create our simulated sensor array we need to specify the position of our 
# virtual sensors and the times that they should take synthetic measurements.
# We then use a helper function to create a regular grid of sensor locations by
# passing in the number of sensors we want in each direction and the spatial 
# bounds for the grid of sensors (based on `SimData.coords`). We could also have
# manually built the numpy array of sensor locations which has 
# shape=(num_sensors,coord[x,y,z]).
#
# We then collect these into the `SensorData` class which we will use to create 
# our virtual sensor array. You can also set `sample_times=None` here which will 
# make our virtual sensors sample at the simulation time steps. 

sens.simtools.print_dimensions(sim_data)

sens_pos: np.ndarray = sens.create_sensor_pos_array(num_sensors=(1,4,1),
                                                    x_lims=(12.5,12.5),
                                                    y_lims=(0.0,33.0),
                                                    z_lims=(0.0,12.0))

sample_times: np.ndarray = np.linspace(0.0,np.max(sim_data.time),50)

sens_data = sens.SensorData(positions=sens_pos,
                            sample_times=sample_times)

#%%
# We now use our simulation data and sensor data objects to create our virtual
# sensor array using a helper function from the sensor factory. We are also 
# going to need to specify which field we want our simulated sensors to
# sample so we create a `field_key` here which wel will reuse later. This should 
# match the dictionary key for the nodal variable of interest in our `SimData` 
# object. Also, as our simulation is 3D we have elements with 3 spatial 
# dimensions which we specify here.

field_key: str = "temperature"

sens_array: sens.SensorArrayPoint = sens.SensorFactory.scalar_no_errs(
    sim_data,
    sens_data,
    elem_dims=3,
    field_name=field_key,
    descriptor=None,
)

#%%
# Up to this point we have created a plain sensor array that does not have any
# simulated measurement errors, so let's add some. We do this by creating an
# `error_chain` which is a list of objects that implement the `IErrSimulator`
# interface. 
err_chain: list[sens.IErrCalculator] = []





#%%
# We have built our sensor array so now we can call `sim_measurements()` to
# generate simulated sensor traces.
measurements: np.ndarray = sens_array.sim_measurements()
#print(f"\nMeasurements for last sensor:\n{measurements[-1,0,:]}\n")


#%%
# We are going to save some figures produced by the `pyvale` visualisation tools 
# so we need to make sure the save path exists. Here we create a default output 
# directory based on your current working directory.
output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

#%%
# We can now visualise the sensor locations on the simulation mesh and the
# simulated sensor traces using `pyvale` visualisation tools which use
# `pyvista` for meshes and `matplotlib` for sensor traces. `pyvale` will return
# plot and axes objects to the user allowing additional customisation using
# `pyvista` and `matplotlib`. This also means that we need to call `.show()`
# ourselves to display the figure as pyvale does not do this for us.
#
# This creates a pyvista visualisation of the sensor locations on the
# simulation mesh. The plot will can be shown in interactive mode by calling
# `pv_plot.show()`.
pv_plot = sens.plot_point_sensors_on_sim(sens_array,field_key)

#%%
# We determined manually by moving camera in interative mode and then
# printing camera position to console after window close, as below.
pv_plot.camera_position = [(59.354, 43.428, 69.946),
                            (-2.858, 13.189, 4.523),
                            (-0.215, 0.948, -0.233)]

#%%
# This allows us to save a vector graphic and raster graphic showing the
# sensor locations on the simulation mesh
save_render = output_path / "sensorsim_basics_ex1_locs.svg"
pv_plot.save_graphic(save_render) # only for .svg .eps .ps .pdf .tex
pv_plot.screenshot(save_render.with_suffix(".png"))

#%%
# We can also show the simulation and sensor locations in interative mode
# by calling `.show()`
pv_plot.show()

# print(80*"-")
# print("Camera position after interactive view:")
# print(pv_plot.camera_position)
# print(80*"-"+"\n")

#%%
# This plots the time traces for all of our sensors. The solid line shows
# the 'truth' interpolated from the simulation and the dashed line with
# markers shows the simulated sensor traces. In later examples we will see
# how to configure this plot but for now we note we that we are returned a
# matplotlib figure and axes object which allows for further customisation.
(fig,ax) = sens.plot_time_traces(sens_array,field_key)

#%%
# We can also save the sensor trace plot as a vector and raster graphic
save_traces = output_path/"sensorsim_basics_ex1_traces.png"
fig.savefig(save_traces, dpi=300, bbox_inches="tight")
fig.savefig(save_traces.with_suffix(".svg"), dpi=300, bbox_inches="tight")

#%%
# The trace plot can also be shown in interactive mode using `plt.show()`
plt.show()

#%%
# That is it for this example. In the next one we will look at simulating 
# sensors for vector and tensor fields (e.g. displacament and strain fields).
