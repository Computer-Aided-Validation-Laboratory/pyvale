# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Basics: Pyvale point sensor simulation
================================================================================

In this example we introduce the basic features of `pyvale` for point sensor
simulation. We demonstrate quick sensor array construction with defaults using
the `pyvale` sensor array factory. Finally we run a sensor simulation and display
the output.

Test case: Scalar field point sensors (thermocouples) on a 2D thermal simulation
"""

from pathlib import Path
import matplotlib.pyplot as plt

# Pyvale imports
import pyvale.sensorsim as sens
import pyvale.mooseherder as mh
import pyvale.dataset as dataset



#%%
# This is a basic set up of thermal data to be plot
# See examples/basics/ex1a_basicscalars_therm2d.py for detail regarding this

data_path = dataset.thermal_2d_path()
sim_data = mh.ExodusReader(data_path).read_all_sim_data()

sim_data = sens.scale_length_units(scale=1000.0,
                                    sim_data=sim_data,
                                    disp_comps=None)

n_sens = (3,2,1)
x_lims = (0.0,100.0)
y_lims = (0.0,50.0)
z_lims = (0.0,0.0)
sens_pos = sens.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)


sens_data = sens.SensorData(positions=sens_pos)
field_key: str = "temperature"
tc_array = sens.SensorArrayFactory \
    .thermocouples_basic_errs(sim_data,
                                sens_data,
                                elem_dims=2,
                                field_name=field_key)


measurements = tc_array.calc_measurements()
print(f"\nMeasurements for last sensor:\n{measurements[-1,0,:]}\n")

#%%
# This is a basic set up of displacement data to be plot
# See examples/basics/ex2a_basicvectors_disp2d.py for detail regarding this

data_path2 = dataset.mechanical_2d_path()
sim_data2 = mh.ExodusReader(data_path2).read_all_sim_data()

field_name2 = "disp"
field_comps2 = ("disp_x","disp_y")
sim_data2 = sens.scale_length_units(scale=1000.0,
                                    sim_data=sim_data2,
                                    disp_comps=field_comps2)

n_sens2 = (2,3,1)
x_lims2 = (0.0,100.0)
y_lims2 = (0.0,150.0)
z_lims2 = (0.0,0.0)
sens_pos2 = sens.create_sensor_pos_array(n_sens2,x_lims2,y_lims2,z_lims2)

sens_data2 = sens.SensorData(positions=sens_pos2)

disp_sens_array2 = sens.SensorArrayFactory \
                    .disp_sensors_basic_errs(sim_data2,
                                                sens_data2,
                                                elem_dims=2,
                                                field_name=field_name2,
                                                field_comps=field_comps2,
                                                errs_pc=2.0)

measurements2 = disp_sens_array2.calc_measurements()

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
pv_plot = sens.plot_point_sensors_on_sim(tc_array, field_key)
pv_plot.show()

# using two sensor arrays, each with a different component key
pv_plot = sens.plot_point_sensors_on_sim([tc_array, disp_sens_array2], [field_key, "disp_x"])
pv_plot.show()

# create three subplots where each sensor array plotted uses the same component key
pv_plot = sens.plot_point_sensors_on_sim([tc_array, tc_array, tc_array], field_key)

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

# pv_plot.subplot(0,1)
# pv_plot.camera_position = [(-7.547, 59.753, 134.52),
#                             (41.916, 25.303, 9.297),
#                             (0.0810, 0.969, -0.234)]

pv_plot.save_graphic(save_render) # only for .svg .eps .ps .pdf .tex
pv_plot.screenshot(save_render.with_suffix(".png"))

#%%
# We can also show the simulation and sensor locations in interative mode
# by calling `.show()`
pv_plot.show()

print(80*"-")
print("Camera position after interactive view:")
print(pv_plot.camera_position)
print(80*"-"+"\n")

#%%
# This plots the time traces for all of our sensors. The solid line shows
# the 'truth' interpolated from the simulation and the dashed line with
# markers shows the simulated sensor traces. In later examples we will see
# how to configure this plot but for now we note we that we are returned a
# matplotlib figure and axes object which allows for further customisation.

(fig,ax) = sens.plot_time_traces(tc_array,field_key)

traceopts = sens.TraceOptsSensor()
traceopts.sensors_per_plot = 2

#%%
# We can also save the sensor trace plot as a vector and raster graphic
save_traces = output_path/"basics_ex1_1_sensortraces.png"
#fig.savefig(save_traces, dpi=300, bbox_inches="tight")
#fig.savefig(save_traces.with_suffix(".svg"), dpi=300, bbox_inches="tight")

#%%
# The trace plot can also be shown in interactive mode using `plt.show()`
#plt.show()

# Plot with limit of two traces per subplot
traceopts = sens.TraceOptsSensor()
traceopts.sensors_per_plot = 2
traceopts.sensors_to_plot = [1,3,5, "fake"]

#(fig, ax) = sens.plot_time_traces(tc_array, field_key, trace_opts=traceopts)
#plt.show()


#sens.animate_trace_with_sensors(tc_array,field_key)
