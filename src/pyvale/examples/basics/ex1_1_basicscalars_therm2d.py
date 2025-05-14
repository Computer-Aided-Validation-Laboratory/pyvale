# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: Basics of pyvale point sensor simualtion
--------------------------------------------------------------------------------
In this example we introduce the basic features of pyvale for point sensor
simulation. We demonstrate quick sensor array construction with defaults using
the pyvale sensor array factory. Finally we run a sensor simulation and display
the output.

Test case: Scalar field point sensors (thermocouples) on a 2D thermal simulation
"""

from pathlib import Path
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv


def main() -> None:

    # Here we load a pre-generated MOOSE finite element simulation dataset that
    # comes packaged with pyvale. The simulation is a 2D rectangular plate with
    # a bi-directional temperature gradient. You can replace this with the path
    # to your own MOOSE simulation with exodus output (*.e). Note that the
    # field_key must match the name of your variable in your MOOSE simulation.
    data_path = pyv.DataSet.thermal_2d_path()
    # We use `mooseherder` to load the exodus file into a `SimData` object.
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    # Scale to mm to make 3D visualisation scaling easier as pyvista scales
    # everything to unity
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=None)

    # We now use a helper function to create a grid of sensor locations but we
    # could have also manually built the numpy array of sensor locations which
    # has the shape=(num_sensors,coord[x,y,z]).
    n_sens = (3,2,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    # This dataclass contains the parameters to build our sensor array. We can
    # also customise the output frequency, the sensor area and the sensor
    # orientation. For now we will use the defaults which assumes an ideal point
    # sensor sampling at the simulation time steps.
    sens_data = pyv.SensorData(positions=sens_pos)

    # Now that we have our sensor locations we can use the sensor factory to
    # build a basic thermocouple array with some useful defaults. In later
    # examples we will see how to customise sensor parameters and errors.
    # This basic thermocouple array includes a 1% systematic and random error.
    # If you want to remove the simulated errors and just interpolate at the
    # sensor locations then user `.thermocouples_no_errs()`.
    field_key: str = "temperature"
    tc_array = pyv.SensorArrayFactory \
        .thermocouples_basic_errs(sim_data,
                                  sens_data,
                                  elem_dims=2,
                                  field_name=field_key)

    # We have built our sensor array so now we can call `calc_measurements()` to
    # generate simulated sensor traces.
    measurements = tc_array.calc_measurements()
    print(f"\nMeasurements for last sensor:\n{measurements[-1,0,:]}\n")

    # We can now visualise the sensor locations on the simulation mesh and the
    # simulated sensor traces using pyvale's visualisation tools which use
    # pyvista for meshes and matplotlib for sensor traces. pyvale will return
    # plot and axes objects to the user allowing additional customisation using
    # pyvista and matplotlib. This also means that we need to call `.show()`
    # ourselves to display the figure as pyvale does not do this for us.

    # If we are going to save figures we need to make sure the path exists. Here
    # we create a default output path based on your current working directory.
    output_path = Path.cwd() / "pyvale-output"
    if not output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

    # This creates a pyvista visualisation of the sensor locations on the
    # simulation mesh. The plot will can be shown in interactive mode by calling
    # `pv_plot.show()` alternatively
    pv_plot = pyv.plot_point_sensors_on_sim(tc_array,field_key)

    # We determined manually by moving camera in interative mode and then
    # printing camera position to console after window close, as below.
    pv_plot.camera_position = [(-7.547, 59.753, 134.52),
                                (41.916, 25.303, 9.297),
                                (0.0810, 0.969, -0.234)]


    # This allows us to save a vector graphic and raster graphic showing the
    # sensor locations on the simulation mesh
    save_render = output_path / "basics_ex1_1_sensorlocs.svg"
    pv_plot.save_graphic(save_render) # only for .svg .eps .ps .pdf .tex
    pv_plot.screenshot(save_render.with_suffix(".png"))

    # We can also show the simulation and sensor locations in interative mode
    # by calling `.show()`
    pv_plot.show()

    print(80*"-")
    print("Camera position after interative view:")
    print(pv_plot.camera_position)
    print(80*"-"+"\n")

    # This plots the time traces for all of our sensors. The solid line shows
    # the 'truth' interpolated from the simulation and the dashed line with
    # markers shows the simulated sensor traces. In later examples we will see
    # how to configure this plot but for now we note we that we are returned a
    # matplotlib figure and axes object which allows for further customisation.
    (fig,ax) = pyv.plot_time_traces(tc_array,field_key)

    # We can also save the sensor trace plot as a vector and raster graphic
    save_traces = output_path/"basics_ex1_1_sensortraces.png"
    fig.savefig(save_traces, dpi=300, bbox_inches="tight")
    fig.savefig(save_traces.with_suffix(".svg"), dpi=300, bbox_inches="tight")

    # The trace plot can also be shown in interactive mode using `plt.show()`
    plt.show()


if __name__ == "__main__":
    main()
