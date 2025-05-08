# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: Building a point sensor array from scratch with custom errors
--------------------------------------------------------------------------------
Here we build a custom point sensor array from scratch that is similar to the
pre-built thermocouple array from example 1.1. For this example we switch to a
3D thermal simulation of a fusion heatsink component.

Test case: Scalar field point sensors (thermocouples) on a 3D thermal simulation
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv


def main() -> None:
    # To build our custom point sensor array we need to at minimum provide a
    # `IField` (i.e. `FieldScaler`, `FieldVector`, `FieldTensor`) and a
    # `SensorData` object. For labelling visualisations (e.g. axis labels and
    # unit labels) we can also provide a `SensorDescriptor` object.
    # Once we have built our `SensorArrayPoint` object from these we can then
    # attach custom chains of different types of random and systematic errors
    # to be evaluated when we run our measurement simulation. This example is
    # based on the same thermal example we have used in the last two examples so
    # we start by loading our simulation data:

    data_path = pyv.DataSet.thermal_3d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier as pyvista scales
    # everything to unity
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=None)

    # We are going to build a custom temperature sensor so we need a scalar
    # field object to perform interpolation to the sensor locations at the
    # desired sampling times.
    field_key: str = "temperature"
    t_field = pyv.FieldScalar(sim_data,
                              field_key=field_key,
                              elem_dims=3)


    # Next we need to create our `SensorData` object which will set the position
    # and sampling times of our sensors. We use the same helper function we used
    # previously to create a uniformly spaced grid of sensors in space
    n_sens = (1,4,1)
    x_lims = (12.5,12.5)
    y_lims = (0.0,33.0)
    z_lims = (0.0,12.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    # We are also going to specify the times at which we would like to simulate
    # measurements. Setting this to `None` will default the measurements times
    # to match the simulation time steps.
    sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sensor_data = pyv.SensorData(positions=sens_pos,
                                 sample_times=sample_times)

    # Finally, we can create a `SensorDescriptor` which will be used to label
    # the visualisation and sensor trace plots we have seen in previous
    # examples.
    use_auto_descriptor: str = "blank"
    if use_auto_descriptor == "manual":
        descriptor = pyv.SensorDescriptor(name="Temperature",
                                          symbol="T",
                                          units = r"^{\circ}C",
                                          tag = "TC")
    elif use_auto_descriptor == "factory":
        descriptor = pyv.SensorDescriptorFactory.temperature_descriptor()
    else:
        descriptor = pyv.SensorDescriptor()


    # We can now build our custom point sensor array. This sensor array has no
    # errors so if we call `get_measurements()` or `calc_measurements()` we will
    # be able to extract the simulation truth values at the sensor locations.
    tc_array = pyv.SensorArrayPoint(sensor_data,
                                    t_field,
                                    descriptor)

    # This is a new 3D simulation we are analysing so we should visualise the
    # sensor locations before we run our measurement simulation. We use the same
    # code as we did in example 1.1 to display the sensor locations.

    # We are also going to save some figures to disk as well as displaying them
    # interactively so we create a directory for this:
    output_path = Path.cwd() / "pyvale-output"
    if not output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

    pv_plot = pyv.plot_point_sensors_on_sim(tc_array,field_key)

    pv_plot.camera_position = [(59.354, 43.428, 69.946),
                               (-2.858, 13.189, 4.523),
                               (-0.215, 0.948, -0.233)]

    save_render = output_path / "customsensors_ex1_3_sensorlocs.svg"
    pv_plot.save_graphic(save_render) # only for .svg .eps .ps .pdf .tex
    pv_plot.screenshot(save_render.with_suffix(".png"))

    pv_plot.show()

    # If we want to simulate sources of uncertainty for our sensor array we need
    # to add an `ErrIntegrator` to our sensor array using the method
    # `set_error_integrator()`. We provide our `ErrIntegrator` a list of error
    # objects which will be evaluated in the order specified in the list.
    #
    # In pyvale errors have a type specified as: random / systematic
    # (`EErrorType`) and a dependence `EErrDependence` as: independent /
    # dependent. When analysing errors all random all systematic errors are
    # grouped and summed together.
    #
    # The error dependence determines if an error is
    # calculated based on the truth (independent) or the accumulated measurement
    # based on all previous errors in the chain (dependent). Some errors are
    # purely independent such as random noise with a normal distribution with a
    # set standard devitation. An example of an error that is dependent would be
    # saturation which must be place last in the error chain and will clamp the
    # final sensor value to be within the specified bounds.
    #
    # pyvale provides a library of different random `ErrRand*` and systematic
    # `ErrSys*` errors which can be found listed in the docs. In the next
    # example we will explore the error library but for now we will specify some
    # common error types. Try experimenting with the code below to turn the
    # different error types off and on to see how it changes the virtual sensor
    # measurements.
    errors_on = {"sys": True,
                 "rand": True}

    error_chain = []
    if errors_on["sys"]:
        # This systematic error is just a constant offset of -5 to all simulated
        # measurements. Note that error values should be specified in the same
        # units as the simulation.
        error_chain.append(pyv.ErrSysOffset(offset=-10.0))

        # This systematic error samples from a uniform probability distribution.
        error_chain.append(pyv.ErrSysUnif(low=-10.0,
                                             high=10.0))

    if errors_on["rand"]:
        # This random error is generated by sampling from a normal distribution
        # with the given standard deviation in simulation units.
        error_chain.append(pyv.ErrRandNorm(std=5.0))

        # This random error is generated as a percentage sampled from uniform
        # probability distribution
        error_chain.append(pyv.ErrRandUnifPercent(low_percent=-5.0,
                                                  high_percent=5.0))


    # By default pyvale does not store all individual error source
    # calculations (i.e. only the total random and total systematic error are
    # stored) to save memory but this can be changed using `ErrIntOpts`. This
    # can also be used to force all errors to behave if they

    if len(error_chain) > 0:
        err_int_opts = pyv.ErrIntOpts()
        error_integrator = pyv.ErrIntegrator(error_chain,
                                             sensor_data,
                                             tc_array.get_measurement_shape(),
                                             err_int_opts=err_int_opts)
        tc_array.set_error_integrator(error_integrator)


    # Now that we have added our error chain we can run a simulation to sample
    # from all our error sources.
    measurements = tc_array.calc_measurements()

    # We display the simulation results by printing to the console and by
    # plotting the sensor times traces. Try experimenting with the errors above
    # to see how the results change.
    print("\n"+80*"-")
    print("For a virtual sensor: measurement = truth + sysematic error + random error")
    print(f"measurements.shape = {measurements.shape} = "+
          "(n_sensors,n_field_components,n_timesteps)\n")
    print("The truth, systematic error and random error arrays have the same "+
          "shape.")

    print(80*"-")

    sens_print: int = 0
    time_print: int = 5
    comp_print: int = 0

    print(f"These are the last {time_print} virtual measurements of sensor "
          + f"{sens_print}:")

    pyv.print_measurements(sens_array=tc_array,
                           sensors=(sens_print,sens_print+1),
                           components=(comp_print,comp_print+1),
                           time_steps=(measurements.shape[2]-time_print,
                                       measurements.shape[2]))
    print(80*"-")

    (fig,ax) = pyv.plot_time_traces(tc_array,field_key)

    save_traces = output_path/"customsensors_ex1_3_sensortraces.png"
    fig.savefig(save_traces, dpi=300, bbox_inches="tight")
    fig.savefig(save_traces.with_suffix(".svg"), dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
