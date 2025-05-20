# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: Sensor spatial averaging and averaging errors
--------------------------------------------------------------------------------
In this example we show how pyvale can simulate sensor spatial averaging for
ground truth calculations as well as for calculating systematic errors.

Test case: Scalar field point sensors (thermocouples) on a 2D thermal simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv


def main() -> None:
    # First we are going to build a custom sensor array so we can control how
    # the ground truth is extracted for a sensor using area averaging. Note that
    # the default is an ideal point sensor with no spatial averaging. Later we
    # will add area averaging as a systematic error. Note that it is possible to
    # have an ideal point sensor with no area averaging for the truth and then
    # add an area averaging error. It is also possible to have a truth that is
    # area averaged without and area averaging error. The first part of this is
    # the same as the 3D thermal example we have used previously then we control
    # the area averaging using the sensor data object.
    data_path = pyv.DataSet.thermal_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=None)

    descriptor = pyv.SensorDescriptorFactory.temperature_descriptor()

    field_key = "temperature"
    t_field = pyv.FieldScalar(sim_data,
                                 field_key=field_key,
                                 elem_dims=2)

    n_sens = (4,1,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sample_times = np.linspace(0.0,np.max(sim_data.time),50) # | None

    # This is where we control the setup of the area averaging. We need to
    # specify the sensor dimensions and the type of numerical spatial
    # integration to use. Here we specify a square sensor in x and y with 4
    # point Gaussian quadrature integration. It is worth noting that increasing
    # the number of integration points will increase computational cost as each
    # additional integration point requires an additional interpolation of the
    # physical field.
    sensor_dims = np.array([20.0,20.0,0]) # units = mm
    sensor_data = pyv.SensorData(positions=sens_pos,
                                 sample_times=sample_times,
                                 spatial_averager=pyv.EIntSpatialType.QUAD4PT,
                                 spatial_dims=sensor_dims)

    # We have added spatial averaging to our sensor data so we can now create
    # our sensor array as we have done in previous examples.
    tc_array = pyv.SensorArrayPoint(sensor_data,
                                    t_field,
                                    descriptor)

    # We are also going to create a field error that includes area averaging as
    # an error. We do this by adding the option to our field error data class
    # specifying rectangular integration with 1 point.
    area_avg_err_data = pyv.ErrFieldData(
        spatial_averager=pyv.EIntSpatialType.RECT1PT,
        spatial_dims=np.array((5.0,5.0)),
    )

    # We add the field error to our error chain as normal. We could combine it
    # with any of our other error models but we will isolate it for now so we
    # can see what it does.
    err_chain = []
    err_chain.append(pyv.ErrSysField(t_field,
                                     area_avg_err_data))
    error_int = pyv.ErrIntegrator(err_chain,
                                  sensor_data,
                                  tc_array.get_measurement_shape())
    tc_array.set_error_integrator(error_int)

    # Now we run our sensor simulation to see how spatial averaging changes our
    measurements = tc_array.calc_measurements()

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

    pyv.plot_time_traces(tc_array,field_key)
    plt.show()

    # From here you now have everything you need to build your own sensor
    # simulations for scalar field sensors using pyvale. In the next examples
    # we will look at sensors applied to vector (e.g. displacement) and tensor
    # fields (e.g. strain). If you don't need to sample vector or tensor fields
    # then skip ahead to the examples on experiment simulation where you will
    # learn how to perform Monte-Carlo sensor uncertainty quantification
    # simulations and to analyse the results with pyvale.


if __name__ == "__main__":
    main()
