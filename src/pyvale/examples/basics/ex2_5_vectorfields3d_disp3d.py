# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: 3D vector field sensors
--------------------------------------------------------------------------------
In all our previous examples we have looked at a 2D solid mechanics simulation
and applied displacement sensors to the vector field. Here we will build a
custom vector field sensor array on a 3D simulation of a small linear elastic
cube loaded in tension with the addition of an applied thermal gradient.

Note that this tutorial assumes you are familiar with the use of pyvale for
scalar fields as described in the first set of examples.

Test case: Simple 3D cube thermo-mechanical in tension with temp gradient.
"""

import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv


def main() -> None:
    # Frist we load our simulation as a `SimData` object. In this case we are
    # loading a 10mm cube loaded in tension in the y direction with the addition
    # of a thermal gradient in the y direction.
    data_path = pyv.DataSet.element_case_path(pyv.EElemTest.HEX20)
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    # As we are creating a 3D vector field sensor we now have a third
    # displacement field component here.
    field_name = "disp"
    field_comps = ("disp_x","disp_y","disp_z")
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=field_comps)

    # We use a helper function to print the extent of the dimensions in our
    # `SimData` object to help us locate our sensors on the cube.
    pyv.print_dimensions(sim_data)

    descriptor = pyv.SensorDescriptorFactory.displacement_descriptor()

    # We pass in the string keys for the three vector field components as they
    # appear in our `SimData` object as well as specifying that our elements are
    # 3 dimensional.
    disp_field = pyv.FieldVector(sim_data,field_name,field_comps,elem_dims=3)

    # Here we manually define our sensor positions to place a sensor on the
    # centre of each face of our 10mm cube. From here everything is the same as
    # for our 2D vector field sensor arrays.
    sensor_positions = np.array(((5.0,0.0,5.0),
                                 (5.0,10.0,5.0),
                                 (5.0,5.0,0.0),
                                 (5.0,5.0,10.0),
                                 (0.0,5.0,5.0),
                                 (10.0,5.0,5.0),))

    sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sensor_data = pyv.SensorData(positions=sensor_positions,
                                 sample_times=sample_times)

    disp_sens_array = pyv.SensorArrayPoint(sensor_data,
                                           disp_field,
                                           descriptor)

    measurements = disp_sens_array.calc_measurements()

    # Let's have a look at the y displacement field in relation to the location
    # of our displacement sensors.
    pv_plot = pyv.plot_point_sensors_on_sim(disp_sens_array,"disp_y")
    pv_plot.show()

    # We print the results for one of the sensors so we can see what the errors
    # are for the last few sampling times.
    print(80*"-")

    sens_print: int = 0
    time_print: int = 5
    comp_print: int = 0

    print(f"These are the last {time_print} virtual measurements of sensor "
          + f"{sens_print} for {field_comps[comp_print]}:")

    pyv.print_measurements(sens_array=disp_sens_array,
                           sensors=(sens_print,sens_print+1),
                           components=(comp_print,comp_print+1),
                           time_steps=(measurements.shape[2]-time_print,
                                       measurements.shape[2]))
    print(80*"-")


    # Finally, we plot the time traces for all field components noting that we
    # expect the bottom of the cube to be fixed, the top of the cube to have the
    # maximum y displacement, and that all sensors on the sides of the cube
    # should give the same results.
    for ff in field_comps:
        pyv.plot_time_traces(disp_sens_array,ff)

    plt.show()


if __name__ == "__main__":
    main()