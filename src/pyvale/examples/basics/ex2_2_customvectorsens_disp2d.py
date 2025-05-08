# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: Custom vector field sensors
----------------------------------------------------------------------------
In this example we build a custom vector field sensor array

Note that this tutorial assumes you are familiar with the use of pyvale for
scalar fields as described in the first set of examples.

Test case: point displacement sensors on a 2D plate with hole loaded in tension
"""

import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv

def main() -> None:


    data_path = pyv.DataSet.mechanical_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    field_name = "disp"
    field_comps = ("disp_x","disp_y")
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=field_comps)


    descriptor = pyv.SensorDescriptor(name="Disp.",
                                      symbol=r"u",
                                      units=r"mm",
                                      tag="DS",
                                      components=("x","y","z"))


    disp_field = pyv.FieldVector(sim_data,field_name,field_comps,elem_dims=2)

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    # We set custom sampling times here but we could also set this to None so
    # that the sensors sample at the simulation time steps.
    sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sens_data = pyv.SensorData(positions=sens_pos,
                               sample_times=sample_times)

    disp_sens_array = pyv.SensorArrayPoint(sens_data,
                                           disp_field,
                                           descriptor)

    error_chain = []
    error_chain.append(pyv.ErrSysUnif(low=-0.01,high=0.01))  # units = mm
    error_chain.append(pyv.ErrRandNorm(std=0.01))            # units = mm
    error_int = pyv.ErrIntegrator(error_chain,
                                  sens_data,
                                  disp_sens_array.get_measurement_shape())
    disp_sens_array.set_error_integrator(error_int)

    disp_sens_array.calc_measurements()

    # Now that we have multiple field components we can plot each of them on the
    # simulation mesh and visulise the sensor locations with respect to these
    # fields.
    for ff in field_comps:
        pv_plot = pyv.plot_point_sensors_on_sim(disp_sens_array,ff)
        pv_plot.show(cpos="xy")

    # We can also plot the traces for each component of the displacement field.
    for ff in field_comps:
        pyv.plot_time_traces(disp_sens_array,ff)

    plt.show()


if __name__ == "__main__":
    main()