# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: Basic displacement (vector) field sensors
----------------------------------------------------------------------------
In this example we use the sensor array factory to build a set of  displacement
sensors that can sample the displacement vector field from a solid mechanics
simulation. In the next example we will examine how we can build custom vector
field sensors as we did for scalar field in the first set of examples.

Note that this tutorial assumes you are familiar with the use of pyvale for
scalar fields as described in the first set of examples.

Test case: point displacement sensors on a 2D plate with hole loaded in tension
"""


import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv

def main() -> None:
    # Here we load a pre-packaged dataset from pyvale that is the output of a
    # MOOSE simulation in exodus format. The simulation is a linear elastic
    # rectangular plate with a central hole that is loaded in tension (we will
    # see a visualisation of the mesh and results later).
    data_path = pyv.DataSet.mechanical_2d_path()
    # We use `mooseherder` to load the exodus file into a `SimData` object.
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    # We scale our SI simulation to mm including the displacement fields which
    # are also in length units. The string keys we have provided here must match
    # the variable names you have in your SimData object.
    field_name = "disp"
    field_comps = ("disp_x","disp_y")
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=field_comps)

    # Creating a displacement field point sensor array is similar to what we
    # have already done for scalar fields we just need to specify the string
    # keys for the displacement fields in the sim data object we have loaded.
    # For 2D vector fields we expect to have 2 components which are typically:
    # ("disp_x","disp_y"). For 3D vector fields we have 3 field components which
    # are typically: ("disp_x","disp_y","disp_z").
    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sens_data = pyv.SensorData(positions=sens_pos)

    disp_sens_array = pyv.SensorArrayFactory \
                        .disp_sensors_basic_errs(sim_data,
                                                 sens_data,
                                                 elem_dims=2,
                                                 field_name=field_name,
                                                 field_comps=field_comps,
                                                 errs_pc=2.0)

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