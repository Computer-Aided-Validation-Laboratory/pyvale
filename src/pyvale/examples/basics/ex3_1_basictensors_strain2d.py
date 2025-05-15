# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: Basic tensor field sensors (strain gauges)
--------------------------------------------------------------------------------
In this example we use the sensor array factory to build a set of strain
sensors that can sample the strain tensor field from a solid mechanics
simulation. In the next example we will examine how we can build custom tensor
field sensors as we did for scalar field in the first set of examples.

Note that this tutorial assumes you are familiar with the use of pyvale for
scalar fields as described in the first set of examples.

Test case: point strain sensors on a 2D plate with hole loaded in tension
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
from mooseherder import SimData
import pyvale as pyv

def main() -> None:
    # First we load the same 2D solid mechanics simulation we used previously
    # for vector displacement fields. Most of this setup code is similar to our
    # vector field examples except we will need to specify the string keys for
    # the normal a deviatoric components of our tensor field (as they appear in
    # our `SimData` object).
    data_path: Path = pyv.DataSet.mechanical_2d_path()
    sim_data: SimData = mh.ExodusReader(data_path).read_all_sim_data()
    sim_data: SimData = pyv.scale_length_units(scale=1000.0,
                                                sim_data=sim_data,
                                                disp_comps=("disp_x","disp_y"))

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sens_data = pyv.SensorData(positions=sens_pos,
                               sample_times=sample_times)

    # This is where we need to specify the string keys for the normal and
    # deviatoric components of our strain field. In 2D we have two normal
    # normal components and one deviatoric. In 3D we will have 3 of each as we
    # will see in a later example. Otherwise this is very similar to what we
    # have seen previously for scalar and vector fields.
    norm_comps = ("strain_xx","strain_yy")
    dev_comps = ("strain_xy",)
    straingauge_array = pyv.SensorArrayFactory \
                            .strain_gauges_basic_errs(sim_data,
                                                      sens_data,
                                                      elem_dims=2,
                                                      field_name="strain",
                                                      norm_comps=norm_comps,
                                                      dev_comps=dev_comps,
                                                      errs_pc=5.0)

    # We run our virtual sensor simulation as normal. The only thing to note is
    # that the second dimension of our measurement array will contain our tensor
    # components in the order they are specified in the tuples with the normal
    # components first followed by the deviatoric. In our case this will be
    # (strain_xx,strain_yy,strain_xy).
    measurements = straingauge_array.calc_measurements()

    # Here we print the shape of the measurement array so we can see that the
    # second dimension contains both our tensor components. We also print some
    # of the sensor measurements for the first tensor component.
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

    pyv.print_measurements(sens_array=straingauge_array,
                           sensors=(sens_print,sens_print+1),
                           components=(comp_print,comp_print+1),
                           time_steps=(measurements.shape[2]-time_print,
                                       measurements.shape[2]))
    print(80*"-")

    # We can plot a given component of our tensor field and display our sensor
    # locations with respect to the field.
    plot_field = "strain_yy"
    pv_plot = pyv.plot_point_sensors_on_sim(straingauge_array,plot_field)
    pv_plot.show(cpos="xy")

    # We can also plot time traces for all components of the tensor field.
    for cc in (norm_comps+dev_comps):
        pyv.plot_time_traces(straingauge_array,cc)

    plt.show()


if __name__ == "__main__":
    main()