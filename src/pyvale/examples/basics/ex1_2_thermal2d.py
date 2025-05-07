# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""pyvale example: point sensors on a 2D thermal simulation
----------------------------------------------------------------------------
- Explanation of the usage of "get_measurements()" and "calc_measurements()".

Notes:
- In pyvale a virtual sensor measurement is defined as:
  measurement = truth + systematic error + random error.
- Calling the "get" method of the sensor array will retrieve the results for
  the current experiment.
- Calling the "calc" method will generate a new experiment by sampling /
  calculating the systematic and random errors.
"""

import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv


def main() -> None:
    # The first part of this example is the similar to basics example 1.1, so
    # feel free to skip to after the first call to `calc_measurements()`.

    # Here we load a pre-generated MOOSE finite element simulation dataset that
    # comes packaged with pyvale. The simulation is a 2D rectangular plate with
    # a bi-directional temperature gradient.
    data_path = pyv.DataSet.thermal_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_key: str = "temperature"
    # Scale to mm to make 3D visualisation scaling easier as pyvista scales
    # everything to unity
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=None)

    # We now use a helper function to create a grid of sensor locations but we
    # could have also manually built the numpy array of sensor locations which
    # has the shape=(num_sensors,coord[x,y,z]).
    n_sens = (4,1,1)
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
    # This basic thermocouple array includes a 5% systematic and random error -
    # We are specifically using exaggerated errors here for visualisation.
    tc_array = pyv.SensorArrayFactory \
        .thermocouples_basic_errs(sim_data,
                                  sens_data,
                                  field_key,
                                  spat_dims=2,
                                  errs_pc=5.0)


    # We have built our sensor array so now we can call `calc_measurements()` to
    # generate simulated sensor traces.
    measurements = tc_array.calc_measurements()

    # From here we are going to experiment with repeated calls to
    # `calc_measurements()` and `get_measurements()` for our sensor array. We
    # will print the results to the console as well as plotting time traces of
    # the simulated sensor output. All further explanations are in the print
    # statements below.

    print("\n"+80*"-")
    print("For a sensor array: measurement = truth + sysematic error + random error")
    print(f"\nmeasurements.shape = {measurements.shape} = "+
          "(n_sensors,n_field_components,n_timesteps)\n")
    print("Here we have a scalar temperature field so only 1 field component.")
    print("The truth, systematic error and random error arrays all have the same "+
          "shape.")

    print(80*"-")
    print("Looking at the last 5 time steps (measurements) of sensor 0:")
    pyv.print_measurements(tc_array,
                            (0,1),
                            (0,1),
                            (measurements.shape[2]-5,measurements.shape[2]))
    print(80*"-")
    print("If we call the `calc_measurements()` method then the errors are "+
          "re-calculated.")
    measurements = tc_array.calc_measurements()

    pyv.print_measurements(tc_array,
                              (0,1),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))


    (fig,ax) = pyv.plot_time_traces(tc_array,field_key)
    ax.set_title("Exp 1: called calc_measurements()")

    print(80*"-")
    print("If we call the `get_measurements()` method then the errors are the "+
          "same:")
    measurements = tc_array.get_measurements()

    pyv.print_measurements(tc_array,
                              (0,1),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))

    (fig,ax) = pyv.plot_time_traces(tc_array,field_key)
    ax.set_title("Exp 2: called get_measurements()")

    print(80*"-")
    print("If we call the `calc_measurements()` method again we generate/sample"+
           "new errors:")
    measurements = tc_array.calc_measurements()

    pyv.print_measurements(tc_array,
                              (0,1),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))

    (fig,ax) = pyv.plot_time_traces(tc_array,field_key)
    ax.set_title("Exp 3: called calc_measurements()")

    print(80*"-")

    plt.show()

if __name__ == "__main__":
    main()
