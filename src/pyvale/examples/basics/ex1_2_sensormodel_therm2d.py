# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: Sensor model & `get_measurements()` vs `calc_measurements()`
--------------------------------------------------------------------------------
In this example we explain the pyvale virtual sensor measurement model. For a
virtual sensor in pyvale a measurement is defined as measurement = truth +
systematic error + random error. Sources of systematic errors include: spatial/
temporal averaging, uncertainty in position / sampling time / orientation,
digitisation, saturation, and calibration. Sources of random error are generally
due to measurement noise characterised by a given probability distribution.

Random errors can be mitigated by performing multiple experiments and averaging.
However, systematic errors cannot easily be accounted for without a forward
model of the source of the error. Characterising the contribution of systematic
errors to the total measurement error is a key application of `pyvale`.

between the `get_measurements()` and `calc_measurements()` methods for a sensor
array. Calling `get_measurements()` retrieves the results for the current
simulated experiment whereas calling `calc_measurements()` will generate a new
simulated experiment by sampling / calculating the systematic and random errors.

Test case: Scalar field point sensors (thermocouples) on a 2D thermal simulation
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
                                  elem_dims=2,
                                  field_name=field_key,
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
    print("For a sensor array: "
          + "measurement = truth + sysematic error + random error")
    print(f"\nmeasurements.shape = {measurements.shape} = "
          + "(n_sensors,n_field_components,n_timesteps)\n")
    print("Here we have a scalar temperature field so only 1 field component.")
    print("The truth, systematic error and random error arrays all have the "+
          "same shape.")

    sens_print: int = 0
    time_print: int = 5
    comp_print: int = 0

    print(80*"-")
    print(f"Looking at the last {time_print} virtual measurements" +
          f" of sensor {sens_print}:")

    pyv.print_measurements(sens_array=tc_array,
                           sensors=(sens_print,sens_print+1),
                           components=(comp_print,comp_print+1),
                           time_steps=(measurements.shape[2]-time_print,
                                       measurements.shape[2]))
    print(80*"-")
    print("If we call the `calc_measurements()` method then the errors are "
          + "re-calculated.")
    measurements = tc_array.calc_measurements()

    pyv.print_measurements(sens_array=tc_array,
                           sensors=(sens_print,sens_print+1),
                           components=(comp_print,comp_print+1),
                           time_steps=(measurements.shape[2]-time_print,
                                       measurements.shape[2]))

    (fig,ax) = pyv.plot_time_traces(tc_array,field_key)
    ax.set_title("Exp 1: called calc_measurements()")

    print(80*"-")
    print("If we call the `get_measurements()` method then the errors are the "
          + "same:")
    measurements = tc_array.get_measurements()

    pyv.print_measurements(sens_array=tc_array,
                           sensors=(sens_print,sens_print+1),
                           components=(comp_print,comp_print+1),
                           time_steps=(measurements.shape[2]-time_print,
                                       measurements.shape[2]))

    (fig,ax) = pyv.plot_time_traces(tc_array,field_key)
    ax.set_title("Exp 2: called get_measurements()")

    print(80*"-")
    print("If we call the `calc_measurements()` method again we generate / "
          "sample new errors:")
    measurements = tc_array.calc_measurements()

    pyv.print_measurements(sens_array=tc_array,
                           sensors=(sens_print,sens_print+1),
                           components=(comp_print,comp_print+1),
                           time_steps=(measurements.shape[2]-time_print,
                                       measurements.shape[2]))

    (fig,ax) = pyv.plot_time_traces(tc_array,field_key)
    ax.set_title("Exp 3: called calc_measurements()")

    print(80*"-")

    plt.show()

if __name__ == "__main__":
    main()
