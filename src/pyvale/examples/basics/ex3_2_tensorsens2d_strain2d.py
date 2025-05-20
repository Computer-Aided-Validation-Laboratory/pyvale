# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: Custom tensor field sensors (strain gauges) in 2D
--------------------------------------------------------------------------------
In this example we build a custom tensor field sensor array (i.e. a strain gauge
array) in 2D.

Note that this tutorial assumes you are familiar with the use of pyvale for
scalar fields as described in the first set of examples.

Test case: point strain sensors on a 2D plate with hole loaded in tension
"""

import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv

def main() -> None:
    # First we load the same 2D solid mechanics simulation of a plate with a
    # hole loaded in tension as a `SimData` object. We scale the units to mm
    # from SI including the coordinates and displacement. Strain is unitless so
    # we leave it alone.
    data_path = pyv.DataSet.mechanical_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=("disp_x","disp_y"))


    # Here is the main difference when creating a tensor field sensor array. We
    # create a tensor field where we need to specify the normal and deviatoric
    # component string keys as they appear in our `SimData` object. We have a 2d
    # simulation here so we have 2 normal components and 1 deviatoric (shear).
    field_name = "strain"
    norm_comps = ("strain_xx","strain_yy")
    dev_comps = ("strain_xy",)
    strain_field = pyv.FieldTensor(sim_data,
                                   field_name=field_name,
                                   norm_comps=norm_comps,
                                   dev_comps=dev_comps,
                                   elem_dims=2)

    # The setup of our sensor data object is exactly the same as for any other
    # point sensor array. We could optionally specify the sample time to be None
    # in which case the sensor will sample at the simulation time steps.
    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sens_data = pyv.SensorData(positions=sens_pos,
                               sample_times=sample_times)

    # Here we create a descriptor that will be used to label visualisations of
    # the sensor locations and time traces for our sensors. For the strain
    # gauges we are modelling here we could also use the descriptor factory to
    # get these defaults.
    descriptor = pyv.SensorDescriptor(name="Strain",
                                      symbol=r"\varepsilon",
                                      units=r"-",
                                      tag="SG",
                                      components=("xx","yy","xy"))

    # We build our point sensor array as normal.
    straingauge_array = pyv.SensorArrayPoint(sens_data,
                                             strain_field,
                                             descriptor)


    # We can add any errors we like to our error chain. Here we add some basic
    # percentage errors.
    error_chain = []
    error_chain.append(pyv.ErrSysUnifPercent(low_percent=-2.0,high_percent=2.0))
    error_chain.append(pyv.ErrRandNormPercent(std_percent=2.0))
    error_int = pyv.ErrIntegrator(error_chain,
                                  sens_data,
                                  straingauge_array.get_measurement_shape())
    straingauge_array.set_error_integrator(error_int)

    # We run our virtual sensor simulation as normal. The only thing to note is
    # that the second dimension of our measurement array will contain our tensor
    # components in the order they are specified in the tuples with the normal
    # components first followed by the deviatoric. In our case this will be
    # (strain_xx,strain_yy,strain_xy).
    measurements = straingauge_array.calc_measurements()

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