# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv


def main() -> None:
    """pyvale example: Point sensors on a 2D thermal simulation
    ----------------------------------------------------------------------------
    - Full construction of a point sensor array from scratch
    - Explanation of the different types of error models
    - There are flags throughout the example allowing the user to toggle on/off
      parts of the sensor array construction
    """
    data_path = pyv.DataSet.thermal_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    use_auto_descriptor = "blank"
    if use_auto_descriptor == "factory":
        descriptor = pyv.SensorDescriptorFactory.temperature_descriptor()
    elif use_auto_descriptor == "manual":
        descriptor = pyv.SensorDescriptor()
        descriptor.name = "Temperature"
        descriptor.symbol = "T"
        descriptor.units = r"^{\circ}C"
        descriptor.tag = "TC"
    else:
        descriptor = pyv.SensorDescriptor()

    field_key = "temperature" # ("disp_x","disp_y")
    t_field = pyv.FieldScalar(sim_data,
                                 field_key=field_key,
                                 spat_dims=2)

    n_sens = (4,1,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sensor_data = pyv.SensorData(positions=sens_pos,
                                   sample_times=sample_times)

    tc_array = pyv.SensorArrayPoint(sensor_data,
                                       t_field,
                                       descriptor)

    errors_on = {"indep_sys": True,
                 "rand": True,
                 "dep_sys": True}

    error_chain = []
    if errors_on["indep_sys"]:
        error_chain.append(pyv.ErrSysOffset(offset=-5.0))
        error_chain.append(pyv.ErrSysUniform(low=-5.0,
                                                high=5.0))

    if errors_on["rand"]:
        error_chain.append(pyv.ErrRandNormPercent(std_percent=1.0))
        error_chain.append(pyv.ErrRandUnifPercent(low_percent=-1.0,
                                            high_percent=1.0))

    if errors_on["dep_sys"]:
        error_chain.append(pyv.ErrSysDigitisation(bits_per_unit=2**8/100))
        error_chain.append(pyv.ErrSysSaturation(meas_min=0.0,meas_max=300.0))

    if len(error_chain) > 0:
        error_integrator = pyv.ErrIntegrator(error_chain,
                                                  sensor_data,
                                                  tc_array.get_measurement_shape())
        tc_array.set_error_integrator(error_integrator)


    measurements = tc_array.get_measurements()

    print("\n"+80*"-")
    print("For a sensor: measurement = truth + sysematic error + random error")
    print(f"measurements.shape = {measurements.shape} = "+
          "(n_sensors,n_field_components,n_timesteps)\n")
    print("The truth, systematic error and random error arrays have the same "+
          "shape.")

    print(80*"-")
    print("Looking at the last 5 time steps (measurements) of sensor 0:")
    pyv.print_measurements(tc_array,
                              (0,1),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))
    print(80*"-")

    pyv.plot_time_traces(tc_array,field_key)
    plt.show()


if __name__ == "__main__":
    main()
