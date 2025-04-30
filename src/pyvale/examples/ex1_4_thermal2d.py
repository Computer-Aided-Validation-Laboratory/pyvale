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
    """pyvale example: point sensors on a 2D thermal simulation
    ----------------------------------------------------------------------------
    - Demonstrates options for controlling plots of points sensor traces using
      matplotlib
    """
    data_path = pyv.DataSet.thermal_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_key = list(sim_data.node_vars.keys())[0] # type: ignore
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    n_sens = (4,1,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sample_times = np.linspace(0.0,np.max(sim_data.time),12)

    sens_data = pyv.SensorData(positions=sens_pos,
                                  sample_times=sample_times)

    tc_array = pyv.SensorArrayFactory \
        .thermocouples_basic_errs(sim_data,
                                  sens_data,
                                  field_key,
                                  spat_dims=2)

    err_int = pyv.ErrIntegrator([pyv.ErrSysOffset(offset=-5.0)],
                                     sens_data,
                                     tc_array.get_measurement_shape())
    tc_array.set_error_integrator(err_int)

    measurements = tc_array.get_measurements()

    print(80*"-")
    print("Looking at the last 5 time steps (measurements) of sensor 0:")
    pyv.print_measurements(tc_array,
                              (0,1),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))
    print(80*"-")

    trace_props = pyv.TraceOptsSensor()

    trace_props.truth_line = None
    trace_props.sim_line = None
    pyv.plot_time_traces(tc_array,field_key,trace_props)

    trace_props.meas_line = "--o"
    trace_props.truth_line = "-x"
    trace_props.sim_line = ":+"
    pyv.plot_time_traces(tc_array,field_key,trace_props)

    trace_props.sensors_to_plot = np.arange(measurements.shape[0]-2
                                           ,measurements.shape[0])
    pyv.plot_time_traces(tc_array,field_key,trace_props)

    trace_props.sensors_to_plot = None
    trace_props.time_min_max = (0.0,100.0)
    pyv.plot_time_traces(tc_array,field_key,trace_props)

    plt.show()

    pv_plot = pyv.plot_point_sensors_on_sim(tc_array,field_key)
    pv_plot.camera_position = [(-7.547, 59.753, 134.52),
                                   (41.916, 25.303, 9.297),
                                   (0.0810, 0.969, -0.234)]
    pv_plot.show()


if __name__ == "__main__":
    main()
