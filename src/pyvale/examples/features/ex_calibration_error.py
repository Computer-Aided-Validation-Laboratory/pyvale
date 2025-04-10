'''
================================================================================
DEV: calibration check

pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''
import matplotlib.pyplot as plt
import numpy as np
import mooseherder as mh
import pyvale


def assumed_calib(signal: np.ndarray) -> np.ndarray:
    return 24.3*signal + 0.616


def truth_calib(signal: np.ndarray) -> np.ndarray:
    return -0.01897 + 25.41881*signal - 0.42456*signal**2 + 0.04365*signal**3


def main() -> None:
    """pyvale example: sensor calibration error
    Based on K-type thermocouple calibration
    """
    n_divs = 10000
    signal_calib_range = np.array((0,6))
    v = np.linspace(signal_calib_range[0],signal_calib_range[1],n_divs)

    temp_truth = -0.01897 + 25.41881*v - 0.42456*v**2 + 0.04365*v**3
    temp_assumed = 24.3*v + 0.616
    calib_error = temp_assumed - temp_truth
    print()
    print(80*"-")
    print(f"Calibration error over signal:"+\
          f" {signal_calib_range[0]} to {signal_calib_range[1]} mV")
    print(calib_error)
    print(f"Max calib error: {np.max(calib_error)}")
    print(f"Min calib error: {np.min(calib_error)}")
    print()

    data_path = pyvale.DataSet.thermal_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0

    descriptor = pyvale.SensorDescriptorFactory.temperature_descriptor()

    field_key = 'temperature'
    t_field = pyvale.FieldScalar(sim_data,
                                 field_key=field_key,
                                 spat_dims=2)

    n_sens = (4,1,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sens_data = pyvale.SensorData(positions=sens_pos,
                                  sample_times=sample_times)

    tc_array = pyvale.SensorArrayPoint(sens_data,
                                       t_field,
                                       descriptor)

    cal_err = pyvale.ErrSysCalibration(assumed_calib,
                                        truth_calib,
                                        signal_calib_range,
                                        n_cal_divs=10000)
    sys_err_int = pyvale.ErrIntegrator([cal_err],
                                       sens_data,
                                       tc_array.get_measurement_shape())
    tc_array.set_error_integrator(sys_err_int)

    measurements = tc_array.get_measurements()

    print('\n'+80*'-')
    print('For a sensor: measurement = truth + sysematic error + random error')
    print(f'measurements.shape = {measurements.shape} = '+
          '(n_sensors,n_field_components,n_timesteps)\n')
    print("The truth, systematic error and random error arrays have the same "+
          "shape.")

    print(80*'-')
    print('Looking at the last 5 time steps (measurements) of sensor 0:')
    pyvale.print_measurements(tc_array,
                              (0,1),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))
    print(80*'-')

    pyvale.plot_time_traces(tc_array,field_key)
    plt.show()




if __name__ == "__main__":
    main()