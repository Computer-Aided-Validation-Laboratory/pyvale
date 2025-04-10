"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale

def main() -> None:
    data_path = Path('src/data/case13_out.e')
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore


    descriptor = pyvale.SensorDescriptorFactory.temperature_descriptor()

    field_key = 'temperature'
    t_field = pyvale.FieldScalar(sim_data,
                                 field_key=field_key,
                                 spat_dim=2)

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


    tc_array = pyvale.SensorArrayPoint(sens_pos,
                                       t_field,
                                       sample_times,
                                       descriptor)


    drift_1 = pyvale.DriftConstant(offset=0.0)
    indep_sys_err1 = pyvale.ErrSysTimeDrift(t_field,
                                            sens_pos,
                                            drift_1,
                                            sample_times)

    drift_2 = pyvale.DriftLinear(slope=0.0)
    indep_sys_err2 = pyvale.ErrSysTimeDrift(t_field,
                                            sens_pos,
                                            drift_2,
                                            sample_times)

    indep_sys_err3 = pyvale.ErrSysTimeRand(t_field,
                                            sens_pos,
                                            time_std=5.0,
                                            sample_times=sample_times)

    indep_sys_err_int = pyvale.ErrIntegrator([indep_sys_err1,
                                                indep_sys_err2,
                                                indep_sys_err3],
                                        tc_array.get_measurement_shape())

    tc_array.set_systematic_err_integrator_independent(indep_sys_err_int)

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