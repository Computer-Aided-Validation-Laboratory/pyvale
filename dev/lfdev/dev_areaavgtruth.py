'''
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale


def main() -> None:
    data_path = Path('src/data/case13_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    descriptor = pyvale.SensorDescriptorFactory.temperature_descriptor()
    field_key = 'temperature'
    t_field = pyvale.ScalarField(sim_data,
                                 field_key=field_key,
                                 spat_dim=2)

    n_sens = (4,1,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = True
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)


    sens_size = 10.0
    spat_int_truth = pyvale.SpatialIntegratorFactory.quad_2d_4pt(t_field,
                                                           sens_pos,
                                                           np.array((sens_size,sens_size,0.0)))

    tc_array = pyvale.PointSensorArray(sens_pos,
                                       t_field,
                                       sample_times,
                                       descriptor,
                                       area_avg=spat_int_truth)

    spat_int_err = pyvale.SpatialIntegratorFactory.rect_2d_1pt(t_field,
                                                           sens_pos,
                                                           np.array((sens_size,sens_size,0.0)))

    sys_errs = []
    sys_errs.append(pyvale.SysErrSpatialAverage(t_field,
                                                spat_int_err,
                                                sample_times))

    indep_sys_err_int = pyvale.ErrorIntegrator(sys_errs,
                                        tc_array.get_measurement_shape())

    tc_array.set_indep_sys_err_integrator(indep_sys_err_int)

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
                              (0,10))
    print(80*'-')

    pyvale.plot_time_traces(tc_array,field_key)
    plt.show()


if __name__ == '__main__':
    main()