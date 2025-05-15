# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pyvale

def main() -> None:
    (sim_data,_) = pyvale.AnalyticCaseFactory.scalar_linear_2d()

    descriptor = pyvale.SensorDescriptorFactory.temperature_descriptor()

    field_key = 'scalar'
    t_field = pyvale.FieldScalar(sim_data,
                                 field_key=field_key,
                                 elem_dims=2)

    n_sens = (4,1,1)
    x_lims = (0.0,10.0)
    y_lims = (0.0,7.5)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sensor_data = pyvale.SensorData(positions=sens_pos,
                                         sample_times=sample_times)

    tc_array = pyvale.SensorArrayPoint(sensor_data,
                                       t_field,
                                       descriptor)

    errors_on = {'indep_sys': True,
                 'rand': True,
                 'dep_sys': True}

    error_chain = []
    if errors_on['indep_sys']:
        error_chain.append(pyvale.ErrSysOffset(offset=-5.0))
        error_chain.append(pyvale.ErrSysUnif(low=-5.0,
                                            high=5.0))
        gen_norm = pyvale.GenNormal(std=1.0)

    if errors_on['rand']:
        error_chain.append(pyvale.ErrRandNormPercent(std_percent=1.0))
        error_chain.append(pyvale.ErrRandUnifPercent(low_percent=-1.0,
                                            high_percent=1.0))

    if errors_on['dep_sys']:
        error_chain.append(pyvale.ErrSysDigitisation(bits_per_unit=2**8/100))
        error_chain.append(pyvale.ErrSysSaturation(meas_min=0.0,meas_max=300.0))

    if len(error_chain) > 0:
        error_integrator = pyvale.ErrIntegrator(error_chain,
                                                  sensor_data,
                                                  tc_array.get_measurement_shape())
        tc_array.set_error_integrator(error_integrator)

    measurements = tc_array.get_measurements()

    pyvale.print_measurements(tc_array,
                            (0,1), # Sensor 1
                            (0,1), # Component 1: scalar field = 1 component
                            (measurements.shape[2]-5,measurements.shape[2]))

    (fig,_) = pyvale.plot_time_traces(tc_array,field_key)
    plt.show()

    pv_plot = pyvale.plot_point_sensors_on_sim(tc_array,field_key)
    pv_plot.show(cpos="xy")

if __name__ == '__main__':
    main()
