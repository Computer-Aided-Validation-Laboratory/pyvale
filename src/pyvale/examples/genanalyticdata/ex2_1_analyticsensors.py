# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pyvale.sensorsim as sens
import pyvale.verif.analyticsimdatafactory as asd

def main() -> None:
    # 10x7.5 plate with bi-directional field gradient
    # 40x30 elements [x,y]
    # Field slope of 20/lengX in X
    # Field slope of 10/lengY in Y
    # Field max in top corner of 220, field min in bottom corner 20
    (sim_data,_) = asd.scalar_linear_2d()

    descriptor = sens.SensorDescriptorFactory.temperature_descriptor()

    field_key = 'scalar'
    t_field = sens.FieldScalar(sim_data,
                              field_key=field_key,
                              elem_dims=2)

    n_sens = (4,1,1)
    x_lims = (0.0,10.0)
    y_lims = (0.0,7.5)
    z_lims = (0.0,0.0)
    sens_pos = sens.gen_pos_grid_inside(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sensor_data = sens.SensorData(positions=sens_pos,
                                         sample_times=sample_times)

    tc_array = sens.SensorsPoint(sensor_data,
                                       t_field,
                                       descriptor)

    errors_on = {'indep_sys': True,
                 'rand': True,
                 'dep_sys': True}

    error_chain = []
    if errors_on['indep_sys']:
        error_chain.append(sens.ErrSysOffset(offset=-5.0))
        error_chain.append(sens.ErrSysUnif(low=-5.0,
                                            high=5.0))
        gen_norm = sens.GenNormal(std=1.0)

    if errors_on['rand']:
        error_chain.append(sens.ErrRandNormPercent(std_percent=1.0))
        error_chain.append(sens.ErrRandUnifPercent(low_percent=-1.0,
                                            high_percent=1.0))

    if errors_on['dep_sys']:
        error_chain.append(sens.ErrSysDigitisation(bits_per_unit=2**8/100))
        error_chain.append(sens.ErrSysSaturation(meas_min=0.0,meas_max=300.0))

    if len(error_chain) > 0:
        error_integrator = sens.ErrIntegrator(error_chain,
                                                  sensor_data,
                                                  tc_array.get_measurement_shape())
        tc_array.set_error_integrator(error_integrator)

    measurements = tc_array.get_measurements()

    sens.print_measurements(tc_array,
                            slice(0,1), # Sensor 1
                            slice(0,1), # Component 1: scalar field = 1 component
                            slice (measurements.shape[2]-5,measurements.shape[2]))

    (fig,_) = sens.plot_time_traces(tc_array,field_key)
    plt.show()

    pv_plot = sens.plot_point_sensors_on_sim(tc_array,field_key)
    pv_plot.show(cpos="xy")

if __name__ == '__main__':
    main()
