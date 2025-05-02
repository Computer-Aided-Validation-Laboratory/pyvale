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

    sample_times = np.linspace(0.0,np.max(sim_data.time),50) # | None

    sensor_data = pyv.SensorData(positions=sens_pos,
                                    sample_times=sample_times)

    tc_array = pyv.SensorArrayFactory \
        .thermocouples_no_errs(sim_data,
                               sensor_data,
                               field_key,
                               spat_dims=2)

    #===========================================================================
    # Examples of full error library

    #---------------------------------------------------------------------------
    # Standard independent systematic errors
    err_chain = []
    err_chain.append(pyv.ErrSysOffset(offset=-1.0))
    err_chain.append(pyv.ErrSysOffsetPercent(offset_percent=-1.0))

    err_chain.append(pyv.ErrSysUniform(low=-2.0,
                                        high=2.0))
    err_chain.append(pyv.ErrSysUniformPercent(low_percent=-2.0,
                                                    high_percent=2.0))

    err_chain.append(pyv.ErrSysNormal(std=1.0))
    err_chain.append(pyv.ErrSysNormPercent(std_percent=2.0))

    sys_gen = pyv.GeneratorTriangular(left=-1.0,
                                         mode=0.0,
                                         right=1.0)
    err_chain.append(pyv.ErrSysGenerator(sys_gen))

    #---------------------------------------------------------------------------
    err_chain.append(pyv.ErrRandNormal(std = 2.0))
    err_chain.append(pyv.ErrRandNormPercent(std_percent=2.0))

    err_chain.append(pyv.ErrRandUniform(low=-2.0,high=2.0))
    err_chain.append(pyv.ErrRandUnifPercent(low_percent=-2.0,
                                               high_percent=2.0))

    rand_gen = pyv.GeneratorTriangular(left=-5.0,
                                          mode=0.0,
                                          right=5.0)
    err_chain.append(pyv.ErrRandGenerator(rand_gen))

    #---------------------------------------------------------------------------
    err_chain.append(pyv.ErrSysDigitisation(bits_per_unit=2**8/100))
    err_chain.append(pyv.ErrSysSaturation(meas_min=0.0,meas_max=300.0))

    err_int = pyv.ErrIntegrator(err_chain,
                                     sensor_data,
                                     tc_array.get_measurement_shape())
    tc_array.set_error_integrator(err_int)


    #===========================================================================

    measurements = tc_array.calc_measurements()
    print(80*'-')
    sens_num = 4
    print('The last 5 time steps (measurements) of sensor {sens_num}:')
    pyv.print_measurements(tc_array,
                              (sens_num-1,sens_num),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))
    print(80*'-')

    pyv.plot_time_traces(tc_array,field_key)
    plt.show()


if __name__ == '__main__':
    main()
