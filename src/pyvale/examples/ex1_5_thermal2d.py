'''
================================================================================
Example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale


def main() -> None:
    """pyvale example: point sensors on a 2D thermal simulation
    ----------------------------------------------------------------------------
    """
    data_path = pyvale.DataSet.thermal_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_key = list(sim_data.node_vars.keys())[0] # type: ignore
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    n_sens = (4,1,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sample_times = np.linspace(0.0,np.max(sim_data.time),50) # | None

    sensor_data = pyvale.SensorData(positions=sens_pos,
                                    sample_times=sample_times)

    tc_array = pyvale.SensorArrayFactory \
        .thermocouples_no_errs(sim_data,
                               sensor_data,
                               field_key,
                               spat_dims=2)

    #===========================================================================
    # Examples of full error library

    #---------------------------------------------------------------------------
    # Standard independent systematic errors
    err_chain = []
    err_chain.append(pyvale.ErrSysOffset(offset=-1.0))
    err_chain.append(pyvale.ErrSysOffsetPercent(offset_percent=-1.0))

    err_chain.append(pyvale.ErrSysUniform(low=-2.0,
                                        high=2.0))
    err_chain.append(pyvale.ErrSysUniformPercent(low_percent=-2.0,
                                                    high_percent=2.0))

    err_chain.append(pyvale.ErrSysNormal(std=1.0))
    err_chain.append(pyvale.ErrSysNormPercent(std_percent=2.0))

    sys_gen = pyvale.GeneratorTriangular(left=-1.0,
                                         mode=0.0,
                                         right=1.0)
    err_chain.append(pyvale.ErrSysGenerator(sys_gen))

    #---------------------------------------------------------------------------
    err_chain.append(pyvale.ErrRandNormal(std = 2.0))
    err_chain.append(pyvale.ErrRandNormPercent(std_percent=2.0))

    err_chain.append(pyvale.ErrRandUniform(low=-2.0,high=2.0))
    err_chain.append(pyvale.ErrRandUnifPercent(low_percent=-2.0,
                                               high_percent=2.0))

    rand_gen = pyvale.GeneratorTriangular(left=-5.0,
                                          mode=0.0,
                                          right=5.0)
    err_chain.append(pyvale.ErrRandGenerator(rand_gen))

    #---------------------------------------------------------------------------
    err_chain.append(pyvale.ErrSysDigitisation(bits_per_unit=2**8/100))
    err_chain.append(pyvale.ErrSysSaturation(meas_min=0.0,meas_max=300.0))

    err_int = pyvale.ErrIntegrator(err_chain,
                                     sensor_data,
                                     tc_array.get_measurement_shape())
    tc_array.set_error_integrator(err_int)


    #===========================================================================

    measurements = tc_array.calc_measurements()
    print(80*'-')
    sens_num = 4
    print('The last 5 time steps (measurements) of sensor {sens_num}:')
    pyvale.print_measurements(tc_array,
                              (sens_num-1,sens_num),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))
    print(80*'-')

    pyvale.plot_time_traces(tc_array,field_key)
    plt.show()


if __name__ == '__main__':
    main()
