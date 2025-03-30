'''
================================================================================
Example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
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

    #---------------------------------------------------------------------------
    # Standard independent systematic errors
    pos_rand = pyvale.GeneratorNormal(std=1.0) # mm
    pos_lock = np.full_like(sensor_data.positions,False,dtype=bool)
    pos_lock[:,2] = True
    field_err_data = pyvale.ErrFieldData(
        pos_rand_xyz=(pos_rand,pos_rand,pos_rand),
        pos_lock_xyz=pos_lock
    )

    err_chain = []
    err_chain.append(pyvale.ErrSysField(tc_array.get_field(),
                                        field_err_data))
    err_int = pyvale.ErrIntegrator(err_chain,
                                   sensor_data,
                                   tc_array.get_measurement_shape())
    tc_array.set_error_integrator(err_int)


    #---------------------------------------------------------------------------
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
