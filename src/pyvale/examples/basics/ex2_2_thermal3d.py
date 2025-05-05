'''
================================================================================
Example: 3d thermocouples on a monoblock

pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''
import mooseherder as mh
import pyvale as pyv


def main() -> None:
    """pyvale example: thermocouples on a 3D divertor monoblock heatsink
    ----------------------------------------------------------------------------
    """
    data_path = pyv.DataSet.thermal_3d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_name = 'temperature'

    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    n_sens = (1,4,1)
    x_lims = (12.5,12.5)
    y_lims = (0,31.0)
    z_lims = (0.0,12.5)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sens_data = pyv.SensorData(positions=sens_pos)

    tc_array = pyv.SensorArrayFactory() \
        .thermocouples_basic_errs(sim_data,
                                  sens_data,
                                  field_name,
                                  spat_dims=3)


    measurements = tc_array.get_measurements()
    print(f'\nMeasurements for sensor at top of block:\n{measurements[-1,0,:]}\n')

    pv_plot = pyv.plot_point_sensors_on_sim(tc_array,field_name)
    pv_plot.camera_position = [(59.354, 43.428, 69.946),
                                (-2.858, 13.189, 4.523),
                                (-0.215, 0.948, -0.233)]
    pv_plot.show()


if __name__ == '__main__':
    main()
