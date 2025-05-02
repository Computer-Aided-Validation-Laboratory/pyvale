# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import mooseherder as mh
import pyvale as pyv

def main() -> None:
    """pyvale example: displacement sensors on a 2D plate with a hole
    ----------------------------------------------------------------------------
    """
    data_path = pyv.DataSet.mechanical_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    descriptor = pyv.SensorDescriptorFactory.displacement_descriptor()

    spat_dims = 2
    field_key = "disp"
    components = ("disp_x","disp_y")
    disp_field = pyv.FieldVector(sim_data,field_key,components,spat_dims)

    #---------------------------------------------------------------------------
    n_sens = (2,2,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sensor_positions = pyv.create_sensor_pos_array(n_sens,
                                                      x_lims,
                                                      y_lims,
                                                      z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sensor_angles = sensor_positions.shape[0] * \
        (Rotation.from_euler("zyx", [0, 0, 0], degrees=True),)

    sensor_data = pyv.SensorData(positions=sensor_positions,
                                  sample_times=sample_times,
                                  angles=sensor_angles,
                                  spatial_averager=pyv.EIntSpatialType.QUAD4PT,
                                  spatial_dims=np.array([5.0,5.0,0.0]))

    #---------------------------------------------------------------------------
    disp_sensors = pyv.SensorArrayPoint(sensor_data,
                                           disp_field,
                                           descriptor)

    pos_offset = -10.0*np.ones_like(sensor_positions)
    pos_offset[:,2] = 0 # in 2d we only have offset in x and y so zero z

    angle_offset = np.zeros_like(sensor_positions)
    angle_offset[:,0] = 5.0 # only rotate about z in 2D

    time_offset = 1.0*np.ones_like(disp_sensors.get_sample_times())

    field_error_data = pyv.ErrFieldData(pos_offset_xyz=pos_offset,
                                           ang_offset_zyx=angle_offset,
                                           time_offset=time_offset)

    error_chain = []
    error_chain.append(pyv.ErrSysField(disp_field,field_error_data))
    error_integrator = pyv.ErrIntegrator(error_chain,
                                            sensor_data,
                                            disp_sensors.get_measurement_shape())

    disp_sensors.set_error_integrator(error_integrator)

    measurements = disp_sensors.calc_measurements()

    #---------------------------------------------------------------------------
    print(80*"-")
    sens_num = 4
    print("The last 5 time steps (measurements) of sensor {sens_num}:")
    pyv.print_measurements(disp_sensors,
                              (sens_num-1,sens_num),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))
    print(80*"-")

    #---------------------------------------------------------------------------
    plot_field = "disp_x"

    pv_plot = pyv.plot_point_sensors_on_sim(disp_sensors,plot_field)
    pv_plot.show(cpos="xy")

    pyv.plot_time_traces(disp_sensors,plot_field)
    plt.show()


if __name__ == "__main__":
    main()