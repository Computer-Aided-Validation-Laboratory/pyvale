"""
================================================================================
Example: displacement sensors on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import mooseherder as mh
import pyvale

def main() -> None:
    """pyvale example: single rotation batch processing
    ----------------------------------------------------------------------------
    - Tests that when only one sensor rotation is provided that
      all sensors are assumed to have the same rotation and batch processed.
    """
    data_path = pyvale.DataSet.mechanical_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    descriptor = pyvale.SensorDescriptorFactory.displacement_descriptor()

    spat_dims = 2
    field_key = "disp"
    components = ("disp_x","disp_y")
    disp_field = pyvale.FieldVector(sim_data,field_key,components,spat_dims)

    #---------------------------------------------------------------------------
    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sensor_positions = pyvale.create_sensor_pos_array(n_sens,
                                                      x_lims,
                                                      y_lims,
                                                      z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    # Provide only a single rotation for the 6 sensors
    sensor_angles = (Rotation.from_euler("zyx", [180, 0, 0], degrees=True),)

    sensor_data_norot = pyvale.SensorData(positions=sensor_positions,
                                          sample_times=sample_times)


    sensor_data_rot = pyvale.SensorData(positions=sensor_positions,
                                  sample_times=sample_times,
                                  angles=sensor_angles)

    #---------------------------------------------------------------------------
    disp_sensors_norot = pyvale.SensorArrayPoint(sensor_data_norot,
                                                disp_field,
                                                descriptor)


    disp_sensors_rot = pyvale.SensorArrayPoint(sensor_data_rot,
                                               disp_field,
                                               descriptor)



    measurements_norot = disp_sensors_norot.calc_measurements()
    measurements_rot = disp_sensors_rot.calc_measurements()

    #---------------------------------------------------------------------------
    sens_to_print = 1
    print(80*"-")
    print(f"The last 5 time steps (measurements) of non-rotated sensor {sens_to_print}:")
    pyvale.print_measurements(disp_sensors_norot,
                              (sens_to_print-1,sens_to_print),
                              (0,1),
                              (measurements_norot.shape[2]-5,measurements_norot.shape[2]))
    print(80*"-")
    print(f"The last 5 time steps (measurements) of rotated sensor {sens_to_print}:")
    pyvale.print_measurements(disp_sensors_rot,
                              (sens_to_print-1,sens_to_print),
                              (0,1),
                              (measurements_rot.shape[2]-5,measurements_rot.shape[2]))
    print(80*"-")

    #---------------------------------------------------------------------------
    plot_field = "disp_x"

    if plot_field == "disp_x":
        pv_plot = pyvale.plot_point_sensors_on_sim(disp_sensors_rot,"disp_x")
        pv_plot.show(cpos="xy")
    elif plot_field == "disp_y":
        pv_plot = pyvale.plot_point_sensors_on_sim(disp_sensors_rot,"disp_y")
        pv_plot.show(cpos="xy")

    (fig,ax) = pyvale.plot_time_traces(disp_sensors_norot,plot_field)
    ax.set_title("No rotation")
    (fig,ax) = pyvale.plot_time_traces(disp_sensors_rot,plot_field)
    ax.set_title("Rotated")
    plt.show()


if __name__ == "__main__":
    main()