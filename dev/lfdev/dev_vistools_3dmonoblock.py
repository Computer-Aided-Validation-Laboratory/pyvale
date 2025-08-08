'''
================================================================================
Example: 3d thermocouples on a monoblock

pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import numpy as np
import pyvale.mooseherder as mh
import pyvale


def main() -> None:
    """Basics
    """
    # Use mooseherder to read the exodus and get a SimData object
    data_path = Path('src/data/case16_out.e')
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_name = 'temperature'
    # Scale m to mm to make 3D visualisation scaling correct for pyvista
    sim_data = pyv.scale_length_units(1000.0,sim_data)

    pyvale.print_dimensions(sim_data)

    n_sens = (1,4,1)
    x_lims = (12.5,12.5)
    y_lims = (0,33.0)
    z_lims = (0.0,12.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sens_data = pyvale.SensorData(positions=sens_pos)

    tc_array = pyvale.SensorArrayFactory() \
        .thermocouples_basic_errs(sim_data,
                                  sens_data,
                                  field_name,
                                  elem_dims=3)

    measurements = tc_array.get_measurements()
    print(f'\nMeasurements for sensor at top of block:\n{measurements[-1,0,:]}\n')

    vis_opts = pyvale.VisOptsSimSensors()
    vis_opts.window_size_px = (1200,800)
    vis_opts.camera_position = np.array([(59.354, 43.428, 69.946),
                                         (-2.858, 13.189, 4.523),
                                         (-0.215, 0.948, -0.233)])
    image_save_opts = pyvale.VisOptsImageSave()
    image_save_opts.path = Path.cwd() / "dev" / "test_output" / "test_image"
    image_save_opts.image_type = pyvale.EImageType.SVG

    anim_opts = pyvale.VisOptsAnimation()
    anim_opts.save_path = Path.cwd() / "dev" / "test_output" / "test_animation"
    anim_opts.save_animation = pyvale.EAnimationType.MP4

    pv_plot = pyvale.plot_point_sensors_on_sim(tc_array,
                                              field_name,
                                              time_step=-1,
                                              vis_opts=vis_opts,
                                              image_save_opts=None)
    pv_plot.show()

    # pv_anim = pyvale.animate_sim_with_sensors(tc_array,
    #                                           field_name,
    #                                           time_steps=None,
    #                                           vis_opts=vis_opts,
    #                                           anim_opts=anim_opts)



if __name__ == '__main__':
    main()
