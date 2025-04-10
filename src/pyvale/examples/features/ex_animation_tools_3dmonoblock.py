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
import mooseherder as mh
import pyvale


def main() -> None:
    """pyvale example: visualisation tools 3D
    """
    # Use mooseherder to read the exodus and get a SimData object
    data_path = pyvale.DataSet.thermal_3d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_name = 'temperature'
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

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
                                  spat_dims=3)

    measurements = tc_array.get_measurements()
    print(f'\nMeasurements for sensor at top of block:\n{measurements[-1,0,:]}\n')

    vis_opts = pyvale.VisOptsSimSensors()
    vis_opts.window_size_px = (1200,800)
    vis_opts.camera_position = np.array([(59.354, 43.428, 69.946),
                                         (-2.858, 13.189, 4.523),
                                         (-0.215, 0.948, -0.233)])

    vis_mode = "vector"
    save_dir = Path.cwd() / "exampleoutput"
    if not save_dir.is_dir():
        save_dir.mkdir()

    if vis_mode == "animate":
        anim_opts = pyvale.VisOptsAnimation()

        anim_opts.save_path = save_dir / "test_animation"
        anim_opts.save_animation = pyvale.EAnimationType.MP4

        pv_anim = pyvale.animate_sim_with_sensors(tc_array,
                                                  field_name,
                                                  time_steps=None,
                                                  vis_opts=vis_opts,
                                                  anim_opts=anim_opts)

    else:
        image_save_opts = pyvale.VisOptsImageSave()

        image_save_opts.path = save_dir / "test_vector_graphics"
        image_save_opts.image_type = pyvale.EImageType.SVG

        pv_plot = pyvale.plot_point_sensors_on_sim(tc_array,
                                                field_name,
                                                time_step=-1,
                                                vis_opts=vis_opts,
                                                image_save_opts=image_save_opts)
        pv_plot.show()


if __name__ == '__main__':
    main()
