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
import pyvale as pyv


def main() -> None:
    """pyvale example: visualisation tools 3D
    """
    # Use mooseherder to read the exodus and get a SimData object
    data_path = pyv.DataSet.thermal_3d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_name = 'temperature'
    # Scale m to mm to make 3D visualisation scaling correct for pyvista
    sim_data = pyv.scale_length_units(1000.0,sim_data)

    pyv.print_dimensions(sim_data)

    n_sens = (1,4,1)
    x_lims = (12.5,12.5)
    y_lims = (0,33.0)
    z_lims = (0.0,12.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sens_data = pyv.SensorData(positions=sens_pos)

    tc_array = pyv.SensorArrayFactory() \
        .thermocouples_basic_errs(sim_data,
                                  sens_data,
                                  field_name,
                                  spat_dims=3)

    measurements = tc_array.get_measurements()
    print(f'\nMeasurements for sensor at top of block:\n{measurements[-1,0,:]}\n')

    vis_opts = pyv.VisOptsSimSensors()
    vis_opts.window_size_px = (1200,800)
    vis_opts.camera_position = np.array([(59.354, 43.428, 69.946),
                                         (-2.858, 13.189, 4.523),
                                         (-0.215, 0.948, -0.233)])

    vis_mode = "vector"

    save_path = Path.cwd()/"pyvale-output"
    if not save_path.is_dir():
        save_path.mkdir(parents=True, exist_ok=True)

    if vis_mode == "animate":
        anim_opts = pyv.VisOptsAnimation()

        anim_opts.save_path = save_path / "test_animation"
        anim_opts.save_animation = pyv.EAnimationType.MP4

        pv_anim = pyv.animate_sim_with_sensors(tc_array,
                                                  field_name,
                                                  time_steps=None,
                                                  vis_opts=vis_opts,
                                                  anim_opts=anim_opts)

    else:
        image_save_opts = pyv.VisOptsImageSave()

        image_save_opts.path = save_path / "test_vector_graphics"
        image_save_opts.image_type = pyv.EImageType.SVG

        pv_plot = pyv.plot_point_sensors_on_sim(tc_array,
                                                field_name,
                                                time_step=-1,
                                                vis_opts=vis_opts,
                                                image_save_opts=image_save_opts)
        pv_plot.show()


if __name__ == '__main__':
    main()
